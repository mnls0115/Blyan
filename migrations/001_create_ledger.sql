-- Blyan Network Ledger Tables
-- Double-entry bookkeeping with immutability and audit trail

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schema for ledger
CREATE SCHEMA IF NOT EXISTS ledger;

-- Account types enum
CREATE TYPE ledger.account_type AS ENUM (
    'user_wallet',       -- Individual user BLY balance
    'validation_pool',   -- Pool funded by Stripe payments
    'reward_pool',       -- Pool for validator rewards
    'treasury',          -- Platform operational funds
    'escrow',           -- Held during validation
    'stripe_gateway',    -- Stripe payment gateway account
    'burn_address'       -- Tokens burned (removed from circulation)
);

-- Transaction status enum
CREATE TYPE ledger.transaction_status AS ENUM (
    'pending',
    'completed',
    'failed',
    'reversed'
);

-- Accounts table
CREATE TABLE IF NOT EXISTS ledger.accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_code VARCHAR(100) UNIQUE NOT NULL,
    account_type ledger.account_type NOT NULL,
    account_name VARCHAR(255) NOT NULL,
    owner_address VARCHAR(42),  -- Ethereum address if applicable
    balance DECIMAL(20,8) DEFAULT 0 CHECK (balance >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'::jsonb,
    INDEX idx_account_type (account_type),
    INDEX idx_owner_address (owner_address)
);

-- Ledger entries table (immutable)
CREATE TABLE IF NOT EXISTS ledger.entries (
    id BIGSERIAL PRIMARY KEY,
    transaction_id UUID NOT NULL,
    account_id UUID NOT NULL REFERENCES ledger.accounts(id),
    debit DECIMAL(20,8) CHECK (debit >= 0),
    credit DECIMAL(20,8) CHECK (credit >= 0),
    balance_before DECIMAL(20,8) NOT NULL,
    balance_after DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT check_debit_or_credit CHECK (
        (debit IS NOT NULL AND credit IS NULL) OR 
        (debit IS NULL AND credit IS NOT NULL)
    ),
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_account_id (account_id),
    INDEX idx_created_at (created_at DESC)
);

-- Transactions table
CREATE TABLE IF NOT EXISTS ledger.transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idempotency_key VARCHAR(255) UNIQUE,
    transaction_type VARCHAR(50) NOT NULL,
    status ledger.transaction_status DEFAULT 'pending',
    amount DECIMAL(20,8) NOT NULL CHECK (amount > 0),
    currency VARCHAR(10) DEFAULT 'BLY',
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    reversed_at TIMESTAMP WITH TIME ZONE,
    reversed_by UUID REFERENCES ledger.transactions(id),
    INDEX idx_idempotency_key (idempotency_key),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at DESC)
);

-- Stripe webhooks table (for reconciliation)
CREATE TABLE IF NOT EXISTS ledger.stripe_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stripe_event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payment_intent_id VARCHAR(255),
    amount_cents INTEGER,
    currency VARCHAR(10),
    transaction_id UUID REFERENCES ledger.transactions(id),
    raw_event JSONB NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_stripe_event_id (stripe_event_id),
    INDEX idx_payment_intent_id (payment_intent_id)
);

-- Reward distributions table
CREATE TABLE IF NOT EXISTS ledger.reward_distributions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID REFERENCES ledger.transactions(id),
    validator_address VARCHAR(42) NOT NULL,
    amount DECIMAL(20,8) NOT NULL CHECK (amount > 0),
    reward_type VARCHAR(50) NOT NULL, -- 'validation', 'learning', 'inference'
    quality_score DECIMAL(5,4),
    distributed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_validator_address (validator_address),
    INDEX idx_reward_type (reward_type),
    INDEX idx_distributed_at (distributed_at DESC)
);

-- Audit log table (append-only)
CREATE TABLE IF NOT EXISTS ledger.audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(20) NOT NULL,
    user_id VARCHAR(100),
    row_id UUID,
    old_data JSONB,
    new_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create default accounts
INSERT INTO ledger.accounts (account_code, account_type, account_name) VALUES
    ('VALIDATION_POOL', 'validation_pool', 'Validation Pool'),
    ('REWARD_POOL', 'reward_pool', 'Reward Distribution Pool'),
    ('TREASURY', 'treasury', 'Blyan Treasury'),
    ('STRIPE_GATEWAY', 'stripe_gateway', 'Stripe Payment Gateway'),
    ('BURN_ADDRESS', 'burn_address', 'Token Burn Address')
ON CONFLICT (account_code) DO NOTHING;

-- Function to create user wallet account
CREATE OR REPLACE FUNCTION ledger.create_user_wallet(
    p_address VARCHAR(42),
    p_initial_balance DECIMAL(20,8) DEFAULT 0
) RETURNS UUID AS $$
DECLARE
    v_account_id UUID;
BEGIN
    INSERT INTO ledger.accounts (
        account_code,
        account_type,
        account_name,
        owner_address,
        balance
    ) VALUES (
        LOWER(p_address),
        'user_wallet',
        'User Wallet: ' || SUBSTRING(p_address, 1, 10) || '...',
        LOWER(p_address),
        p_initial_balance
    )
    ON CONFLICT (account_code) DO UPDATE
        SET updated_at = NOW()
    RETURNING id INTO v_account_id;
    
    RETURN v_account_id;
END;
$$ LANGUAGE plpgsql;

-- Function to record transaction (atomic double-entry)
CREATE OR REPLACE FUNCTION ledger.record_transaction(
    p_idempotency_key VARCHAR(255),
    p_transaction_type VARCHAR(50),
    p_amount DECIMAL(20,8),
    p_from_account VARCHAR(100),
    p_to_account VARCHAR(100),
    p_description TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    v_transaction_id UUID;
    v_from_account_id UUID;
    v_to_account_id UUID;
    v_from_balance DECIMAL(20,8);
    v_to_balance DECIMAL(20,8);
BEGIN
    -- Check idempotency
    SELECT id INTO v_transaction_id
    FROM ledger.transactions
    WHERE idempotency_key = p_idempotency_key;
    
    IF FOUND THEN
        RETURN v_transaction_id;
    END IF;
    
    -- Lock accounts for update
    SELECT id, balance INTO v_from_account_id, v_from_balance
    FROM ledger.accounts
    WHERE account_code = p_from_account
    FOR UPDATE;
    
    SELECT id, balance INTO v_to_account_id, v_to_balance
    FROM ledger.accounts
    WHERE account_code = p_to_account
    FOR UPDATE;
    
    -- Check sufficient balance
    IF v_from_balance < p_amount THEN
        RAISE EXCEPTION 'Insufficient balance in account %', p_from_account;
    END IF;
    
    -- Create transaction record
    INSERT INTO ledger.transactions (
        idempotency_key,
        transaction_type,
        amount,
        description,
        metadata,
        status
    ) VALUES (
        p_idempotency_key,
        p_transaction_type,
        p_amount,
        p_description,
        p_metadata,
        'pending'
    ) RETURNING id INTO v_transaction_id;
    
    -- Record debit entry
    INSERT INTO ledger.entries (
        transaction_id,
        account_id,
        debit,
        balance_before,
        balance_after
    ) VALUES (
        v_transaction_id,
        v_from_account_id,
        p_amount,
        v_from_balance,
        v_from_balance - p_amount
    );
    
    -- Record credit entry
    INSERT INTO ledger.entries (
        transaction_id,
        account_id,
        credit,
        balance_before,
        balance_after
    ) VALUES (
        v_transaction_id,
        v_to_account_id,
        p_amount,
        v_to_balance,
        v_to_balance + p_amount
    );
    
    -- Update account balances
    UPDATE ledger.accounts 
    SET balance = balance - p_amount, updated_at = NOW()
    WHERE id = v_from_account_id;
    
    UPDATE ledger.accounts 
    SET balance = balance + p_amount, updated_at = NOW()
    WHERE id = v_to_account_id;
    
    -- Mark transaction as completed
    UPDATE ledger.transactions
    SET status = 'completed', completed_at = NOW()
    WHERE id = v_transaction_id;
    
    RETURN v_transaction_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger for audit logging
CREATE OR REPLACE FUNCTION ledger.audit_trigger() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO ledger.audit_log (
        table_name,
        operation,
        row_id,
        old_data,
        new_data
    ) VALUES (
        TG_TABLE_NAME,
        TG_OP,
        COALESCE(NEW.id, OLD.id),
        to_jsonb(OLD),
        to_jsonb(NEW)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers
CREATE TRIGGER audit_accounts
    AFTER INSERT OR UPDATE OR DELETE ON ledger.accounts
    FOR EACH ROW EXECUTE FUNCTION ledger.audit_trigger();

CREATE TRIGGER audit_transactions
    AFTER INSERT OR UPDATE OR DELETE ON ledger.transactions
    FOR EACH ROW EXECUTE FUNCTION ledger.audit_trigger();

-- Create indexes for performance
CREATE INDEX idx_entries_created_at_desc ON ledger.entries(created_at DESC);
CREATE INDEX idx_transactions_metadata ON ledger.transactions USING GIN(metadata);
CREATE INDEX idx_accounts_metadata ON ledger.accounts USING GIN(metadata);

-- Grant permissions (adjust for your user)
GRANT USAGE ON SCHEMA ledger TO blyan_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ledger TO blyan_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ledger TO blyan_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ledger TO blyan_user;