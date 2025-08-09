-- PostgreSQL Ledger Schema for Blyan Network
-- Version: 1.0.0
-- Date: 2025-01-08

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- User Balances Table
-- =====================================================
CREATE TABLE IF NOT EXISTS user_balances (
    user_address VARCHAR(42) PRIMARY KEY,
    balance DECIMAL(36, 18) NOT NULL DEFAULT 0 CHECK (balance >= 0),
    locked_balance DECIMAL(36, 18) NOT NULL DEFAULT 0 CHECK (locked_balance >= 0),
    total_earned DECIMAL(36, 18) NOT NULL DEFAULT 0,
    total_spent DECIMAL(36, 18) NOT NULL DEFAULT 0,
    trust_level VARCHAR(20) DEFAULT 'NEWCOMER',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for performance
CREATE INDEX idx_user_balances_updated_at ON user_balances(updated_at DESC);
CREATE INDEX idx_user_balances_trust_level ON user_balances(trust_level);

-- =====================================================
-- Transactions Table with State Machine
-- =====================================================
CREATE TYPE transaction_type AS ENUM (
    'charge',
    'refund', 
    'reward',
    'transfer',
    'stake',
    'unstake',
    'slash'
);

CREATE TYPE transaction_status AS ENUM (
    'pending',
    'quoted',
    'authorized',
    'captured',
    'credited',
    'failed',
    'reversed'
);

CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idempotency_key VARCHAR(255) UNIQUE,
    user_address VARCHAR(42) NOT NULL REFERENCES user_balances(user_address),
    
    -- Transaction details
    tx_type transaction_type NOT NULL,
    status transaction_status NOT NULL DEFAULT 'pending',
    amount DECIMAL(36, 18) NOT NULL,
    fee DECIMAL(36, 18) DEFAULT 0,
    
    -- Quote tracking
    quote_id VARCHAR(100),
    quoted_amount DECIMAL(36, 18),
    quote_expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    description TEXT,
    metadata JSONB DEFAULT '{}',
    error_message TEXT,
    
    -- Related entities
    related_tx_id UUID REFERENCES transactions(id),
    block_hash VARCHAR(64),
    expert_name VARCHAR(100),
    
    -- Timestamps for state transitions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    quoted_at TIMESTAMP WITH TIME ZONE,
    authorized_at TIMESTAMP WITH TIME ZONE,
    captured_at TIMESTAMP WITH TIME ZONE,
    credited_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE,
    reversed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX idx_transactions_user_address ON transactions(user_address);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX idx_transactions_idempotency ON transactions(idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX idx_transactions_quote_id ON transactions(quote_id) WHERE quote_id IS NOT NULL;

-- =====================================================
-- Ledger Entries (Double-Entry Bookkeeping)
-- =====================================================
CREATE TABLE IF NOT EXISTS ledger_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    account_type VARCHAR(50) NOT NULL, -- 'user', 'system', 'reward_pool', 'fee_pool'
    account_id VARCHAR(100) NOT NULL,
    debit DECIMAL(36, 18) DEFAULT 0 CHECK (debit >= 0),
    credit DECIMAL(36, 18) DEFAULT 0 CHECK (credit >= 0),
    balance_after DECIMAL(36, 18) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT check_debit_or_credit CHECK (
        (debit > 0 AND credit = 0) OR 
        (credit > 0 AND debit = 0)
    )
);

-- Indexes for ledger
CREATE INDEX idx_ledger_entries_transaction_id ON ledger_entries(transaction_id);
CREATE INDEX idx_ledger_entries_account ON ledger_entries(account_type, account_id);
CREATE INDEX idx_ledger_entries_created_at ON ledger_entries(created_at DESC);

-- =====================================================
-- Daily Reconciliation Table
-- =====================================================
CREATE TABLE IF NOT EXISTS daily_reconciliations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reconciliation_date DATE NOT NULL UNIQUE,
    
    -- Aggregates
    total_users INTEGER NOT NULL DEFAULT 0,
    total_balance DECIMAL(36, 18) NOT NULL DEFAULT 0,
    total_locked DECIMAL(36, 18) NOT NULL DEFAULT 0,
    total_transactions INTEGER NOT NULL DEFAULT 0,
    total_volume DECIMAL(36, 18) NOT NULL DEFAULT 0,
    
    -- By transaction type
    charges_count INTEGER DEFAULT 0,
    charges_volume DECIMAL(36, 18) DEFAULT 0,
    rewards_count INTEGER DEFAULT 0,
    rewards_volume DECIMAL(36, 18) DEFAULT 0,
    refunds_count INTEGER DEFAULT 0,
    refunds_volume DECIMAL(36, 18) DEFAULT 0,
    
    -- Reconciliation results
    ledger_balance_sum DECIMAL(36, 18) NOT NULL,
    user_balance_sum DECIMAL(36, 18) NOT NULL,
    discrepancy DECIMAL(36, 18) GENERATED ALWAYS AS (ledger_balance_sum - user_balance_sum) STORED,
    
    -- Status
    is_reconciled BOOLEAN DEFAULT FALSE,
    reconciled_at TIMESTAMP WITH TIME ZONE,
    reconciled_by VARCHAR(100),
    notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- Audit Log Table
-- =====================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(50) NOT NULL,
    record_id VARCHAR(100) NOT NULL,
    action VARCHAR(20) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

CREATE INDEX idx_audit_logs_table_record ON audit_logs(table_name, record_id);
CREATE INDEX idx_audit_logs_changed_at ON audit_logs(changed_at DESC);

-- =====================================================
-- Functions and Triggers
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for user_balances
CREATE TRIGGER update_user_balances_updated_at
    BEFORE UPDATE ON user_balances
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to enforce transaction state machine
CREATE OR REPLACE FUNCTION validate_transaction_status_transition()
RETURNS TRIGGER AS $$
BEGIN
    -- Define valid state transitions
    IF OLD.status = 'pending' AND NEW.status NOT IN ('quoted', 'failed') THEN
        RAISE EXCEPTION 'Invalid status transition from pending to %', NEW.status;
    ELSIF OLD.status = 'quoted' AND NEW.status NOT IN ('authorized', 'failed') THEN
        RAISE EXCEPTION 'Invalid status transition from quoted to %', NEW.status;
    ELSIF OLD.status = 'authorized' AND NEW.status NOT IN ('captured', 'failed') THEN
        RAISE EXCEPTION 'Invalid status transition from authorized to %', NEW.status;
    ELSIF OLD.status = 'captured' AND NEW.status NOT IN ('credited', 'failed') THEN
        RAISE EXCEPTION 'Invalid status transition from captured to %', NEW.status;
    ELSIF OLD.status = 'credited' AND NEW.status NOT IN ('reversed') THEN
        RAISE EXCEPTION 'Invalid status transition from credited to %', NEW.status;
    ELSIF OLD.status = 'failed' THEN
        RAISE EXCEPTION 'Cannot transition from failed status';
    END IF;
    
    -- Update transition timestamps
    CASE NEW.status
        WHEN 'quoted' THEN NEW.quoted_at = CURRENT_TIMESTAMP;
        WHEN 'authorized' THEN NEW.authorized_at = CURRENT_TIMESTAMP;
        WHEN 'captured' THEN NEW.captured_at = CURRENT_TIMESTAMP;
        WHEN 'credited' THEN NEW.credited_at = CURRENT_TIMESTAMP;
        WHEN 'failed' THEN NEW.failed_at = CURRENT_TIMESTAMP;
        WHEN 'reversed' THEN NEW.reversed_at = CURRENT_TIMESTAMP;
        ELSE NULL;
    END CASE;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for transaction status validation
CREATE TRIGGER validate_transaction_status
    BEFORE UPDATE OF status ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION validate_transaction_status_transition();

-- Function for atomic balance update
CREATE OR REPLACE FUNCTION update_user_balance(
    p_user_address VARCHAR(42),
    p_amount DECIMAL(36, 18),
    p_operation VARCHAR(20) -- 'add', 'subtract', 'lock', 'unlock'
) RETURNS user_balances AS $$
DECLARE
    v_user user_balances;
BEGIN
    -- Lock the row for update
    SELECT * INTO v_user FROM user_balances 
    WHERE user_address = p_user_address 
    FOR UPDATE;
    
    IF NOT FOUND THEN
        -- Create new user if not exists
        INSERT INTO user_balances (user_address, balance) 
        VALUES (p_user_address, 0)
        RETURNING * INTO v_user;
    END IF;
    
    -- Perform operation
    CASE p_operation
        WHEN 'add' THEN
            UPDATE user_balances 
            SET balance = balance + p_amount,
                total_earned = total_earned + p_amount
            WHERE user_address = p_user_address
            RETURNING * INTO v_user;
            
        WHEN 'subtract' THEN
            IF v_user.balance < p_amount THEN
                RAISE EXCEPTION 'Insufficient balance: % < %', v_user.balance, p_amount;
            END IF;
            
            UPDATE user_balances 
            SET balance = balance - p_amount,
                total_spent = total_spent + p_amount
            WHERE user_address = p_user_address
            RETURNING * INTO v_user;
            
        WHEN 'lock' THEN
            IF v_user.balance < p_amount THEN
                RAISE EXCEPTION 'Insufficient balance to lock: % < %', v_user.balance, p_amount;
            END IF;
            
            UPDATE user_balances 
            SET balance = balance - p_amount,
                locked_balance = locked_balance + p_amount
            WHERE user_address = p_user_address
            RETURNING * INTO v_user;
            
        WHEN 'unlock' THEN
            IF v_user.locked_balance < p_amount THEN
                RAISE EXCEPTION 'Insufficient locked balance: % < %', v_user.locked_balance, p_amount;
            END IF;
            
            UPDATE user_balances 
            SET balance = balance + p_amount,
                locked_balance = locked_balance - p_amount
            WHERE user_address = p_user_address
            RETURNING * INTO v_user;
            
        ELSE
            RAISE EXCEPTION 'Invalid operation: %', p_operation;
    END CASE;
    
    RETURN v_user;
END;
$$ language 'plpgsql';

-- =====================================================
-- Views for Reporting
-- =====================================================

-- User balance summary view
CREATE OR REPLACE VIEW v_user_balance_summary AS
SELECT 
    ub.user_address,
    ub.balance,
    ub.locked_balance,
    ub.trust_level,
    COUNT(DISTINCT t.id) as transaction_count,
    COALESCE(SUM(CASE WHEN t.tx_type = 'charge' THEN t.amount ELSE 0 END), 0) as total_charges,
    COALESCE(SUM(CASE WHEN t.tx_type = 'reward' THEN t.amount ELSE 0 END), 0) as total_rewards,
    MAX(t.created_at) as last_transaction_at
FROM user_balances ub
LEFT JOIN transactions t ON ub.user_address = t.user_address
GROUP BY ub.user_address, ub.balance, ub.locked_balance, ub.trust_level;

-- Transaction daily summary
CREATE OR REPLACE VIEW v_daily_transaction_summary AS
SELECT 
    DATE(created_at) as transaction_date,
    tx_type,
    status,
    COUNT(*) as count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount
FROM transactions
GROUP BY DATE(created_at), tx_type, status
ORDER BY transaction_date DESC, tx_type;

-- =====================================================
-- Initial System Accounts
-- =====================================================
INSERT INTO user_balances (user_address, balance, trust_level) VALUES
    ('SYSTEM_REWARD_POOL', 1000000000, 'SYSTEM'),
    ('SYSTEM_FEE_POOL', 0, 'SYSTEM'),
    ('SYSTEM_TREASURY', 0, 'SYSTEM')
ON CONFLICT (user_address) DO NOTHING;

-- =====================================================
-- Grants (adjust as needed)
-- =====================================================
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO blyan_api;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO blyan_api;