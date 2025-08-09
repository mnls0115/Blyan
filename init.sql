-- Initialize Blyan Database
-- This file is automatically executed when PostgreSQL container starts

-- Include the main ledger migration
\i /docker-entrypoint-initdb.d/001_create_ledger.sql

-- Create additional indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_balance ON ledger.accounts(balance) WHERE is_active = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_amount ON ledger.transactions(amount);

-- Create materialized view for account summaries
CREATE MATERIALIZED VIEW IF NOT EXISTS ledger.account_summary AS
SELECT 
    a.account_code,
    a.account_type,
    a.balance,
    COUNT(e.id) as transaction_count,
    MAX(e.created_at) as last_activity
FROM ledger.accounts a
LEFT JOIN ledger.entries e ON e.account_id = a.id
WHERE a.is_active = true
GROUP BY a.id, a.account_code, a.account_type, a.balance;

-- Create index on materialized view
CREATE UNIQUE INDEX ON ledger.account_summary(account_code);

-- Refresh materialized view (schedule this periodically)
REFRESH MATERIALIZED VIEW CONCURRENTLY ledger.account_summary;

-- Initial data integrity check
DO $$
DECLARE
    v_debit_total DECIMAL(20,8);
    v_credit_total DECIMAL(20,8);
BEGIN
    SELECT COALESCE(SUM(debit), 0), COALESCE(SUM(credit), 0)
    INTO v_debit_total, v_credit_total
    FROM ledger.entries;
    
    IF v_debit_total != v_credit_total THEN
        RAISE WARNING 'Ledger imbalance detected: Debits=%, Credits=%', 
                      v_debit_total, v_credit_total;
    ELSE
        RAISE NOTICE 'Ledger balanced: Total=% BLY', v_debit_total;
    END IF;
END $$;

-- Create periodic refresh function for materialized view
CREATE OR REPLACE FUNCTION ledger.refresh_account_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY ledger.account_summary;
END;
$$ LANGUAGE plpgsql;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE '✅ Blyan Ledger Database initialized successfully';
    RAISE NOTICE '   - Ledger schema created';
    RAISE NOTICE '   - Audit logging enabled';
    RAISE NOTICE '   - Default accounts created';
    RAISE NOTICE '   - Double-entry functions ready';
END $$;