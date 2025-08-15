#!/bin/bash
# PostgreSQL Setup Script for Production

echo "🐘 PostgreSQL Production Setup"
echo "=============================="

# Run migrations
echo "📊 Running database migrations..."
python3 << 'PYTHON'
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'blyandb',
    'user': 'blyan',
    'password': 'password'  # Update with actual password
}

def create_tables():
    """Create necessary tables if they don't exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Create tables
    tables = [
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id VARCHAR(255) PRIMARY KEY,
            key_hash VARCHAR(255) NOT NULL,
            key_type VARCHAR(50) NOT NULL,
            name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS node_registry (
            node_id VARCHAR(255) PRIMARY KEY,
            host VARCHAR(255) NOT NULL,
            port INTEGER NOT NULL,
            available_experts TEXT[],
            last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_balances (
            address VARCHAR(255) PRIMARY KEY,
            balance DECIMAL(20, 8) DEFAULT 0,
            total_earned DECIMAL(20, 8) DEFAULT 0,
            total_spent DECIMAL(20, 8) DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(255) UNIQUE,
            user_address VARCHAR(255),
            prompt_hash VARCHAR(64),
            tokens_generated INTEGER,
            cost DECIMAL(10, 8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    for table_sql in tables:
        try:
            cur.execute(table_sql)
            print(f"✓ Table created/verified")
        except Exception as e:
            print(f"⚠ Table creation warning: {e}")
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_node_registry_active ON node_registry(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_user_balances_address ON user_balances(address)",
        "CREATE INDEX IF NOT EXISTS idx_inference_logs_user ON inference_logs(user_address)"
    ]
    
    for index_sql in indexes:
        try:
            cur.execute(index_sql)
            print(f"✓ Index created/verified")
        except Exception as e:
            print(f"⚠ Index creation warning: {e}")
    
    conn.close()
    print("✅ Database tables created/verified")

if __name__ == "__main__":
    create_tables()
PYTHON

# Enable WAL archiving
echo ""
echo "📁 Enabling WAL archiving..."
sudo -u postgres psql << SQL
ALTER SYSTEM SET wal_level = replica;
ALTER SYSTEM SET archive_mode = on;
ALTER SYSTEM SET archive_command = 'test ! -f /backup/postgres/wal/%f && cp %p /backup/postgres/wal/%f';
ALTER SYSTEM SET max_wal_size = '1GB';
ALTER SYSTEM SET min_wal_size = '80MB';
SELECT pg_reload_conf();
SQL

# Create backup directories
echo "📁 Creating backup directories..."
sudo mkdir -p /backup/postgres/wal
sudo mkdir -p /backup/postgres/daily
sudo chown -R postgres:postgres /backup/postgres

# Set up daily backup cron
echo "⏰ Setting up daily backup cron..."
cat > /tmp/postgres_backup.sh << 'BACKUP'
#!/bin/bash
BACKUP_DIR="/backup/postgres/daily"
DB_NAME="blyandb"
DATE=$(date +%Y%m%d)
BACKUP_FILE="$BACKUP_DIR/blyan_$DATE.dump"

# Perform backup
pg_dump -U blyan -d $DB_NAME -Fc > $BACKUP_FILE

# Keep only last 7 days
find $BACKUP_DIR -name "*.dump" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
BACKUP

sudo mv /tmp/postgres_backup.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/postgres_backup.sh
echo "0 3 * * * postgres /usr/local/bin/postgres_backup.sh" | sudo tee /etc/cron.d/postgres_backup

# Verify setup
echo ""
echo "🔍 Verifying PostgreSQL setup..."
echo "Tables:"
psql -U blyan -d blyandb -c '\dt'
echo ""
echo "Current time test:"
psql -U blyan -d blyandb -c 'SELECT NOW()'
echo ""
echo "WAL settings:"
psql -U blyan -d blyandb -c "SHOW wal_level; SHOW archive_mode;"

echo ""
echo "✅ PostgreSQL setup complete!"