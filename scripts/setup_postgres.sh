#!/bin/bash

# PostgreSQL Setup Script for Blyan Network Ledger
# This script sets up PostgreSQL database for the ledger system

set -e

echo "🚀 Setting up PostgreSQL for Blyan Network Ledger"
echo "================================================"

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed. Please install it first."
    echo "   macOS: brew install postgresql"
    echo "   Ubuntu: sudo apt-get install postgresql postgresql-contrib"
    echo "   CentOS: sudo yum install postgresql-server postgresql-contrib"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  .env file not found. Using defaults..."
    POSTGRES_DB=${POSTGRES_DB:-blyan_ledger}
    POSTGRES_USER=${POSTGRES_USER:-blyan_user}
    POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-blyan_secure_password}
fi

echo "📝 Database Configuration:"
echo "   Database: $POSTGRES_DB"
echo "   User: $POSTGRES_USER"
echo ""

# Create database and user
echo "🔧 Creating database and user..."

# Check if we can connect as postgres user
if sudo -u postgres psql -c '\q' 2>/dev/null; then
    # Linux style (postgres user exists)
    sudo -u postgres psql <<EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$POSTGRES_USER') THEN
        CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';
    END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$POSTGRES_DB')\gexec

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
EOF
else
    # macOS style (current user)
    psql postgres <<EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$POSTGRES_USER') THEN
        CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';
    END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$POSTGRES_DB')\gexec

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
EOF
fi

echo "✅ Database and user created"

# Run migrations
echo "🔄 Running migrations..."
PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U $POSTGRES_USER -d $POSTGRES_DB < migrations/001_create_ledger.sql

echo "✅ Migrations completed"

# Test connection
echo "🧪 Testing database connection..."
python3 - <<EOF
import asyncio
import os
os.environ['POSTGRES_PASSWORD'] = '$POSTGRES_PASSWORD'

async def test():
    from backend.accounting.db_config import test_connection
    success = await test_connection()
    if success:
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")
        exit(1)

asyncio.run(test())
EOF

echo ""
echo "🎉 PostgreSQL setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure your .env file has the correct database credentials"
echo "   2. Restart the API server: ./server.sh restart api"
echo "   3. Test wallet endpoints:"
echo "      curl http://localhost:8000/wallet/balance/test_address"
echo "      curl http://localhost:8000/wallet/transactions/test_address"
echo ""
echo "🔐 Security reminder:"
echo "   - Use a strong password in production"
echo "   - Enable SSL for database connections"
echo "   - Set up regular backups"