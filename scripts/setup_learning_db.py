#!/usr/bin/env python3
"""
Setup script for PostgreSQL learning database
Creates database, tables, and initial data
"""

import os
import sys
import asyncio
import asyncpg
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def create_database():
    """Create the learning database if it doesn't exist"""
    
    # Database connection parameters
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    db_name = os.getenv("LEARNING_DB_NAME", "blyan_learning")
    
    # Connect to default postgres database
    conn_str = f"postgresql://{user}"
    if password:
        conn_str = f"postgresql://{user}:{password}"
    conn_str += f"@{host}:{port}/postgres"
    
    print(f"Connecting to PostgreSQL at {host}:{port}")
    
    try:
        conn = await asyncpg.connect(conn_str)
        
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)",
            db_name
        )
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✅ Database '{db_name}' created")
        else:
            print(f"Database '{db_name}' already exists")
        
        await conn.close()
        
        # Now connect to the new database and create schema
        db_conn_str = conn_str.replace("/postgres", f"/{db_name}")
        return db_conn_str
        
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        raise


async def setup_tables(conn_str: str):
    """Create all required tables"""
    
    print("Setting up tables...")
    
    conn = await asyncpg.connect(conn_str)
    
    try:
        # Read schema file
        schema_path = Path(__file__).parent.parent / "backend" / "database" / "learning_schema.sql"
        
        if schema_path.exists():
            print(f"Reading schema from {schema_path}")
            with open(schema_path, 'r') as f:
                schema = f.read()
            
            # Execute the entire schema as one block
            # This preserves function definitions and complex statements
            try:
                await conn.execute(schema)
                print("✅ Tables created successfully")
            except Exception as e:
                print(f"  Error executing schema: {e}")
                print("  Falling back to minimal schema...")
        else:
            print(f"⚠️  Schema file not found at {schema_path}")
            print("Creating minimal schema...")
            
            # Create minimal tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS burns_ledger (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                    user_addr TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    bly_amount NUMERIC(38, 18) NOT NULL,
                    round_candidate BOOLEAN DEFAULT FALSE
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_rounds (
                    round_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    state TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    bly_sum NUMERIC(38,18) NOT NULL,
                    min_interval_ok BOOLEAN NOT NULL,
                    seed BYTEA NOT NULL,
                    config JSONB NOT NULL,
                    target_expert TEXT,
                    base_version TEXT
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS gpu_nodes (
                    node_id TEXT PRIMARY KEY,
                    base_url TEXT NOT NULL,
                    pubkey TEXT NOT NULL,
                    last_seen TIMESTAMPTZ,
                    capabilities JSONB,
                    status TEXT DEFAULT 'active',
                    reputation_score NUMERIC DEFAULT 1.0
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS data_assignments (
                    round_id UUID,
                    node_id TEXT,
                    shard_id TEXT NOT NULL,
                    shard_uri TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    assigned_at TIMESTAMPTZ DEFAULT now(),
                    fetched_at TIMESTAMPTZ,
                    PRIMARY KEY(round_id, node_id, shard_id)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS deltas (
                    round_id UUID,
                    node_id TEXT,
                    delta_id TEXT NOT NULL,
                    parent_model_hash TEXT NOT NULL,
                    delta_hash TEXT NOT NULL,
                    train_meta JSONB NOT NULL,
                    metrics JSONB NOT NULL,
                    proof_hash TEXT,
                    sig TEXT NOT NULL,
                    submitted_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY(round_id, node_id, delta_id)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    key TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    response JSONB,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    expires_at TIMESTAMPTZ DEFAULT now() + INTERVAL '24 hours'
                )
            """)
            
            print("✅ Minimal schema created")
        
    finally:
        await conn.close()


async def add_test_data(conn_str: str):
    """Add some test data for development"""
    
    print("Adding test data...")
    
    conn = await asyncpg.connect(conn_str)
    
    try:
        # Add test GPU nodes
        test_nodes = [
            ("gpu_node_001", "http://localhost:8001", "pubkey_001", {"gpu": "RTX 3090", "vram_gb": 24}),
            ("gpu_node_002", "http://localhost:8002", "pubkey_002", {"gpu": "RTX 4090", "vram_gb": 24}),
            ("gpu_node_003", "http://localhost:8003", "pubkey_003", {"gpu": "A100", "vram_gb": 40}),
        ]
        
        for node_id, base_url, pubkey, capabilities in test_nodes:
            await conn.execute("""
                INSERT INTO gpu_nodes (node_id, base_url, pubkey, capabilities, last_seen)
                VALUES ($1, $2, $3, $4::jsonb, now())
                ON CONFLICT (node_id) DO UPDATE
                SET base_url = $2, last_seen = now()
            """, node_id, base_url, pubkey, json.dumps(capabilities))
        
        print(f"  Added {len(test_nodes)} test GPU nodes")
        
        # Add some test burns
        await conn.execute("""
            INSERT INTO burns_ledger (user_addr, request_id, bly_amount)
            VALUES 
                ('user_001', 'req_001', 100.5),
                ('user_002', 'req_002', 200.0),
                ('user_003', 'req_003', 150.75)
        """)
        
        print("  Added test burn records")
        
        print("✅ Test data added")
        
    finally:
        await conn.close()


async def verify_setup(conn_str: str):
    """Verify the database setup"""
    
    print("\nVerifying setup...")
    
    conn = await asyncpg.connect(conn_str)
    
    try:
        # Check tables
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        
        print(f"\n📊 Tables created ({len(tables)}):")
        for table in tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['tablename']}")
            print(f"  - {table['tablename']}: {count} rows")
        
        # Check burn accumulator
        total = await conn.fetchval("""
            SELECT COALESCE(SUM(bly_amount), 0) 
            FROM burns_ledger 
            WHERE round_candidate = FALSE
        """)
        
        print(f"\n🔥 Burn accumulator: {total} BLY")
        
        # Check nodes
        nodes = await conn.fetchval("SELECT COUNT(*) FROM gpu_nodes WHERE status = 'active'")
        print(f"🖥️  Active GPU nodes: {nodes}")
        
    finally:
        await conn.close()


async def main():
    """Main setup function"""
    
    print("=" * 60)
    print("🚀 Blyan Learning Database Setup")
    print("=" * 60)
    
    try:
        # Create database
        conn_str = await create_database()
        
        # Setup tables
        await setup_tables(conn_str)
        
        # Add test data (optional)
        if "--test-data" in sys.argv:
            await add_test_data(conn_str)
        
        # Verify setup
        await verify_setup(conn_str)
        
        print("\n" + "=" * 60)
        print("✨ Database setup complete!")
        print("=" * 60)
        
        print("\n📝 Connection string for .env file:")
        print(f"DATABASE_URL={conn_str}")
        
        print("\n🎯 Next steps:")
        print("1. Add DATABASE_URL to your .env file")
        print("2. Start the API server: ./server.sh start")
        print("3. Monitor learning metrics: curl http://localhost:8000/learning/metrics")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())