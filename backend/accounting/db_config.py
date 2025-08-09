#!/usr/bin/env python3
"""
PostgreSQL Database Configuration for Ledger System
Production-ready database connection management
"""

import os
import asyncpg
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration and connection management."""
    
    def __init__(self):
        # Database connection parameters from environment
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "blyan_ledger")
        self.user = os.getenv("POSTGRES_USER", "blyan_user")
        self.password = os.getenv("POSTGRES_PASSWORD", "")
        self.schema = os.getenv("POSTGRES_SCHEMA", "ledger")
        
        # Connection pool settings
        self.min_pool_size = int(os.getenv("DB_MIN_POOL_SIZE", "2"))
        self.max_pool_size = int(os.getenv("DB_MAX_POOL_SIZE", "10"))
        self.max_queries = int(os.getenv("DB_MAX_QUERIES", "50000"))
        self.max_inactive_connection_lifetime = float(
            os.getenv("DB_MAX_INACTIVE_LIFETIME", "300")
        )
        
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection pool."""
        if self._initialized:
            return
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
                command_timeout=60
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Connected to PostgreSQL: {version[:50]}...")
                
                # Set default schema
                await conn.execute(f"SET search_path TO {self.schema}, public")
            
            self._initialized = True
            logger.info(f"Database pool initialized with {self.min_pool_size}-{self.max_pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a database connection from pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            # Set schema for this connection
            await conn.execute(f"SET search_path TO {self.schema}, public")
            yield conn
    
    @asynccontextmanager
    async def transaction(self):
        """Start a database transaction."""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query without returning results."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetchone(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchall(self, query: str, *args) -> list:
        """Fetch all rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)


# Global database instance
db = DatabaseConfig()


async def init_database():
    """Initialize database and run migrations."""
    await db.initialize()
    
    # Check if schema exists
    async with db.acquire() as conn:
        schema_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = $1)",
            db.schema
        )
        
        if not schema_exists:
            logger.info(f"Creating schema {db.schema}")
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {db.schema}")
            
            # Run migration script
            migration_path = "migrations/001_create_ledger.sql"
            if os.path.exists(migration_path):
                logger.info("Running database migration...")
                with open(migration_path, 'r') as f:
                    migration_sql = f.read()
                    await conn.execute(migration_sql)
                logger.info("Database migration completed")
            else:
                logger.warning(f"Migration file not found: {migration_path}")
        else:
            logger.info(f"Schema {db.schema} already exists")


async def test_connection():
    """Test database connection."""
    try:
        await db.initialize()
        
        # Test query
        result = await db.fetchval("SELECT COUNT(*) FROM ledger.accounts")
        logger.info(f"Found {result} accounts in database")
        
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
    finally:
        await db.close()


if __name__ == "__main__":
    # Test database connection
    asyncio.run(test_connection())