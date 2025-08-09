#!/usr/bin/env python3
"""
Migration script from file-based ledger to PostgreSQL
Zero-downtime migration with data validation
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import logging
import asyncpg

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.accounting.postgres_ledger import get_postgres_ledger, TransactionType, TransactionStatus
from backend.accounting.ledger import get_ledger  # Existing file-based ledger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LedgerMigration:
    """Migrate from file-based ledger to PostgreSQL."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://blyan_user:blyan_pass@localhost/blyan_db"
        )
        self.old_ledger = get_ledger()
        self.new_ledger = get_postgres_ledger()
        self.new_ledger.database_url = self.database_url
        
        # Migration stats
        self.stats = {
            "users_migrated": 0,
            "transactions_migrated": 0,
            "errors": [],
            "discrepancies": []
        }
    
    async def setup_database(self):
        """Create database and run schema migration."""
        logger.info("Setting up PostgreSQL database...")
        
        # Parse connection string
        parts = self.database_url.replace("postgresql://", "").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid database URL")
        
        user_host = parts[0]
        db_name = parts[1].split("?")[0]
        
        # Connect to postgres database to create our database
        admin_url = f"postgresql://{user_host}/postgres"
        
        try:
            conn = await asyncpg.connect(admin_url)
            
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                db_name
            )
            
            if not exists:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Created database: {db_name}")
            else:
                logger.info(f"Database already exists: {db_name}")
            
            await conn.close()
            
            # Now connect to our database and run schema
            await self.new_ledger.connect()
            
            # Read and execute schema
            schema_path = Path(__file__).parent.parent / "migrations" / "001_ledger_schema.sql"
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            async with self.new_ledger.pool.acquire() as conn:
                await conn.execute(schema_sql)
                logger.info("Schema migration completed")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    async def migrate_users(self):
        """Migrate user balances."""
        logger.info("Migrating user balances...")
        
        # Load existing balances from file
        ledger_path = Path("./data/ledger/user_balances.json")
        if not ledger_path.exists():
            logger.warning("No existing user balances found")
            return
        
        with open(ledger_path, 'r') as f:
            user_balances = json.load(f)
        
        async with self.new_ledger.pool.acquire() as conn:
            for user_address, balance_str in user_balances.items():
                try:
                    balance = Decimal(balance_str)
                    
                    # Skip system accounts
                    if user_address.startswith("SYSTEM_"):
                        continue
                    
                    # Insert or update user balance
                    await conn.execute(
                        """
                        INSERT INTO user_balances (
                            user_address, balance, total_earned
                        ) VALUES ($1, $2, $2)
                        ON CONFLICT (user_address) 
                        DO UPDATE SET 
                            balance = EXCLUDED.balance,
                            total_earned = GREATEST(user_balances.total_earned, EXCLUDED.total_earned)
                        """,
                        user_address, balance
                    )
                    
                    self.stats["users_migrated"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate user {user_address}: {e}")
                    self.stats["errors"].append({
                        "type": "user_migration",
                        "user": user_address,
                        "error": str(e)
                    })
        
        logger.info(f"Migrated {self.stats['users_migrated']} users")
    
    async def migrate_transactions(self):
        """Migrate transaction history."""
        logger.info("Migrating transaction history...")
        
        # Load transaction log
        tx_log_path = Path("./data/ledger/transaction_log.json")
        if not tx_log_path.exists():
            logger.warning("No transaction history found")
            return
        
        with open(tx_log_path, 'r') as f:
            transactions = json.load(f)
        
        async with self.new_ledger.pool.acquire() as conn:
            for tx in transactions:
                try:
                    # Determine transaction type
                    if "reward" in tx.get("type", "").lower():
                        tx_type = TransactionType.REWARD
                    elif "refund" in tx.get("type", "").lower():
                        tx_type = TransactionType.REFUND
                    else:
                        tx_type = TransactionType.CHARGE
                    
                    # Insert transaction
                    await conn.execute(
                        """
                        INSERT INTO transactions (
                            id, user_address, tx_type, status, amount,
                            description, metadata, created_at, credited_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        tx.get("id", str(self.stats["transactions_migrated"])),
                        tx.get("user_address", "unknown"),
                        tx_type.value,
                        TransactionStatus.CREDITED.value,
                        Decimal(tx.get("amount", "0")),
                        tx.get("description", "Migrated transaction"),
                        json.dumps(tx),
                        datetime.fromisoformat(tx.get("timestamp", datetime.utcnow().isoformat()))
                    )
                    
                    self.stats["transactions_migrated"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate transaction: {e}")
                    self.stats["errors"].append({
                        "type": "transaction_migration",
                        "tx": tx,
                        "error": str(e)
                    })
        
        logger.info(f"Migrated {self.stats['transactions_migrated']} transactions")
    
    async def validate_migration(self):
        """Validate migrated data integrity."""
        logger.info("Validating migration...")
        
        # Compare total balances
        old_total = Decimal(0)
        ledger_path = Path("./data/ledger/user_balances.json")
        
        if ledger_path.exists():
            with open(ledger_path, 'r') as f:
                user_balances = json.load(f)
                for user, balance in user_balances.items():
                    if not user.startswith("SYSTEM_"):
                        old_total += Decimal(balance)
        
        # Get new total from PostgreSQL
        async with self.new_ledger.pool.acquire() as conn:
            new_total = await conn.fetchval(
                """
                SELECT COALESCE(SUM(balance), 0) 
                FROM user_balances 
                WHERE user_address NOT LIKE 'SYSTEM_%'
                """
            )
            new_total = Decimal(str(new_total))
        
        discrepancy = abs(old_total - new_total)
        
        if discrepancy > Decimal("0.000001"):
            logger.error(f"Balance discrepancy detected: Old={old_total}, New={new_total}, Diff={discrepancy}")
            self.stats["discrepancies"].append({
                "type": "total_balance",
                "old": str(old_total),
                "new": str(new_total),
                "difference": str(discrepancy)
            })
        else:
            logger.info(f"Balance validation passed: {new_total}")
        
        # Run reconciliation
        result = await self.new_ledger.run_daily_reconciliation()
        logger.info(f"Reconciliation result: {result}")
        
        return discrepancy < Decimal("0.000001")
    
    async def run_migration(self, setup_db: bool = True):
        """Run complete migration."""
        logger.info("Starting ledger migration to PostgreSQL...")
        
        try:
            # Step 1: Setup database
            if setup_db:
                await self.setup_database()
            else:
                await self.new_ledger.connect()
            
            # Step 2: Migrate users
            await self.migrate_users()
            
            # Step 3: Migrate transactions
            await self.migrate_transactions()
            
            # Step 4: Validate
            is_valid = await self.validate_migration()
            
            # Print summary
            logger.info("=" * 50)
            logger.info("Migration Summary:")
            logger.info(f"  Users migrated: {self.stats['users_migrated']}")
            logger.info(f"  Transactions migrated: {self.stats['transactions_migrated']}")
            logger.info(f"  Errors: {len(self.stats['errors'])}")
            logger.info(f"  Discrepancies: {len(self.stats['discrepancies'])}")
            logger.info(f"  Validation: {'PASSED' if is_valid else 'FAILED'}")
            
            if self.stats['errors']:
                logger.error("Errors encountered:")
                for error in self.stats['errors'][:5]:  # Show first 5 errors
                    logger.error(f"  - {error}")
            
            if self.stats['discrepancies']:
                logger.warning("Discrepancies found:")
                for disc in self.stats['discrepancies']:
                    logger.warning(f"  - {disc}")
            
            # Save migration report
            report_path = Path("./data/migration_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "stats": self.stats,
                    "valid": is_valid
                }, f, indent=2, default=str)
            
            logger.info(f"Migration report saved to: {report_path}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        
        finally:
            await self.new_ledger.disconnect()

async def main():
    """Main migration entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate ledger to PostgreSQL")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://blyan_user:blyan_pass@localhost/blyan_db"),
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip database setup (assume schema exists)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, don't migrate"
    )
    
    args = parser.parse_args()
    
    migration = LedgerMigration(args.database_url)
    
    if args.dry_run:
        logger.info("DRY RUN - Validation only")
        await migration.new_ledger.connect()
        await migration.validate_migration()
    else:
        success = await migration.run_migration(setup_db=not args.skip_setup)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())