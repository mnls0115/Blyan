#!/usr/bin/env python3
"""
PostgreSQL-backed Ledger System
Production-ready double-entry bookkeeping with atomic transactions
"""

import asyncio
import uuid
from decimal import Decimal
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging

from .db_config import db, init_database

logger = logging.getLogger(__name__)


class PostgresLedger:
    """PostgreSQL-backed ledger with double-entry bookkeeping."""
    
    def __init__(self):
        self.db = db
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection and schema."""
        if self._initialized:
            return
        
        await init_database()
        self._initialized = True
        logger.info("PostgreSQL Ledger initialized")
    
    async def create_user_wallet(self, address: str, initial_balance: Decimal = Decimal("0")) -> str:
        """Create or get user wallet account."""
        async with self.db.acquire() as conn:
            account_id = await conn.fetchval(
                "SELECT ledger.create_user_wallet($1, $2)",
                address.lower(),
                initial_balance
            )
            return str(account_id)
    
    async def get_balance(self, address: str) -> Decimal:
        """Get user balance."""
        async with self.db.acquire() as conn:
            balance = await conn.fetchval(
                """
                SELECT balance 
                FROM ledger.accounts 
                WHERE owner_address = $1 AND account_type = 'user_wallet'
                """,
                address.lower()
            )
            return Decimal(str(balance)) if balance else Decimal("0")
    
    async def process_payment(
        self,
        stripe_event_id: str,
        user_address: str,
        gross_amount: Decimal,
        stripe_fee: Decimal,
        bly_amount: Decimal
    ) -> Dict[str, Any]:
        """Process Stripe payment with atomic transaction."""
        
        idempotency_key = f"stripe_{stripe_event_id}"
        
        async with self.db.transaction() as conn:
            # Check if already processed
            existing = await conn.fetchrow(
                "SELECT id, status FROM ledger.transactions WHERE idempotency_key = $1",
                idempotency_key
            )
            
            if existing:
                return {
                    "transaction_id": str(existing["id"]),
                    "status": existing["status"],
                    "idempotent": True
                }
            
            # Ensure user wallet exists
            await self.create_user_wallet(user_address)
            
            # Record Stripe event
            await conn.execute(
                """
                INSERT INTO ledger.stripe_events 
                (stripe_event_id, event_type, amount_cents, currency, raw_event)
                VALUES ($1, $2, $3, $4, $5)
                """,
                stripe_event_id,
                "payment_intent.succeeded",
                int(gross_amount * 100),  # Convert to cents
                "USD",
                {"amount": str(gross_amount), "fee": str(stripe_fee)}
            )
            
            # Transfer from Stripe gateway to validation pool (minus fee)
            net_amount = gross_amount - stripe_fee
            
            # Record payment transaction
            tx1_id = await conn.fetchval(
                """
                SELECT ledger.record_transaction(
                    $1, -- idempotency_key
                    $2, -- transaction_type
                    $3, -- amount
                    $4, -- from_account
                    $5, -- to_account
                    $6, -- description
                    $7  -- metadata
                )
                """,
                f"{idempotency_key}_payment",
                "stripe_payment",
                net_amount,
                "STRIPE_GATEWAY",
                "VALIDATION_POOL",
                f"Payment from {user_address}",
                {
                    "stripe_event_id": stripe_event_id,
                    "user_address": user_address,
                    "stripe_fee": str(stripe_fee)
                }
            )
            
            # Issue BLY tokens to user
            tx2_id = await conn.fetchval(
                """
                SELECT ledger.record_transaction(
                    $1, -- idempotency_key
                    $2, -- transaction_type
                    $3, -- amount
                    $4, -- from_account
                    $5, -- to_account
                    $6, -- description
                    $7  -- metadata
                )
                """,
                f"{idempotency_key}_issue",
                "token_issuance",
                bly_amount,
                "TREASURY",
                user_address.lower(),
                f"BLY tokens issued for payment",
                {
                    "payment_tx": str(tx1_id),
                    "exchange_rate": str(gross_amount / bly_amount) if bly_amount > 0 else "0"
                }
            )
            
            # Burn 50% of issued tokens
            burn_amount = bly_amount * Decimal("0.5")
            if burn_amount > 0:
                tx3_id = await conn.fetchval(
                    """
                    SELECT ledger.record_transaction(
                        $1, -- idempotency_key
                        $2, -- transaction_type
                        $3, -- amount
                        $4, -- from_account
                        $5, -- to_account
                        $6, -- description
                        $7  -- metadata
                    )
                    """,
                    f"{idempotency_key}_burn",
                    "token_burn",
                    burn_amount,
                    user_address.lower(),
                    "BURN_ADDRESS",
                    "Automatic 50% burn",
                    {"parent_tx": str(tx2_id)}
                )
            
            return {
                "transaction_id": str(tx2_id),
                "payment_tx": str(tx1_id),
                "burn_tx": str(tx3_id) if burn_amount > 0 else None,
                "bly_amount": str(bly_amount),
                "burn_amount": str(burn_amount),
                "net_balance": str(bly_amount - burn_amount),
                "status": "completed",
                "idempotent": False
            }
    
    async def distribute_rewards(
        self,
        validator_address: str,
        amount: Decimal,
        reward_type: str,
        quality_score: float = None
    ) -> str:
        """Distribute rewards to validator."""
        
        idempotency_key = f"reward_{validator_address}_{reward_type}_{datetime.utcnow().isoformat()}"
        
        async with self.db.transaction() as conn:
            # Ensure validator wallet exists
            await self.create_user_wallet(validator_address)
            
            # Transfer from reward pool to validator
            tx_id = await conn.fetchval(
                """
                SELECT ledger.record_transaction(
                    $1, -- idempotency_key
                    $2, -- transaction_type
                    $3, -- amount
                    $4, -- from_account
                    $5, -- to_account
                    $6, -- description
                    $7  -- metadata
                )
                """,
                idempotency_key,
                f"reward_{reward_type}",
                amount,
                "REWARD_POOL",
                validator_address.lower(),
                f"{reward_type} reward distribution",
                {
                    "reward_type": reward_type,
                    "quality_score": quality_score
                }
            )
            
            # Record in reward distributions table
            await conn.execute(
                """
                INSERT INTO ledger.reward_distributions
                (transaction_id, validator_address, amount, reward_type, quality_score)
                VALUES ($1, $2, $3, $4, $5)
                """,
                uuid.UUID(tx_id),
                validator_address.lower(),
                amount,
                reward_type,
                Decimal(str(quality_score)) if quality_score else None
            )
            
            return str(tx_id)
    
    async def get_user_transactions(
        self, 
        address: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user transaction history."""
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    t.id,
                    t.transaction_type,
                    t.amount,
                    t.currency,
                    t.description,
                    t.status,
                    t.created_at,
                    t.metadata,
                    CASE 
                        WHEN e_debit.account_id IS NOT NULL THEN 'debit'
                        WHEN e_credit.account_id IS NOT NULL THEN 'credit'
                    END as direction
                FROM ledger.transactions t
                LEFT JOIN ledger.entries e_debit ON 
                    t.id = e_debit.transaction_id 
                    AND e_debit.account_id = (
                        SELECT id FROM ledger.accounts 
                        WHERE owner_address = $1 AND account_type = 'user_wallet'
                    )
                    AND e_debit.debit IS NOT NULL
                LEFT JOIN ledger.entries e_credit ON 
                    t.id = e_credit.transaction_id 
                    AND e_credit.account_id = (
                        SELECT id FROM ledger.accounts 
                        WHERE owner_address = $1 AND account_type = 'user_wallet'
                    )
                    AND e_credit.credit IS NOT NULL
                WHERE e_debit.account_id IS NOT NULL OR e_credit.account_id IS NOT NULL
                ORDER BY t.created_at DESC
                LIMIT $2 OFFSET $3
                """,
                address.lower(),
                limit,
                offset
            )
            
            return [
                {
                    "id": str(row["id"]),
                    "type": row["transaction_type"],
                    "amount": str(row["amount"]),
                    "currency": row["currency"],
                    "description": row["description"],
                    "status": row["status"],
                    "direction": row["direction"],
                    "created_at": row["created_at"].isoformat(),
                    "metadata": dict(row["metadata"]) if row["metadata"] else {}
                }
                for row in rows
            ]
    
    async def get_account_summary(self, address: str) -> Dict[str, Any]:
        """Get comprehensive account summary."""
        async with self.db.acquire() as conn:
            # Get account info
            account = await conn.fetchrow(
                """
                SELECT 
                    id,
                    balance,
                    created_at,
                    updated_at
                FROM ledger.accounts
                WHERE owner_address = $1 AND account_type = 'user_wallet'
                """,
                address.lower()
            )
            
            if not account:
                return {
                    "address": address,
                    "balance": "0",
                    "total_received": "0",
                    "total_sent": "0",
                    "transaction_count": 0
                }
            
            # Get aggregated stats
            stats = await conn.fetchrow(
                """
                SELECT 
                    COALESCE(SUM(CASE WHEN credit IS NOT NULL THEN credit ELSE 0 END), 0) as total_received,
                    COALESCE(SUM(CASE WHEN debit IS NOT NULL THEN debit ELSE 0 END), 0) as total_sent,
                    COUNT(DISTINCT transaction_id) as transaction_count
                FROM ledger.entries
                WHERE account_id = $1
                """,
                account["id"]
            )
            
            return {
                "address": address,
                "balance": str(account["balance"]),
                "total_received": str(stats["total_received"]),
                "total_sent": str(stats["total_sent"]),
                "transaction_count": stats["transaction_count"],
                "created_at": account["created_at"].isoformat(),
                "updated_at": account["updated_at"].isoformat()
            }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        async with self.db.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    (SELECT COUNT(*) FROM ledger.accounts WHERE account_type = 'user_wallet') as total_users,
                    (SELECT COUNT(*) FROM ledger.transactions WHERE status = 'completed') as total_transactions,
                    (SELECT COALESCE(SUM(balance), 0) FROM ledger.accounts WHERE account_type = 'user_wallet') as total_user_balance,
                    (SELECT balance FROM ledger.accounts WHERE account_code = 'BURN_ADDRESS') as total_burned,
                    (SELECT balance FROM ledger.accounts WHERE account_code = 'REWARD_POOL') as reward_pool_balance,
                    (SELECT balance FROM ledger.accounts WHERE account_code = 'VALIDATION_POOL') as validation_pool_balance
                """
            )
            
            return {
                "total_users": stats["total_users"],
                "total_transactions": stats["total_transactions"],
                "total_user_balance": str(stats["total_user_balance"]),
                "total_burned": str(stats["total_burned"]),
                "reward_pool_balance": str(stats["reward_pool_balance"]),
                "validation_pool_balance": str(stats["validation_pool_balance"]),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global ledger instance
postgres_ledger = PostgresLedger()


async def test_ledger():
    """Test ledger operations."""
    ledger = PostgresLedger()
    await ledger.initialize()
    
    # Test user creation
    user_address = "0x1234567890abcdef"
    await ledger.create_user_wallet(user_address)
    
    # Test balance
    balance = await ledger.get_balance(user_address)
    print(f"User balance: {balance}")
    
    # Test payment processing
    result = await ledger.process_payment(
        stripe_event_id=f"evt_test_{uuid.uuid4()}",
        user_address=user_address,
        gross_amount=Decimal("100.00"),
        stripe_fee=Decimal("2.90"),
        bly_amount=Decimal("1000.00")
    )
    print(f"Payment result: {result}")
    
    # Check new balance
    new_balance = await ledger.get_balance(user_address)
    print(f"New balance: {new_balance}")
    
    # Get transactions
    transactions = await ledger.get_user_transactions(user_address)
    print(f"Transactions: {transactions}")
    
    # Get system stats
    stats = await ledger.get_system_stats()
    print(f"System stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_ledger())