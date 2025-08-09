#!/usr/bin/env python3
"""
PostgreSQL-based Ledger Implementation
Production-grade double-entry bookkeeping with state machine
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
import asyncpg
from asyncpg.pool import Pool
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """Transaction types."""
    CHARGE = "charge"
    REFUND = "refund"
    REWARD = "reward"
    TRANSFER = "transfer"
    STAKE = "stake"
    UNSTAKE = "unstake"
    SLASH = "slash"

class TransactionStatus(Enum):
    """Transaction status in state machine."""
    PENDING = "pending"
    QUOTED = "quoted"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    CREDITED = "credited"
    FAILED = "failed"
    REVERSED = "reversed"

@dataclass
class Transaction:
    """Transaction record."""
    id: str
    user_address: str
    tx_type: TransactionType
    status: TransactionStatus
    amount: Decimal
    fee: Decimal = Decimal(0)
    idempotency_key: Optional[str] = None
    quote_id: Optional[str] = None
    quoted_amount: Optional[Decimal] = None
    quote_expires_at: Optional[datetime] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['tx_type'] = self.tx_type.value
        data['status'] = self.status.value
        data['amount'] = str(self.amount)
        data['fee'] = str(self.fee)
        if self.quoted_amount:
            data['quoted_amount'] = str(self.quoted_amount)
        if self.metadata:
            data['metadata'] = json.dumps(self.metadata)
        return data

@dataclass
class UserBalance:
    """User balance record."""
    user_address: str
    balance: Decimal
    locked_balance: Decimal
    total_earned: Decimal
    total_spent: Decimal
    trust_level: str
    created_at: datetime
    updated_at: datetime

class PostgresLedger:
    """
    PostgreSQL-based ledger with atomic operations and state machine.
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "postgresql://localhost/blyan"
        self.pool: Optional[Pool] = None
        
    async def connect(self):
        """Initialize database connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool created")
    
    async def disconnect(self):
        """Close database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def get_user_balance(self, user_address: str) -> Decimal:
        """Get user's available balance."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT balance FROM user_balances WHERE user_address = $1",
                user_address
            )
            
            if row:
                return Decimal(str(row['balance']))
            return Decimal(0)
    
    async def get_user_details(self, user_address: str) -> Optional[UserBalance]:
        """Get full user balance details."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_address, balance, locked_balance, 
                       total_earned, total_spent, trust_level,
                       created_at, updated_at
                FROM user_balances 
                WHERE user_address = $1
                """,
                user_address
            )
            
            if row:
                return UserBalance(
                    user_address=row['user_address'],
                    balance=Decimal(str(row['balance'])),
                    locked_balance=Decimal(str(row['locked_balance'])),
                    total_earned=Decimal(str(row['total_earned'])),
                    total_spent=Decimal(str(row['total_spent'])),
                    trust_level=row['trust_level'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
    
    async def create_quote(
        self,
        user_address: str,
        amount: Decimal,
        quote_id: str,
        expires_in_seconds: int = 300,
        metadata: Dict[str, Any] = None
    ) -> Transaction:
        """Create a price quote (step 1 of state machine)."""
        async with self.pool.acquire() as conn:
            tx_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
            
            await conn.execute(
                """
                INSERT INTO transactions (
                    id, user_address, tx_type, status, amount,
                    quote_id, quoted_amount, quote_expires_at,
                    description, metadata, quoted_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
                """,
                tx_id, user_address, TransactionType.CHARGE.value,
                TransactionStatus.QUOTED.value, amount,
                quote_id, amount, expires_at,
                f"Quote for {quote_id}", 
                json.dumps(metadata) if metadata else '{}'
            )
            
            return Transaction(
                id=tx_id,
                user_address=user_address,
                tx_type=TransactionType.CHARGE,
                status=TransactionStatus.QUOTED,
                amount=amount,
                quote_id=quote_id,
                quoted_amount=amount,
                quote_expires_at=expires_at,
                metadata=metadata
            )
    
    async def authorize_transaction(
        self,
        quote_id: str,
        idempotency_key: Optional[str] = None
    ) -> Optional[Transaction]:
        """Authorize a quoted transaction (step 2)."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check idempotency
                if idempotency_key:
                    existing = await conn.fetchrow(
                        """
                        SELECT id, status FROM transactions 
                        WHERE idempotency_key = $1
                        """,
                        idempotency_key
                    )
                    if existing:
                        logger.info(f"Idempotent transaction found: {existing['id']}")
                        return None
                
                # Find and lock the quote
                row = await conn.fetchrow(
                    """
                    SELECT * FROM transactions 
                    WHERE quote_id = $1 AND status = 'quoted'
                    FOR UPDATE
                    """,
                    quote_id
                )
                
                if not row:
                    raise ValueError(f"Quote not found or already processed: {quote_id}")
                
                # Check expiry
                if row['quote_expires_at'] < datetime.utcnow():
                    await conn.execute(
                        """
                        UPDATE transactions 
                        SET status = 'failed', failed_at = CURRENT_TIMESTAMP,
                            error_message = 'Quote expired'
                        WHERE id = $1
                        """,
                        row['id']
                    )
                    raise ValueError(f"Quote expired: {quote_id}")
                
                # Check user balance
                balance = await conn.fetchval(
                    "SELECT balance FROM user_balances WHERE user_address = $1",
                    row['user_address']
                )
                
                if not balance or Decimal(str(balance)) < Decimal(str(row['amount'])):
                    await conn.execute(
                        """
                        UPDATE transactions 
                        SET status = 'failed', failed_at = CURRENT_TIMESTAMP,
                            error_message = 'Insufficient balance'
                        WHERE id = $1
                        """,
                        row['id']
                    )
                    raise ValueError(f"Insufficient balance for {row['user_address']}")
                
                # Authorize transaction
                await conn.execute(
                    """
                    UPDATE transactions 
                    SET status = 'authorized', 
                        authorized_at = CURRENT_TIMESTAMP,
                        idempotency_key = $2
                    WHERE id = $1
                    """,
                    row['id'], idempotency_key
                )
                
                # Lock the amount
                await conn.execute(
                    "SELECT update_user_balance($1, $2, 'lock')",
                    row['user_address'], row['amount']
                )
                
                return Transaction(
                    id=row['id'],
                    user_address=row['user_address'],
                    tx_type=TransactionType(row['tx_type']),
                    status=TransactionStatus.AUTHORIZED,
                    amount=Decimal(str(row['amount'])),
                    quote_id=quote_id,
                    idempotency_key=idempotency_key
                )
    
    async def capture_transaction(self, transaction_id: str) -> Transaction:
        """Capture an authorized transaction (step 3)."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Update status to captured
                row = await conn.fetchrow(
                    """
                    UPDATE transactions 
                    SET status = 'captured', captured_at = CURRENT_TIMESTAMP
                    WHERE id = $1 AND status = 'authorized'
                    RETURNING *
                    """,
                    transaction_id
                )
                
                if not row:
                    raise ValueError(f"Transaction not found or not authorized: {transaction_id}")
                
                return Transaction(
                    id=row['id'],
                    user_address=row['user_address'],
                    tx_type=TransactionType(row['tx_type']),
                    status=TransactionStatus.CAPTURED,
                    amount=Decimal(str(row['amount']))
                )
    
    async def credit_transaction(
        self,
        transaction_id: str,
        actual_amount: Optional[Decimal] = None
    ) -> Transaction:
        """Credit a captured transaction (step 4 - final)."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get transaction
                row = await conn.fetchrow(
                    """
                    SELECT * FROM transactions 
                    WHERE id = $1 AND status = 'captured'
                    FOR UPDATE
                    """,
                    transaction_id
                )
                
                if not row:
                    raise ValueError(f"Transaction not found or not captured: {transaction_id}")
                
                final_amount = actual_amount or Decimal(str(row['amount']))
                user_address = row['user_address']
                
                # Update transaction status
                await conn.execute(
                    """
                    UPDATE transactions 
                    SET status = 'credited', 
                        credited_at = CURRENT_TIMESTAMP,
                        amount = $2
                    WHERE id = $1
                    """,
                    transaction_id, final_amount
                )
                
                # Unlock original amount
                await conn.execute(
                    "SELECT update_user_balance($1, $2, 'unlock')",
                    user_address, row['amount']
                )
                
                # Deduct actual amount
                await conn.execute(
                    "SELECT update_user_balance($1, $2, 'subtract')",
                    user_address, final_amount
                )
                
                # Create ledger entries for double-entry bookkeeping
                ledger_id = str(uuid.uuid4())
                
                # Debit user account
                await conn.execute(
                    """
                    INSERT INTO ledger_entries (
                        id, transaction_id, account_type, account_id,
                        debit, credit, balance_after
                    ) VALUES ($1, $2, $3, $4, $5, $6, 
                        (SELECT balance FROM user_balances WHERE user_address = $4))
                    """,
                    str(uuid.uuid4()), transaction_id, 'user', user_address,
                    final_amount, Decimal(0)
                )
                
                # Credit system revenue account
                await conn.execute(
                    """
                    INSERT INTO ledger_entries (
                        id, transaction_id, account_type, account_id,
                        debit, credit, balance_after
                    ) VALUES ($1, $2, $3, $4, $5, $6,
                        (SELECT balance FROM user_balances WHERE user_address = $4))
                    """,
                    str(uuid.uuid4()), transaction_id, 'system', 'SYSTEM_FEE_POOL',
                    Decimal(0), final_amount
                )
                
                return Transaction(
                    id=row['id'],
                    user_address=user_address,
                    tx_type=TransactionType(row['tx_type']),
                    status=TransactionStatus.CREDITED,
                    amount=final_amount
                )
    
    async def add_reward(
        self,
        user_address: str,
        amount: Decimal,
        description: str,
        metadata: Dict[str, Any] = None
    ) -> Transaction:
        """Add reward to user balance."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                tx_id = str(uuid.uuid4())
                
                # Create transaction record
                await conn.execute(
                    """
                    INSERT INTO transactions (
                        id, user_address, tx_type, status, amount,
                        description, metadata, credited_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                    """,
                    tx_id, user_address, TransactionType.REWARD.value,
                    TransactionStatus.CREDITED.value, amount,
                    description, json.dumps(metadata) if metadata else '{}'
                )
                
                # Update user balance
                await conn.execute(
                    "SELECT update_user_balance($1, $2, 'add')",
                    user_address, amount
                )
                
                # Create ledger entries
                # Debit reward pool
                await conn.execute(
                    """
                    INSERT INTO ledger_entries (
                        transaction_id, account_type, account_id,
                        debit, credit, balance_after
                    ) VALUES ($1, $2, $3, $4, $5,
                        (SELECT balance FROM user_balances WHERE user_address = $3))
                    """,
                    tx_id, 'system', 'SYSTEM_REWARD_POOL',
                    amount, Decimal(0)
                )
                
                # Credit user account
                await conn.execute(
                    """
                    INSERT INTO ledger_entries (
                        transaction_id, account_type, account_id,
                        debit, credit, balance_after
                    ) VALUES ($1, $2, $3, $4, $5,
                        (SELECT balance FROM user_balances WHERE user_address = $3))
                    """,
                    tx_id, 'user', user_address,
                    Decimal(0), amount
                )
                
                return Transaction(
                    id=tx_id,
                    user_address=user_address,
                    tx_type=TransactionType.REWARD,
                    status=TransactionStatus.CREDITED,
                    amount=amount,
                    description=description,
                    metadata=metadata
                )
    
    async def run_daily_reconciliation(self, date: datetime = None) -> Dict[str, Any]:
        """Run daily reconciliation and check for discrepancies."""
        if not date:
            date = datetime.utcnow().date()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Calculate aggregates
                stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(DISTINCT user_address) as total_users,
                        SUM(balance) as total_balance,
                        SUM(locked_balance) as total_locked
                    FROM user_balances
                    WHERE user_address NOT LIKE 'SYSTEM_%'
                    """
                )
                
                tx_stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_transactions,
                        SUM(amount) as total_volume,
                        SUM(CASE WHEN tx_type = 'charge' THEN 1 ELSE 0 END) as charges_count,
                        SUM(CASE WHEN tx_type = 'charge' THEN amount ELSE 0 END) as charges_volume,
                        SUM(CASE WHEN tx_type = 'reward' THEN 1 ELSE 0 END) as rewards_count,
                        SUM(CASE WHEN tx_type = 'reward' THEN amount ELSE 0 END) as rewards_volume,
                        SUM(CASE WHEN tx_type = 'refund' THEN 1 ELSE 0 END) as refunds_count,
                        SUM(CASE WHEN tx_type = 'refund' THEN amount ELSE 0 END) as refunds_volume
                    FROM transactions
                    WHERE DATE(created_at) = $1 AND status = 'credited'
                    """,
                    date
                )
                
                # Calculate ledger balance
                ledger_balance = await conn.fetchval(
                    """
                    SELECT 
                        SUM(CASE 
                            WHEN account_type = 'user' THEN credit - debit
                            ELSE 0
                        END) as user_ledger_balance
                    FROM ledger_entries
                    WHERE DATE(created_at) <= $1
                    """,
                    date
                )
                
                # Save reconciliation record
                recon_id = await conn.fetchval(
                    """
                    INSERT INTO daily_reconciliations (
                        reconciliation_date, total_users, total_balance, total_locked,
                        total_transactions, total_volume,
                        charges_count, charges_volume,
                        rewards_count, rewards_volume,
                        refunds_count, refunds_volume,
                        ledger_balance_sum, user_balance_sum,
                        is_reconciled
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    RETURNING id
                    """,
                    date,
                    stats['total_users'] or 0,
                    stats['total_balance'] or 0,
                    stats['total_locked'] or 0,
                    tx_stats['total_transactions'] or 0,
                    tx_stats['total_volume'] or 0,
                    tx_stats['charges_count'] or 0,
                    tx_stats['charges_volume'] or 0,
                    tx_stats['rewards_count'] or 0,
                    tx_stats['rewards_volume'] or 0,
                    tx_stats['refunds_count'] or 0,
                    tx_stats['refunds_volume'] or 0,
                    ledger_balance or 0,
                    stats['total_balance'] or 0,
                    True
                )
                
                # Check for discrepancy
                discrepancy = Decimal(str(ledger_balance or 0)) - Decimal(str(stats['total_balance'] or 0))
                
                result = {
                    "reconciliation_id": str(recon_id),
                    "date": str(date),
                    "total_users": stats['total_users'] or 0,
                    "total_balance": str(stats['total_balance'] or 0),
                    "ledger_balance": str(ledger_balance or 0),
                    "discrepancy": str(discrepancy),
                    "is_balanced": abs(discrepancy) < Decimal("0.000001"),
                    "transactions": {
                        "total": tx_stats['total_transactions'] or 0,
                        "volume": str(tx_stats['total_volume'] or 0),
                        "charges": tx_stats['charges_count'] or 0,
                        "rewards": tx_stats['rewards_count'] or 0,
                        "refunds": tx_stats['refunds_count'] or 0
                    }
                }
                
                if not result["is_balanced"]:
                    logger.error(f"Reconciliation discrepancy detected: {discrepancy}")
                    # Could trigger alerts here
                
                return result

# Singleton instance
_postgres_ledger = None

def get_postgres_ledger() -> PostgresLedger:
    """Get or create PostgreSQL ledger singleton."""
    global _postgres_ledger
    if _postgres_ledger is None:
        _postgres_ledger = PostgresLedger()
    return _postgres_ledger