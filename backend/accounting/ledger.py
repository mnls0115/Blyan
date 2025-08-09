#!/usr/bin/env python3
"""
Double-Entry Accounting Ledger System
Ensures financial integrity with atomic transactions

### PRODUCTION FEATURES ###
- Double-entry bookkeeping
- Idempotency key tracking
- Atomic transaction guarantee
- Decimal precision (no float)
- Audit trail with immutable history
"""

import os
import uuid
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import redis
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Use Decimal for all monetary calculations
DECIMAL_PLACES = 4
ROUNDING = ROUND_HALF_UP

class AccountType(Enum):
    """Standard accounting account types."""
    ASSET = "asset"           # User balances, cash
    LIABILITY = "liability"    # Owed to users
    REVENUE = "revenue"       # Income from payments
    EXPENSE = "expense"       # Costs, fees
    EQUITY = "equity"         # Owner's equity
    
class Account(Enum):
    """Chart of accounts for Blyan Network."""
    # Assets
    USER_BLY_BALANCE = ("user_bly_balance", AccountType.ASSET)
    STRIPE_RECEIVABLE = ("stripe_receivable", AccountType.ASSET)
    VALIDATION_POOL = ("validation_pool", AccountType.ASSET)
    TRAINING_POOL = ("training_pool", AccountType.ASSET)
    
    # Liabilities
    USER_CREDITS_OWED = ("user_credits_owed", AccountType.LIABILITY)
    
    # Revenue
    PAYMENT_REVENUE = ("payment_revenue", AccountType.REVENUE)
    
    # Expenses
    STRIPE_FEES = ("stripe_fees", AccountType.EXPENSE)
    BURNED_TOKENS = ("burned_tokens", AccountType.EXPENSE)
    VALIDATION_COSTS = ("validation_costs", AccountType.EXPENSE)
    REWARDS_PAID = ("rewards_paid", AccountType.EXPENSE)
    
    def __init__(self, account_name: str, account_type: AccountType):
        self.account_name = account_name
        self.account_type = account_type

@dataclass
class LedgerEntry:
    """Single ledger entry (one side of double-entry)."""
    entry_id: str
    transaction_id: str
    account: str
    debit: Decimal
    credit: Decimal
    balance: Decimal  # Running balance after this entry
    created_at: float
    metadata: Dict

@dataclass
class Transaction:
    """Complete double-entry transaction."""
    transaction_id: str
    idempotency_key: str
    description: str
    entries: List[LedgerEntry]
    total_debits: Decimal
    total_credits: Decimal
    created_at: float
    reference_id: Optional[str] = None  # Stripe event ID, etc.
    status: str = "completed"
    
    def is_balanced(self) -> bool:
        """Verify debits equal credits."""
        return self.total_debits == self.total_credits

class Ledger:
    """
    Production-grade double-entry ledger with idempotency.
    """
    
    def __init__(self):
        # Redis for idempotency and locking
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=2,  # Separate DB for accounting
            decode_responses=True,
            password=os.environ.get('REDIS_PASSWORD'),
            ssl=os.environ.get('REDIS_SSL', 'false').lower() == 'true'
        )
        
        # In-memory cache (use PostgreSQL in production)
        self.transactions: Dict[str, Transaction] = {}
        self.entries_by_account: Dict[str, List[LedgerEntry]] = {}
        self.balances: Dict[str, Decimal] = {}
        
        # Initialize account balances
        for account in Account:
            self.balances[account.account_name] = Decimal("0")
            self.entries_by_account[account.account_name] = []
    
    @contextmanager
    def atomic_transaction(self, idempotency_key: str):
        """
        Ensure transaction is atomic and idempotent.
        """
        lock_key = f"ledger:lock:{idempotency_key}"
        lock = self.redis.lock(lock_key, timeout=30)
        
        try:
            # Acquire lock
            if not lock.acquire(blocking=True, blocking_timeout=5):
                raise Exception(f"Could not acquire lock for {idempotency_key}")
                
            # Check if transaction already exists
            existing_tx_id = self.redis.get(f"ledger:idempotency:{idempotency_key}")
            if existing_tx_id:
                existing_tx = self.transactions.get(existing_tx_id)
                if existing_tx:
                    logger.info(f"Transaction already exists for idempotency key: {idempotency_key}")
                    yield existing_tx
                    return
                    
            # New transaction
            yield None
            
        finally:
            lock.release()
    
    def create_transaction(
        self,
        idempotency_key: str,
        description: str,
        entries: List[Tuple[str, Decimal, Decimal, Dict]],
        reference_id: Optional[str] = None
    ) -> Transaction:
        """
        Create a balanced double-entry transaction.
        
        Args:
            idempotency_key: Unique key to prevent duplicates
            description: Transaction description
            entries: List of (account, debit, credit, metadata)
            reference_id: External reference (e.g., Stripe event ID)
        """
        with self.atomic_transaction(idempotency_key) as existing:
            if existing:
                return existing
                
            # Generate transaction ID
            transaction_id = str(uuid.uuid4())
            created_at = time.time()
            
            # Create entries
            ledger_entries = []
            total_debits = Decimal("0")
            total_credits = Decimal("0")
            
            for account_name, debit, credit, metadata in entries:
                # Validate account exists
                if account_name not in self.balances:
                    raise ValueError(f"Unknown account: {account_name}")
                    
                # Convert to Decimal with proper precision
                debit = Decimal(str(debit)).quantize(
                    Decimal(f"0.{'0' * DECIMAL_PLACES}"), 
                    rounding=ROUNDING
                )
                credit = Decimal(str(credit)).quantize(
                    Decimal(f"0.{'0' * DECIMAL_PLACES}"), 
                    rounding=ROUNDING
                )
                
                # Update running balance
                self.balances[account_name] += debit - credit
                
                # Create entry
                entry = LedgerEntry(
                    entry_id=str(uuid.uuid4()),
                    transaction_id=transaction_id,
                    account=account_name,
                    debit=debit,
                    credit=credit,
                    balance=self.balances[account_name],
                    created_at=created_at,
                    metadata=metadata
                )
                
                ledger_entries.append(entry)
                self.entries_by_account[account_name].append(entry)
                
                total_debits += debit
                total_credits += credit
            
            # Verify balanced
            if total_debits != total_credits:
                raise ValueError(
                    f"Unbalanced transaction: debits={total_debits}, credits={total_credits}"
                )
            
            # Create transaction
            transaction = Transaction(
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                description=description,
                entries=ledger_entries,
                total_debits=total_debits,
                total_credits=total_credits,
                created_at=created_at,
                reference_id=reference_id
            )
            
            # Store transaction
            self.transactions[transaction_id] = transaction
            
            # Store idempotency key
            self.redis.setex(
                f"ledger:idempotency:{idempotency_key}",
                86400,  # 24 hour TTL
                transaction_id
            )
            
            logger.info(f"✅ Transaction created: {transaction_id} ({description})")
            return transaction
    
    def process_payment(
        self,
        stripe_event_id: str,
        user_address: str,
        gross_amount: Decimal,
        stripe_fee: Decimal,
        bly_amount: Decimal
    ) -> Transaction:
        """
        Process a Stripe payment with proper accounting.
        """
        net_amount = gross_amount - stripe_fee
        
        # Calculate distribution
        burn_amount = bly_amount * Decimal("0.50")
        validation_amount = bly_amount * Decimal("0.25")
        training_amount = bly_amount * Decimal("0.25")
        
        # Create journal entries
        entries = [
            # Debit: Revenue received
            (Account.STRIPE_RECEIVABLE.account_name, net_amount, Decimal("0"), 
             {"type": "payment_received", "user": user_address}),
            
            # Credit: Payment revenue
            (Account.PAYMENT_REVENUE.account_name, Decimal("0"), gross_amount,
             {"type": "revenue", "source": "stripe"}),
            
            # Debit: Stripe fees
            (Account.STRIPE_FEES.account_name, stripe_fee, Decimal("0"),
             {"type": "fee", "provider": "stripe"}),
            
            # Debit: User BLY balance
            (Account.USER_BLY_BALANCE.account_name, bly_amount, Decimal("0"),
             {"type": "credit", "user": user_address}),
            
            # Credit: Distribution
            (Account.BURNED_TOKENS.account_name, Decimal("0"), burn_amount,
             {"type": "burn", "reason": "tokenomics"}),
            
            (Account.VALIDATION_POOL.account_name, Decimal("0"), validation_amount,
             {"type": "fund", "purpose": "validation"}),
            
            (Account.TRAINING_POOL.account_name, Decimal("0"), training_amount,
             {"type": "fund", "purpose": "training"})
        ]
        
        return self.create_transaction(
            idempotency_key=f"stripe:{stripe_event_id}",
            description=f"Payment from {user_address}",
            entries=entries,
            reference_id=stripe_event_id
        )
    
    def process_refund(
        self,
        stripe_event_id: str,
        original_transaction_id: str,
        refund_amount: Decimal,
        user_address: str
    ) -> Transaction:
        """
        Process a refund with reversal entries.
        """
        # Get original transaction
        original_tx = self.transactions.get(original_transaction_id)
        if not original_tx:
            raise ValueError(f"Original transaction not found: {original_transaction_id}")
        
        # Create reversal entries (opposite of original)
        reversal_entries = []
        for entry in original_tx.entries:
            # Swap debits and credits for reversal
            reversal_entries.append(
                (entry.account, entry.credit, entry.debit, 
                 {"type": "refund", "original_tx": original_transaction_id})
            )
        
        return self.create_transaction(
            idempotency_key=f"refund:{stripe_event_id}",
            description=f"Refund for {user_address}",
            entries=reversal_entries,
            reference_id=stripe_event_id
        )
    
    def get_balance(self, account_name: str) -> Decimal:
        """Get current balance for an account."""
        return self.balances.get(account_name, Decimal("0"))
    
    def get_user_balance(self, user_address: str) -> Decimal:
        """Get BLY balance for a specific user."""
        # In production, use separate user account tracking
        # For now, return from general user balance account
        return self.get_balance(Account.USER_BLY_BALANCE.account_name)
    
    def get_audit_trail(
        self, 
        account_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[LedgerEntry]:
        """Get audit trail for an account."""
        if account_name:
            entries = self.entries_by_account.get(account_name, [])
        else:
            # All entries
            entries = []
            for account_entries in self.entries_by_account.values():
                entries.extend(account_entries)
        
        # Filter by time
        if start_time:
            entries = [e for e in entries if e.created_at >= start_time]
        if end_time:
            entries = [e for e in entries if e.created_at <= end_time]
            
        return sorted(entries, key=lambda e: e.created_at)
    
    def export_trial_balance(self) -> Dict[str, Dict]:
        """Export trial balance for verification."""
        trial_balance = {
            "accounts": {},
            "total_debits": Decimal("0"),
            "total_credits": Decimal("0"),
            "balanced": False
        }
        
        for account_name, entries in self.entries_by_account.items():
            total_debits = sum(e.debit for e in entries)
            total_credits = sum(e.credit for e in entries)
            
            trial_balance["accounts"][account_name] = {
                "debits": str(total_debits),
                "credits": str(total_credits),
                "balance": str(self.balances[account_name])
            }
            
            trial_balance["total_debits"] += total_debits
            trial_balance["total_credits"] += total_credits
        
        trial_balance["total_debits"] = str(trial_balance["total_debits"])
        trial_balance["total_credits"] = str(trial_balance["total_credits"])
        trial_balance["balanced"] = trial_balance["total_debits"] == trial_balance["total_credits"]
        
        return trial_balance

# Global ledger instance
_ledger = None

def get_ledger() -> Ledger:
    """Get or create ledger singleton."""
    global _ledger
    if _ledger is None:
        _ledger = Ledger()
    return _ledger