#!/usr/bin/env python3
"""
Transaction Manager for Atomic Chat Operations
Ensures all-or-nothing execution for critical paths
"""

import asyncio
import time
import uuid
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import redis.asyncio as aioredis
from decimal import Decimal

logger = logging.getLogger(__name__)

class TransactionState(Enum):
    """Transaction lifecycle states."""
    INITIATED = "initiated"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

@dataclass
class TransactionContext:
    """Context for a single atomic transaction."""
    transaction_id: str
    user_address: str
    idempotency_key: Optional[str] = None
    state: TransactionState = TransactionState.INITIATED
    created_at: float = field(default_factory=time.time)
    
    # Resources to track for rollback
    quota_reserved: bool = False
    quote_consumed: bool = False
    balance_reserved: Decimal = Decimal(0)
    inference_completed: bool = False
    is_free_tier_request: bool = False
    
    # Rollback operations
    rollback_operations: List[Callable] = field(default_factory=list)
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def add_rollback(self, operation: Callable):
        """Add a rollback operation to be executed on failure."""
        self.rollback_operations.append(operation)

class ChatTransactionManager:
    """
    Manages atomic transactions for chat operations.
    Ensures consistency across quota, balance, and inference operations.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.active_transactions: Dict[str, TransactionContext] = {}
        
        # Transaction timeout (seconds)
        self.TRANSACTION_TIMEOUT = 60
        
        # Idempotency cache TTL (seconds)
        self.IDEMPOTENCY_TTL = 300  # 5 minutes
    
    async def connect(self):
        """Initialize Redis connection."""
        if not self.redis_client:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    @asynccontextmanager
    async def atomic_transaction(
        self,
        user_address: str,
        idempotency_key: Optional[str] = None
    ):
        """
        Context manager for atomic chat transactions.
        
        Usage:
            async with transaction_manager.atomic_transaction(user_address) as ctx:
                # Perform operations
                await reserve_quota(ctx)
                await validate_balance(ctx)
                result = await execute_inference(ctx)
                ctx.result = result
        """
        # Generate transaction ID
        transaction_id = f"tx_{uuid.uuid4().hex[:16]}_{int(time.time())}"
        
        # Check idempotency
        if idempotency_key:
            cached_result = await self._check_idempotency(idempotency_key)
            if cached_result:
                logger.info(f"Idempotent request found: {idempotency_key}")
                ctx = TransactionContext(
                    transaction_id=transaction_id,
                    user_address=user_address,
                    idempotency_key=idempotency_key,
                    state=TransactionState.COMMITTED,
                    result=cached_result
                )
                yield ctx
                return
        
        # Create transaction context
        ctx = TransactionContext(
            transaction_id=transaction_id,
            user_address=user_address,
            idempotency_key=idempotency_key
        )
        
        self.active_transactions[transaction_id] = ctx
        
        try:
            # Begin transaction
            ctx.state = TransactionState.VALIDATING
            logger.info(f"Transaction {transaction_id} started for {user_address}")
            
            # Yield control to caller
            yield ctx
            
            # If we get here, transaction succeeded
            ctx.state = TransactionState.COMMITTING
            await self._commit_transaction(ctx)
            
        except Exception as e:
            # Transaction failed - execute rollback
            logger.error(f"Transaction {transaction_id} failed: {e}")
            ctx.error = str(e)
            ctx.state = TransactionState.ROLLING_BACK
            await self._rollback_transaction(ctx)
            raise
            
        finally:
            # Clean up
            del self.active_transactions[transaction_id]
    
    async def _check_idempotency(self, idempotency_key: str) -> Optional[Dict]:
        """Check if this request was already processed."""
        await self.connect()
        
        cache_key = f"idempotency:{idempotency_key}"
        cached = await self.redis_client.get(cache_key)
        
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                return None
        return None
    
    async def _save_idempotency(self, idempotency_key: str, result: Dict):
        """Save result for idempotency."""
        await self.connect()
        
        cache_key = f"idempotency:{idempotency_key}"
        await self.redis_client.setex(
            cache_key,
            self.IDEMPOTENCY_TTL,
            json.dumps(result)
        )
    
    async def _commit_transaction(self, ctx: TransactionContext):
        """Commit a successful transaction."""
        logger.info(f"Committing transaction {ctx.transaction_id}")
        
        # Save idempotency result if applicable
        if ctx.idempotency_key and ctx.result:
            await self._save_idempotency(ctx.idempotency_key, ctx.result)
        
        # Record transaction success metrics
        await self._record_transaction_metrics(ctx, success=True)
        
        ctx.state = TransactionState.COMMITTED
        logger.info(f"Transaction {ctx.transaction_id} committed successfully")
    
    async def _rollback_transaction(self, ctx: TransactionContext):
        """Rollback a failed transaction."""
        logger.warning(f"Rolling back transaction {ctx.transaction_id}")
        
        # Execute rollback operations in reverse order
        for operation in reversed(ctx.rollback_operations):
            try:
                if asyncio.iscoroutinefunction(operation):
                    await operation()
                else:
                    operation()
                logger.info(f"Rollback operation executed: {operation.__name__}")
            except Exception as e:
                logger.error(f"Rollback operation failed: {e}")
        
        # Record transaction failure metrics
        await self._record_transaction_metrics(ctx, success=False)
        
        ctx.state = TransactionState.ROLLED_BACK
        logger.warning(f"Transaction {ctx.transaction_id} rolled back")
    
    async def _record_transaction_metrics(self, ctx: TransactionContext, success: bool):
        """Record transaction metrics for monitoring."""
        await self.connect()
        
        # Increment counters
        if success:
            await self.redis_client.hincrby("tx_metrics", "success_count", 1)
        else:
            await self.redis_client.hincrby("tx_metrics", "failure_count", 1)
            
            # Track failure reasons
            if ctx.error:
                reason_key = self._categorize_error(ctx.error)
                await self.redis_client.hincrby("tx_failure_reasons", reason_key, 1)
        
        # Track transaction duration
        duration = time.time() - ctx.created_at
        await self.redis_client.lpush("tx_durations", duration)
        await self.redis_client.ltrim("tx_durations", 0, 999)  # Keep last 1000
    
    def _categorize_error(self, error: str) -> str:
        """Categorize error for metrics."""
        error_lower = error.lower()
        
        if "quota" in error_lower or "limit" in error_lower:
            return "quota_exceeded"
        elif "balance" in error_lower or "insufficient" in error_lower:
            return "insufficient_balance"
        elif "ttl" in error_lower or "expired" in error_lower:
            return "quote_expired"
        elif "token" in error_lower or "mismatch" in error_lower:
            return "token_mismatch"
        elif "abuse" in error_lower or "challenge" in error_lower:
            return "abuse_detected"
        else:
            return "other"
    
    async def get_transaction_metrics(self) -> Dict[str, Any]:
        """Get transaction metrics for monitoring."""
        await self.connect()
        
        # Get basic counters
        metrics = await self.redis_client.hgetall("tx_metrics")
        failure_reasons = await self.redis_client.hgetall("tx_failure_reasons")
        
        # Calculate success rate
        success = int(metrics.get("success_count", 0))
        failure = int(metrics.get("failure_count", 0))
        total = success + failure
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        # Get duration stats
        durations = await self.redis_client.lrange("tx_durations", 0, -1)
        durations = [float(d) for d in durations]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            p95_duration = sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations)
        else:
            avg_duration = 0
            p95_duration = 0
        
        return {
            "total_transactions": total,
            "successful": success,
            "failed": failure,
            "success_rate": f"{success_rate:.1f}%",
            "avg_duration_ms": int(avg_duration * 1000),
            "p95_duration_ms": int(p95_duration * 1000),
            "failure_reasons": failure_reasons,
            "active_transactions": len(self.active_transactions)
        }

# Singleton instance
_transaction_manager = None

def get_transaction_manager() -> ChatTransactionManager:
    """Get or create transaction manager singleton."""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = ChatTransactionManager()
    return _transaction_manager