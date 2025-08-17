"""Token accounting and receipt generation middleware for BLY economics."""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog

from backend.crypto.signing import KeyPair
from backend.utils.json_canonical import dumps_canonical
from server.middleware.chain_bridge import InferenceReceipt

logger = structlog.get_logger()


@dataclass
class TokenUsage:
    """Token usage tracking with cost calculation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    
    def calculate_cost(self, pricing: Dict[str, Decimal]) -> Decimal:
        """Calculate BLY cost based on token usage."""
        # Get model-specific pricing or use default
        model_pricing = pricing.get(self.model_name, pricing.get("default", {}))
        
        prompt_cost = Decimal(self.prompt_tokens) / 1000 * model_pricing.get("prompt_per_1k", Decimal("0.001"))
        completion_cost = Decimal(self.completion_tokens) / 1000 * model_pricing.get("completion_per_1k", Decimal("0.003"))
        
        return prompt_cost + completion_cost


@dataclass
class BillingReceipt:
    """Billing receipt with BLY token costs."""
    request_id: str
    user_address: str
    token_usage: TokenUsage
    bly_cost: Decimal
    usd_equivalent: Decimal
    timestamp: float
    inference_receipt_hash: Optional[str] = None
    payment_tx_hash: Optional[str] = None
    status: str = "pending"  # pending, paid, failed, free_tier
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with string decimals."""
        data = asdict(self)
        data["bly_cost"] = str(self.bly_cost)
        data["usd_equivalent"] = str(self.usd_equivalent)
        data["token_usage"] = asdict(self.token_usage)
        return data


class AccountingMiddleware:
    """Middleware for token accounting and BLY billing."""
    
    def __init__(
        self,
        pricing_config: Optional[Dict] = None,
        bly_usd_rate: Decimal = Decimal("0.10"),
        ledger_client=None,
        receipt_log_path: Optional[Path] = None
    ):
        # Pricing configuration (BLY per 1k tokens)
        self.pricing = pricing_config or {
            "default": {
                "prompt_per_1k": Decimal("0.001"),
                "completion_per_1k": Decimal("0.003")
            },
            "gpt_oss_20b": {
                "prompt_per_1k": Decimal("0.001"),
                "completion_per_1k": Decimal("0.003")
            },
            "gpt_oss_120b": {
                "prompt_per_1k": Decimal("0.005"),
                "completion_per_1k": Decimal("0.015")
            }
        }
        
        self.bly_usd_rate = bly_usd_rate
        self.ledger_client = ledger_client
        self.receipt_log_path = receipt_log_path or Path("data/receipts/billing_receipts.jsonl")
        
        # Ensure receipt directory exists
        self.receipt_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.receipts: Dict[str, BillingReceipt] = {}
        self.pending_payments: Dict[str, BillingReceipt] = {}
        
        # Metrics
        self.total_tokens_processed = 0
        self.total_bly_billed = Decimal("0")
        self.total_receipts = 0
        
        # Batch processing
        self.batch_lock = asyncio.Lock()
        self.batch_threshold = Decimal("0.5")  # Process payments when user owes > 0.5 BLY
        
    async def track_request(
        self,
        request_id: str,
        user_address: str,
        prompt_tokens: List[int],
        completion_tokens: List[int],
        model_name: str = "gpt_oss_20b",
        inference_receipt: Optional[InferenceReceipt] = None
    ) -> BillingReceipt:
        """Track token usage and generate billing receipt."""
        
        # Create token usage
        usage = TokenUsage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(completion_tokens),
            total_tokens=len(prompt_tokens) + len(completion_tokens),
            model_name=model_name
        )
        
        # Calculate cost
        bly_cost = usage.calculate_cost(self.pricing)
        usd_equivalent = bly_cost * self.bly_usd_rate
        
        # Create billing receipt
        receipt = BillingReceipt(
            request_id=request_id,
            user_address=user_address,
            token_usage=usage,
            bly_cost=bly_cost,
            usd_equivalent=usd_equivalent,
            timestamp=time.time(),
            inference_receipt_hash=inference_receipt.compute_hash() if inference_receipt else None
        )
        
        # Store receipt
        self.receipts[request_id] = receipt
        
        # Update metrics
        self.total_tokens_processed += usage.total_tokens
        self.total_bly_billed += bly_cost
        self.total_receipts += 1
        
        # Log receipt
        await self._log_receipt(receipt)
        
        # Add to pending payments
        async with self.batch_lock:
            if user_address not in self.pending_payments:
                self.pending_payments[user_address] = []
            self.pending_payments[user_address].append(receipt)
            
            # Check if batch payment needed
            total_pending = sum(r.bly_cost for r in self.pending_payments[user_address])
            if total_pending >= self.batch_threshold:
                await self._process_batch_payment(user_address)
        
        logger.info(
            "Billing tracked",
            request_id=request_id,
            user=user_address,
            tokens=usage.total_tokens,
            cost=str(bly_cost),
            model=model_name
        )
        
        return receipt
    
    async def _process_batch_payment(self, user_address: str):
        """Process batch payment for user."""
        if not self.ledger_client:
            return
            
        receipts = self.pending_payments.get(user_address, [])
        if not receipts:
            return
            
        total_cost = sum(r.bly_cost for r in receipts)
        
        try:
            # Record payment in ledger
            tx_hash = await self.ledger_client.charge_user(
                user_address=user_address,
                amount=total_cost,
                description=f"Inference batch: {len(receipts)} requests"
            )
            
            # Update receipts with payment info
            for receipt in receipts:
                receipt.payment_tx_hash = tx_hash
                receipt.status = "paid"
            
            # Clear pending
            del self.pending_payments[user_address]
            
            logger.info(
                "Batch payment processed",
                user=user_address,
                amount=str(total_cost),
                requests=len(receipts),
                tx_hash=tx_hash
            )
            
        except Exception as e:
            logger.error(f"Batch payment failed for {user_address}: {e}")
            # Mark as failed
            for receipt in receipts:
                receipt.status = "failed"
    
    async def apply_free_tier(
        self,
        request_id: str,
        user_address: str,
        prompt_tokens: List[int],
        completion_tokens: List[int],
        model_name: str = "gpt_oss_20b"
    ) -> BillingReceipt:
        """Track free tier usage (no billing)."""
        
        # Create usage tracking
        usage = TokenUsage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(completion_tokens),
            total_tokens=len(prompt_tokens) + len(completion_tokens),
            model_name=model_name
        )
        
        # Create receipt with zero cost
        receipt = BillingReceipt(
            request_id=request_id,
            user_address=user_address,
            token_usage=usage,
            bly_cost=Decimal("0"),
            usd_equivalent=Decimal("0"),
            timestamp=time.time(),
            status="free_tier"
        )
        
        # Store and log
        self.receipts[request_id] = receipt
        await self._log_receipt(receipt)
        
        # Update metrics
        self.total_tokens_processed += usage.total_tokens
        self.total_receipts += 1
        
        return receipt
    
    async def _log_receipt(self, receipt: BillingReceipt):
        """Append receipt to log file."""
        try:
            with open(self.receipt_log_path, 'a') as f:
                f.write(json.dumps(receipt.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log billing receipt: {e}")
    
    def get_user_balance(self, user_address: str) -> Dict:
        """Get user's pending balance."""
        pending_receipts = self.pending_payments.get(user_address, [])
        total_pending = sum(r.bly_cost for r in pending_receipts)
        
        return {
            "user_address": user_address,
            "pending_bly": str(total_pending),
            "pending_usd": str(total_pending * self.bly_usd_rate),
            "pending_requests": len(pending_receipts),
            "batch_threshold": str(self.batch_threshold)
        }
    
    def get_metrics(self) -> Dict:
        """Get accounting metrics."""
        return {
            "total_tokens_processed": self.total_tokens_processed,
            "total_bly_billed": str(self.total_bly_billed),
            "total_usd_equivalent": str(self.total_bly_billed * self.bly_usd_rate),
            "total_receipts": self.total_receipts,
            "pending_users": len(self.pending_payments),
            "cached_receipts": len(self.receipts)
        }
    
    async def finalize_batch_payments(self):
        """Process all pending batch payments (call on shutdown)."""
        async with self.batch_lock:
            for user_address in list(self.pending_payments.keys()):
                await self._process_batch_payment(user_address)