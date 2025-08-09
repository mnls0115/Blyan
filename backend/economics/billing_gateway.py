#!/usr/bin/env python3
"""
Billing Gateway for Blyan Network
Handles fiat/crypto payments and automatic BLY buy-back mechanism
"""

import asyncio
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class PricingTier:
    """Pricing tier for inference services."""
    name: str
    price_per_1k_tokens: Decimal  # in USD
    quality_multiplier: float
    
PRICING_TIERS = {
    "basic": PricingTier("basic", Decimal("0.0008"), 0.8),
    "standard": PricingTier("standard", Decimal("0.0010"), 1.0),  
    "premium": PricingTier("premium", Decimal("0.0015"), 1.2)
}

class BillingGateway:
    """
    Manages inference billing with automatic BLY buy-back and burn mechanism.
    
    Flow:
    1. User pays in USD/USDC
    2. Gateway buys BLY from DEX
    3. 50% burned, 50% to reward pool
    """
    
    def __init__(self):
        self.bly_price = Decimal("0.10")  # Initial price
        self.total_burned = Decimal("0")
        self.total_to_pool = Decimal("0")
        self.revenue_stats = {
            "daily": [],
            "weekly": [],
            "monthly": []
        }
        
    async def process_inference_payment(
        self,
        tokens_count: int,
        tier: str = "standard",
        payment_method: str = "usdc"
    ) -> Dict[str, any]:
        """
        Process payment for inference request.
        
        Args:
            tokens_count: Number of tokens in request
            tier: Service tier (basic/standard/premium)
            payment_method: Payment method (usdc/card)
            
        Returns:
            Payment receipt with BLY allocation details
        """
        # Calculate USD cost
        pricing = PRICING_TIERS.get(tier, PRICING_TIERS["standard"])
        usd_cost = (Decimal(tokens_count) / 1000) * pricing.price_per_1k_tokens
        
        # Simulate payment processing
        payment_success = await self._process_payment(payment_method, usd_cost)
        if not payment_success:
            return {"success": False, "error": "Payment failed"}
            
        # Buy BLY with collected USD
        bly_amount = await self._buy_bly_from_dex(usd_cost)
        
        # Split: 50% burn, 50% to reward pool
        burn_amount = bly_amount * Decimal("0.5")
        pool_amount = bly_amount * Decimal("0.5")
        
        # Execute burn
        await self._burn_tokens(burn_amount)
        
        # Transfer to reward pool
        await self._transfer_to_pool(pool_amount)
        
        # Update stats
        self.total_burned += burn_amount
        self.total_to_pool += pool_amount
        
        # Record revenue
        self._record_revenue(usd_cost, bly_amount)
        
        return {
            "success": True,
            "usd_paid": float(usd_cost),
            "bly_bought": float(bly_amount),
            "bly_burned": float(burn_amount),
            "bly_to_pool": float(pool_amount),
            "tier": tier,
            "tokens": tokens_count
        }
        
    async def _process_payment(self, method: str, amount: Decimal) -> bool:
        """Process USD payment (simulated)."""
        # In production: integrate with Stripe/Coinbase Commerce
        await asyncio.sleep(0.1)  # Simulate payment processing
        logger.info(f"Processed {method} payment: ${amount}")
        return True
        
    async def _buy_bly_from_dex(self, usd_amount: Decimal) -> Decimal:
        """
        Buy BLY from DEX with USD.
        Uses 30-day TWAP (Time-Weighted Average Price) for stability.
        """
        # Get current BLY price (in production: from DEX oracle)
        current_price = await self._get_bly_price()
        
        # Calculate BLY amount
        bly_amount = usd_amount / current_price
        
        # Simulate DEX trade with 0.3% slippage
        slippage = Decimal("0.003")
        actual_bly = bly_amount * (1 - slippage)
        
        logger.info(f"Bought {actual_bly} BLY for ${usd_amount}")
        return actual_bly
        
    async def _burn_tokens(self, amount: Decimal):
        """Burn BLY tokens by sending to burn address."""
        # In production: send to 0x0000...dead or contract burn function
        logger.info(f"Burned {amount} BLY")
        
    async def _transfer_to_pool(self, amount: Decimal):
        """Transfer BLY to reward pool."""
        # In production: transfer to RewardPool contract
        logger.info(f"Transferred {amount} BLY to reward pool")
        
    async def _get_bly_price(self) -> Decimal:
        """
        Get current BLY price from DEX.
        Uses TWAP to prevent manipulation.
        """
        # In production: query Uniswap/PancakeSwap oracle
        # For now, simulate price movement
        import random
        volatility = Decimal(str(random.uniform(0.95, 1.05)))
        self.bly_price *= volatility
        self.bly_price = max(Decimal("0.01"), self.bly_price)  # Floor price
        return self.bly_price
        
    def _record_revenue(self, usd: Decimal, bly: Decimal):
        """Record revenue for analytics."""
        record = {
            "timestamp": time.time(),
            "usd": float(usd),
            "bly": float(bly),
            "price": float(self.bly_price)
        }
        
        # Add to daily stats
        self.revenue_stats["daily"].append(record)
        
        # Cleanup old records (keep 30 days)
        cutoff = time.time() - (30 * 24 * 3600)
        self.revenue_stats["daily"] = [
            r for r in self.revenue_stats["daily"] 
            if r["timestamp"] > cutoff
        ]
        
    def get_revenue_stats(self) -> Dict[str, any]:
        """Get revenue and burn statistics."""
        from datetime import datetime, timedelta
        
        now = time.time()
        day_ago = now - 86400
        week_ago = now - (7 * 86400)
        
        daily_revenue = sum(
            r["usd"] for r in self.revenue_stats["daily"]
            if r["timestamp"] > day_ago
        )
        
        weekly_revenue = sum(
            r["usd"] for r in self.revenue_stats["daily"]
            if r["timestamp"] > week_ago
        )
        
        return {
            "total_burned": float(self.total_burned),
            "total_to_pool": float(self.total_to_pool),
            "current_bly_price": float(self.bly_price),
            "revenue": {
                "daily": daily_revenue,
                "weekly": weekly_revenue,
                "monthly": weekly_revenue * 4.3  # Approximation
            },
            "burn_rate": {
                "daily": float(self.total_burned) / max(1, len(self.revenue_stats["daily"])),
                "projected_annual": float(self.total_burned) * 365 / max(1, len(self.revenue_stats["daily"]))
            }
        }
        
    async def calculate_sustainable_rewards(self) -> Tuple[float, float]:
        """
        Calculate sustainable reward rates based on revenue.
        Returns (learning_reward_budget, inference_reward_budget).
        """
        stats = self.get_revenue_stats()
        
        # Monthly revenue projection
        monthly_revenue = stats["revenue"]["monthly"]
        
        # Convert to BLY at current price
        monthly_bly_revenue = monthly_revenue / float(self.bly_price)
        
        # 50% goes to pool after burn
        monthly_pool_addition = monthly_bly_revenue * 0.5
        
        # Allocate pool: 40% learning, 60% inference
        learning_budget = monthly_pool_addition * 0.4
        inference_budget = monthly_pool_addition * 0.6
        
        return learning_budget, inference_budget

# Singleton
_gateway = None

def get_billing_gateway() -> BillingGateway:
    """Get or create billing gateway singleton."""
    global _gateway
    if _gateway is None:
        _gateway = BillingGateway()
    return _gateway