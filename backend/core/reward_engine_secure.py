#!/usr/bin/env python3
"""
Secure Reward Engine with race condition prevention
"""

import asyncio
import time
from typing import Dict, Tuple, Optional, Any
import logging

from backend.core.reward_engine import RewardEngine as BaseRewardEngine

logger = logging.getLogger(__name__)

class SecureRewardEngine(BaseRewardEngine):
    """Enhanced reward engine with security improvements."""
    
    def __init__(self, config_path: str = "config/tokenomics.yaml"):
        super().__init__(config_path)
        self._coefficient_lock = asyncio.Lock()
        self._is_calculating = False
        self._calculation_event = asyncio.Event()
        
    async def adjust_coefficients(self) -> Tuple[float, float]:
        """
        Thread-safe coefficient adjustment with race condition prevention.
        """
        # Check if calculation is already in progress
        if self._is_calculating:
            # Wait for ongoing calculation to complete
            await self._calculation_event.wait()
            return self.cache['coefficients']
            
        async with self._coefficient_lock:
            # Double-check after acquiring lock
            if self._is_calculating:
                await self._calculation_event.wait()
                return self.cache['coefficients']
                
            # Check cache validity
            last_update = self.cache.get('coefficients_updated', 0)
            if time.time() - last_update < self.config["controls"]["reward_adjustment_interval"]:
                return self.cache['coefficients']
                
            try:
                self._is_calculating = True
                self._calculation_event.clear()
                
                # Perform actual calculation
                logger.info("Starting coefficient calculation...")
                
                # Get current year's minted tokens
                minted_this_year = self.ledger.get_minted_this_year()
                
                # Calculate remaining budget
                annual_budget = self.config["total_cap"] * self.config["annual_inflation"]
                remaining_budget = annual_budget - minted_this_year
                
                # Predict demand
                days_remaining = self._days_remaining_in_year()
                predicted_token_demand = self.metrics.predict_token_demand(days_remaining)
                predicted_learning_events = self.metrics.predict_learning_events(days_remaining)
                
                # Calculate new coefficients
                if predicted_token_demand > 0:
                    beta = max(
                        self.config["controls"]["min_beta"],
                        remaining_budget * 0.6 / predicted_token_demand
                    )
                else:
                    beta = self.config["inference"]["beta_init"]
                    
                base_reward = self.config["learning"]["base_reward_per_1p"]
                if predicted_learning_events > 0:
                    alpha = min(
                        base_reward * self.config["controls"]["max_alpha_multiplier"],
                        remaining_budget * 0.4 / predicted_learning_events
                    )
                else:
                    alpha = base_reward
                    
                # Update cache atomically
                self.cache['coefficients'] = (alpha, beta)
                self.cache['coefficients_updated'] = time.time()
                
                logger.info(f"Coefficients updated: α={alpha:.2f}, β={beta:.6f}")
                
                return alpha, beta
                
            finally:
                self._is_calculating = False
                self._calculation_event.set()
                
    def verify_supply_consistency(self) -> Dict[str, Any]:
        """
        Verify that all supply numbers add up correctly.
        """
        total_cap = self.config["total_cap"]
        
        # Calculate all components
        genesis = 100_000_000  # Genesis supply
        minted = self.ledger.get_total_minted()
        burned = 0  # TODO: Track burns
        
        # Vesting calculations
        team_total = total_cap * 0.15
        investor_total = total_cap * 0.15
        ecosystem_total = total_cap * 0.10
        foundation_total = total_cap * 0.20
        rewards_total = total_cap * 0.40
        
        # Current circulating
        circulating = genesis + minted - burned
        
        # Locked tokens
        locked = team_total + investor_total + ecosystem_total
        
        # Total accounted
        total_accounted = circulating + locked + foundation_total + (rewards_total - minted)
        
        # Verify consistency
        is_consistent = abs(total_accounted - total_cap) < 1.0  # Allow 1 token rounding error
        
        return {
            "is_consistent": is_consistent,
            "total_cap": total_cap,
            "total_accounted": total_accounted,
            "difference": total_cap - total_accounted,
            "breakdown": {
                "circulating": circulating,
                "locked": locked,
                "foundation": foundation_total,
                "remaining_rewards": rewards_total - minted,
                "minted": minted,
                "burned": burned
            }
        }


class BatchRewardProcessor:
    """
    Batch reward payments to reduce transaction overhead.
    """
    
    def __init__(self, threshold: float = 0.5, interval: int = 3600):
        self.threshold = threshold  # Minimum BLY to trigger payout
        self.interval = interval    # Batch interval in seconds
        self.pending_rewards: Dict[str, float] = {}
        self.last_payout = time.time()
        self._lock = asyncio.Lock()
        
    async def add_reward(self, address: str, amount: float):
        """Add reward to pending batch."""
        async with self._lock:
            self.pending_rewards[address] = self.pending_rewards.get(address, 0) + amount
            
            # Check if we should trigger batch payout
            if (self.pending_rewards[address] >= self.threshold or 
                time.time() - self.last_payout >= self.interval):
                await self._process_batch()
                
    async def _process_batch(self):
        """Process pending rewards in batch."""
        if not self.pending_rewards:
            return
            
        # Get rewards above threshold or if interval passed
        current_time = time.time()
        force_payout = (current_time - self.last_payout) >= self.interval
        
        payouts = {}
        remaining = {}
        
        for address, amount in self.pending_rewards.items():
            if amount >= self.threshold or force_payout:
                payouts[address] = amount
            else:
                remaining[address] = amount
                
        if payouts:
            logger.info(f"Processing batch payout: {len(payouts)} recipients, "
                       f"total: {sum(payouts.values()):.2f} BLY")
            
            # TODO: Implement actual blockchain payout
            # await blockchain.batch_transfer(payouts)
            
            self.pending_rewards = remaining
            self.last_payout = current_time
            
    async def force_payout(self):
        """Force immediate payout of all pending rewards."""
        async with self._lock:
            await self._process_batch()


# Singleton instances
_secure_reward_engine = None
_batch_processor = None

def get_secure_reward_engine() -> SecureRewardEngine:
    """Get or create secure reward engine singleton."""
    global _secure_reward_engine
    if _secure_reward_engine is None:
        _secure_reward_engine = SecureRewardEngine()
    return _secure_reward_engine

def get_batch_processor() -> BatchRewardProcessor:
    """Get or create batch processor singleton."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchRewardProcessor()
    return _batch_processor