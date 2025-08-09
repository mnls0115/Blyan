#!/usr/bin/env python3
"""
Budget Controller for Blyan Network
Implements Budget Envelope and Demand-based training control
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class RewardRequest:
    """Pending reward request."""
    request_id: str
    request_type: str  # 'learning' or 'inference'
    recipient: str
    amount: Decimal
    timestamp: float
    metadata: Dict = field(default_factory=dict)
    
    # Priority factors for weighted processing
    performance_delta: float = 0.0  # Learning improvement percentage
    quality_score: float = 0.0      # Inference quality score
    priority_weight: float = 1.0     # Calculated priority weight
    
    def calculate_priority(self) -> float:
        """
        Calculate priority weight for processing order.
        
        Priority Algorithm:
        1. FIFO base weight: 1.0 - (time_waited / max_wait_time)
        2. Performance multiplier: 1 + (performance_delta / 100)
        3. Quality multiplier: quality_score
        
        Higher priority = processed first
        """
        # Time factor (older = higher priority)
        time_waited = time.time() - self.timestamp
        max_wait_time = 86400  # 24 hours
        time_weight = min(1.0 + (time_waited / max_wait_time), 2.0)
        
        # Performance factor (better improvement = higher priority)
        if self.request_type == "learning":
            perf_weight = 1.0 + (self.performance_delta / 100)
        else:
            perf_weight = self.quality_score if self.quality_score > 0 else 1.0
            
        # Combined priority
        self.priority_weight = time_weight * perf_weight
        return self.priority_weight

class BudgetController:
    """
    Controls reward distribution with budget constraints.
    Prevents over-spending and maintains sustainable economics.
    """
    
    def __init__(self):
        # Budget parameters
        self.BUDGET_RATIO = Decimal("0.20")  # Max 20% of pool per week
        self.MIN_DEMAND_RATIO = 0.6  # Need 60% revenue/cost ratio to enable training
        
        # State
        self.reward_pool_balance = Decimal("100000")  # Initial pool
        self.pending_queue = deque()  # Pending rewards when budget exhausted
        self.weekly_spent = Decimal("0")
        self.last_week_reset = time.time()
        
        # Metrics
        self.inference_revenue_7d = deque(maxlen=7)  # Daily revenues
        self.learning_cost_7d = deque(maxlen=7)  # Daily costs
        self.training_enabled = True
        
    def can_afford_reward(self, amount: Decimal) -> bool:
        """Check if reward can be paid within budget constraints."""
        # Reset weekly counter if needed
        self._check_weekly_reset()
        
        # Calculate weekly budget
        weekly_budget = self.reward_pool_balance * self.BUDGET_RATIO
        
        # Check if within budget
        return self.weekly_spent + amount <= weekly_budget
        
    async def request_reward(
        self,
        request_type: str,
        recipient: str,
        amount: Decimal,
        metadata: Dict = None
    ) -> Dict[str, any]:
        """
        Request a reward payout with budget control.
        
        Args:
            request_type: 'learning' or 'inference'
            recipient: Address to receive reward
            amount: BLY amount requested
            metadata: Additional context
            
        Returns:
            Status of reward request
        """
        request = RewardRequest(
            request_id=f"{request_type}_{time.time()}",
            request_type=request_type,
            recipient=recipient,
            amount=amount,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Check if we can afford it
        if self.can_afford_reward(amount):
            # Pay immediately
            await self._pay_reward(request)
            return {
                "status": "paid",
                "request_id": request.request_id,
                "amount": float(amount)
            }
        else:
            # Add to pending queue
            self.pending_queue.append(request)
            logger.info(f"Queued reward {request.request_id} - budget exhausted")
            return {
                "status": "queued",
                "request_id": request.request_id,
                "amount": float(amount),
                "queue_position": len(self.pending_queue)
            }
            
    async def _pay_reward(self, request: RewardRequest):
        """Execute reward payment."""
        # Deduct from pool
        self.reward_pool_balance -= request.amount
        self.weekly_spent += request.amount
        
        # Track costs
        if request.request_type == "learning":
            today_cost = self.learning_cost_7d[-1] if self.learning_cost_7d else Decimal("0")
            if len(self.learning_cost_7d) == 0 or self._is_new_day():
                self.learning_cost_7d.append(request.amount)
            else:
                self.learning_cost_7d[-1] = today_cost + request.amount
                
        # In production: execute blockchain transfer
        logger.info(f"Paid {request.amount} BLY to {request.recipient}")
        
    async def process_pending_queue(self, processing_mode: str = "weighted_fifo"):
        """
        Process pending rewards when budget allows.
        
        Processing modes:
        - "fifo": Pure first-in-first-out
        - "weighted_fifo": FIFO with performance weighting (default)
        - "performance": Pure performance-based priority
        
        Algorithm:
        1. Calculate priority for all pending requests
        2. Sort by priority (descending)
        3. Process in priority order until budget exhausted
        """
        processed = []
        
        if not self.pending_queue:
            return processed
            
        # Convert queue to list for sorting
        pending_list = list(self.pending_queue)
        
        # Calculate priorities based on mode
        if processing_mode == "fifo":
            # Pure FIFO - sort by timestamp only
            pending_list.sort(key=lambda r: r.timestamp)
        elif processing_mode == "performance":
            # Pure performance - sort by performance only
            for request in pending_list:
                if request.request_type == "learning":
                    request.priority_weight = request.performance_delta
                else:
                    request.priority_weight = request.quality_score * 100
            pending_list.sort(key=lambda r: r.priority_weight, reverse=True)
        else:  # weighted_fifo (default)
            # Calculate weighted priority for each request
            for request in pending_list:
                request.calculate_priority()
            pending_list.sort(key=lambda r: r.priority_weight, reverse=True)
            
        # Process in priority order
        still_pending = []
        for request in pending_list:
            if self.can_afford_reward(request.amount):
                await self._pay_reward(request)
                processed.append(request.request_id)
                logger.info(f"Processed {request.request_id} (priority: {request.priority_weight:.2f})")
            else:
                still_pending.append(request)
                
        # Update queue with remaining requests
        self.pending_queue = deque(still_pending)
        
        if processed:
            logger.info(f"Processed {len(processed)} pending rewards ({processing_mode} mode)")
            logger.info(f"Remaining in queue: {len(self.pending_queue)}")
            
        return processed
        
    def update_demand_metrics(self, daily_revenue: Decimal, daily_cost: Decimal):
        """Update demand tracking metrics."""
        # Update rolling windows
        if self._is_new_day():
            self.inference_revenue_7d.append(daily_revenue)
            self.learning_cost_7d.append(daily_cost)
            
        # Calculate demand ratio
        demand_ratio = self.calculate_demand_ratio()
        
        # Update training enable flag
        self.training_enabled = demand_ratio >= self.MIN_DEMAND_RATIO
        
        if not self.training_enabled:
            logger.warning(f"Training disabled - demand ratio {demand_ratio:.2f} < {self.MIN_DEMAND_RATIO}")
            
    def calculate_demand_ratio(self) -> float:
        """
        Calculate revenue/cost ratio for demand-based control.
        """
        if not self.inference_revenue_7d or not self.learning_cost_7d:
            return 1.0  # Default to enabled if no data
            
        total_revenue = sum(self.inference_revenue_7d)
        total_cost = sum(self.learning_cost_7d)
        
        if total_cost == 0:
            return float('inf')
            
        return float(total_revenue / total_cost)
        
    def should_enable_training(self) -> bool:
        """Check if training should be enabled based on demand."""
        return self.training_enabled
        
    def add_revenue(self, amount: Decimal):
        """Add revenue to the reward pool from billing gateway."""
        self.reward_pool_balance += amount
        
        # Track revenue
        today_revenue = self.inference_revenue_7d[-1] if self.inference_revenue_7d else Decimal("0")
        if len(self.inference_revenue_7d) == 0 or self._is_new_day():
            self.inference_revenue_7d.append(amount)
        else:
            self.inference_revenue_7d[-1] = today_revenue + amount
            
        logger.info(f"Added {amount} BLY to reward pool (balance: {self.reward_pool_balance})")
        
    def get_budget_status(self) -> Dict[str, any]:
        """Get current budget and queue status."""
        self._check_weekly_reset()
        
        weekly_budget = self.reward_pool_balance * self.BUDGET_RATIO
        budget_remaining = weekly_budget - self.weekly_spent
        
        return {
            "pool_balance": float(self.reward_pool_balance),
            "weekly_budget": float(weekly_budget),
            "weekly_spent": float(self.weekly_spent),
            "budget_remaining": float(budget_remaining),
            "budget_utilization": float(self.weekly_spent / max(1, weekly_budget)),
            "pending_queue_size": len(self.pending_queue),
            "pending_total": float(sum(r.amount for r in self.pending_queue)),
            "demand_ratio": self.calculate_demand_ratio(),
            "training_enabled": self.training_enabled,
            "metrics": {
                "revenue_7d": float(sum(self.inference_revenue_7d)),
                "cost_7d": float(sum(self.learning_cost_7d))
            }
        }
        
    def _check_weekly_reset(self):
        """Reset weekly counter if new week."""
        current_time = time.time()
        week_seconds = 7 * 24 * 3600
        
        if current_time - self.last_week_reset > week_seconds:
            self.weekly_spent = Decimal("0")
            self.last_week_reset = current_time
            logger.info("Weekly budget counter reset")
            
    def _is_new_day(self) -> bool:
        """Check if it's a new day for metrics."""
        from datetime import datetime
        
        if not hasattr(self, '_last_day'):
            self._last_day = datetime.now().day
            return True
            
        current_day = datetime.now().day
        if current_day != self._last_day:
            self._last_day = current_day
            return True
            
        return False

class LearningScheduler:
    """
    Controls learning/training activation based on demand.
    """
    
    def __init__(self, budget_controller: BudgetController):
        self.budget_controller = budget_controller
        self.training_paused = False
        self.pause_reason = ""
        
    async def check_training_eligibility(self) -> Tuple[bool, str]:
        """
        Check if training should be active.
        
        Returns:
            (can_train, reason)
        """
        # Check budget controller
        if not self.budget_controller.should_enable_training():
            return False, "Insufficient demand (revenue/cost ratio < 0.6)"
            
        # Check pool balance
        status = self.budget_controller.get_budget_status()
        if status["pool_balance"] < 1000:  # Minimum pool threshold
            return False, "Reward pool balance too low"
            
        # Check budget utilization
        if status["budget_utilization"] > 0.95:
            return False, "Weekly budget nearly exhausted"
            
        return True, "Training enabled"
        
    async def pause_training(self, reason: str):
        """Pause all training activities."""
        self.training_paused = True
        self.pause_reason = reason
        logger.warning(f"Training paused: {reason}")
        
        # In production: notify all training nodes
        
    async def resume_training(self):
        """Resume training if conditions allow."""
        can_train, reason = await self.check_training_eligibility()
        
        if can_train:
            self.training_paused = False
            self.pause_reason = ""
            logger.info("Training resumed")
        else:
            logger.warning(f"Cannot resume training: {reason}")
            
    async def monitor_loop(self):
        """Continuous monitoring loop for training control."""
        while True:
            try:
                # Check conditions
                can_train, reason = await self.check_training_eligibility()
                
                if can_train and self.training_paused:
                    await self.resume_training()
                elif not can_train and not self.training_paused:
                    await self.pause_training(reason)
                    
                # Process any pending rewards
                await self.budget_controller.process_pending_queue()
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(60)

# Singleton instances
_budget_controller = None
_learning_scheduler = None

def get_budget_controller() -> BudgetController:
    """Get or create budget controller singleton."""
    global _budget_controller
    if _budget_controller is None:
        _budget_controller = BudgetController()
    return _budget_controller
    
def get_learning_scheduler() -> LearningScheduler:
    """Get or create learning scheduler singleton."""
    global _learning_scheduler
    if _learning_scheduler is None:
        controller = get_budget_controller()
        _learning_scheduler = LearningScheduler(controller)
    return _learning_scheduler