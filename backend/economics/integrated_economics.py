#!/usr/bin/env python3
"""
Integrated Economics System for Blyan Network
Coordinates billing, rewards, and budget control for sustainability
"""

import asyncio
import time
from typing import Dict, Optional
from decimal import Decimal
import logging

from backend.economics.billing_gateway import get_billing_gateway
from backend.economics.budget_controller import get_budget_controller, get_learning_scheduler
from backend.economics.training_cost_calculator import TrainingCostCalculator
from backend.core.reward_engine import get_reward_engine

logger = logging.getLogger(__name__)

class IntegratedEconomicsSystem:
    """
    Unified system for sustainable token economics.
    
    Key features:
    1. Inference billing with automatic BLY buy-back
    2. 50% token burn, 50% to reward pool
    3. Budget-constrained reward distribution
    4. Demand-based learning control
    5. Pending queue for rewards when budget exhausted
    """
    
    def __init__(self):
        self.billing = get_billing_gateway()
        self.budget = get_budget_controller()
        self.scheduler = get_learning_scheduler()
        self.rewards = get_reward_engine()
        self.cost_calc = TrainingCostCalculator()
        
        # System state
        self.is_running = False
        self.metrics = {
            "total_inference_requests": 0,
            "total_revenue_usd": Decimal("0"),
            "total_bly_burned": Decimal("0"),
            "total_bly_distributed": Decimal("0"),
            "learning_paused_count": 0
        }
        
    async def process_inference_request(
        self,
        prompt: str,
        tokens_generated: int,
        quality_score: float = 0.95,
        latency_ms: float = 350,
        tier: str = "standard"
    ) -> Dict[str, any]:
        """
        Process a complete inference request with billing and rewards.
        
        Flow:
        1. User pays for inference
        2. BLY is bought and split (burn/pool)
        3. Inference provider gets rewarded (if budget allows)
        """
        result = {
            "prompt": prompt[:50] + "...",
            "tokens": tokens_generated,
            "billing": {},
            "rewards": {}
        }
        
        # Step 1: Process payment
        payment = await self.billing.process_inference_payment(
            tokens_count=tokens_generated,
            tier=tier
        )
        
        if not payment["success"]:
            result["error"] = "Payment failed"
            return result
            
        result["billing"] = payment
        
        # Step 2: Add to reward pool (50% after burn)
        pool_amount = Decimal(str(payment["bly_to_pool"]))
        self.budget.add_revenue(pool_amount)
        
        # Step 3: Calculate inference reward
        inference_reward = self.rewards.reward_inference(
            tokens_served=tokens_generated,
            quality_score=quality_score,
            latency_ms=latency_ms
        )
        
        # Step 4: Request reward payout (may be queued)
        reward_status = await self.budget.request_reward(
            request_type="inference",
            recipient="inference_provider",  # In production: actual address
            amount=Decimal(str(inference_reward)),
            metadata={
                "quality_score": quality_score,
                "latency_ms": latency_ms,
                "tokens": tokens_generated
            }
        )
        
        result["rewards"] = reward_status
        
        # Update metrics
        self.metrics["total_inference_requests"] += 1
        self.metrics["total_revenue_usd"] += Decimal(str(payment["usd_paid"]))
        self.metrics["total_bly_burned"] += Decimal(str(payment["bly_burned"]))
        
        return result
        
    async def process_learning_contribution(
        self,
        model_params: int,
        improvement_percent: float,
        contributor_address: str
    ) -> Dict[str, any]:
        """
        Process a learning/training contribution.
        
        Args:
            model_params: Model size in billions
            improvement_percent: Performance improvement achieved
            contributor_address: Address to receive rewards
            
        Returns:
            Reward status
        """
        # Check if learning is enabled
        can_train, reason = await self.scheduler.check_training_eligibility()
        
        if not can_train:
            return {
                "success": False,
                "error": f"Training disabled: {reason}"
            }
            
        # Calculate training reward
        baseline_loss = 1.0
        delta_loss = baseline_loss * (improvement_percent / 100)
        
        training_reward = self.rewards.reward_training(
            delta_loss=delta_loss,
            baseline_loss=baseline_loss
        )
        
        # Request reward payout
        reward_status = await self.budget.request_reward(
            request_type="learning",
            recipient=contributor_address,
            amount=Decimal(str(training_reward)),
            metadata={
                "model_params": model_params,
                "improvement_percent": improvement_percent
            }
        )
        
        if reward_status["status"] == "paid":
            self.metrics["total_bly_distributed"] += Decimal(str(training_reward))
            
        return reward_status
        
    async def simulate_economics(
        self,
        daily_inference_volume: int = 10_000_000,  # 10M tokens/day
        daily_learning_events: int = 50,
        days: int = 30,
        bly_price: float = 0.10
    ) -> Dict[str, any]:
        """
        Simulate the economics over a period.
        
        Args:
            daily_inference_volume: Tokens processed per day
            daily_learning_events: Training improvements per day
            days: Simulation period
            bly_price: BLY token price
            
        Returns:
            Simulation results
        """
        simulation = {
            "parameters": {
                "daily_inference_volume": daily_inference_volume,
                "daily_learning_events": daily_learning_events,
                "days": days,
                "bly_price": bly_price
            },
            "results": {
                "total_revenue_usd": 0,
                "total_bly_burned": 0,
                "total_bly_to_pool": 0,
                "total_learning_rewards": 0,
                "total_inference_rewards": 0,
                "learning_disabled_days": 0
            }
        }
        
        # Simulate each day
        for day in range(days):
            # Inference revenue
            daily_usd = (daily_inference_volume / 1000) * 0.001  # $0.001 per 1k tokens
            daily_bly = daily_usd / bly_price
            
            simulation["results"]["total_revenue_usd"] += daily_usd
            simulation["results"]["total_bly_burned"] += daily_bly * 0.5
            simulation["results"]["total_bly_to_pool"] += daily_bly * 0.5
            
            # Learning costs (if enabled)
            demand_ratio = (daily_bly * 0.5) / (daily_learning_events * 100)  # Rough estimate
            
            if demand_ratio >= 0.6:
                # Learning enabled
                daily_learning_rewards = daily_learning_events * 100  # 100 BLY per event average
                simulation["results"]["total_learning_rewards"] += daily_learning_rewards
            else:
                # Learning disabled due to low demand
                simulation["results"]["learning_disabled_days"] += 1
                
            # Inference rewards
            daily_inference_rewards = daily_bly * 0.3  # 30% of revenue as rewards
            simulation["results"]["total_inference_rewards"] += daily_inference_rewards
            
        # Calculate sustainability metrics
        total_rewards = (simulation["results"]["total_learning_rewards"] + 
                        simulation["results"]["total_inference_rewards"])
        total_pool_income = simulation["results"]["total_bly_to_pool"]
        
        simulation["sustainability"] = {
            "pool_income": total_pool_income,
            "total_rewards": total_rewards,
            "surplus_deficit": total_pool_income - total_rewards,
            "sustainable": total_pool_income >= total_rewards,
            "burn_rate": simulation["results"]["total_bly_burned"] / days,
            "learning_uptime": 1 - (simulation["results"]["learning_disabled_days"] / days)
        }
        
        return simulation
        
    def calculate_breakeven_volume(
        self,
        target_learning_rewards_daily: float = 5000,  # BLY
        bly_price: float = 0.10
    ) -> Dict[str, any]:
        """
        Calculate required inference volume for breakeven.
        
        Args:
            target_learning_rewards_daily: Daily learning rewards target
            bly_price: Current BLY price
            
        Returns:
            Breakeven analysis
        """
        # Need to cover learning rewards + inference rewards
        # Assuming 30% of pool goes to inference, 40% to learning
        total_daily_rewards = target_learning_rewards_daily / 0.4
        
        # Pool gets 50% of revenue after burn
        required_daily_revenue_bly = total_daily_rewards * 2
        
        # Convert to USD
        required_daily_revenue_usd = required_daily_revenue_bly * bly_price
        
        # Calculate required token volume
        price_per_1k = 0.001  # $0.001 per 1k tokens
        required_daily_tokens = (required_daily_revenue_usd / price_per_1k) * 1000
        
        return {
            "target_learning_rewards_daily": target_learning_rewards_daily,
            "required_daily_revenue_usd": required_daily_revenue_usd,
            "required_daily_revenue_bly": required_daily_revenue_bly,
            "required_daily_tokens": required_daily_tokens,
            "required_monthly_tokens": required_daily_tokens * 30,
            "bly_price": bly_price,
            "daily_burn_bly": required_daily_revenue_bly * 0.5
        }
        
    async def start_monitoring(self):
        """Start the economics monitoring loop."""
        self.is_running = True
        
        # Start scheduler monitoring
        asyncio.create_task(self.scheduler.monitor_loop())
        
        # Start metrics reporting
        asyncio.create_task(self._report_metrics_loop())
        
        logger.info("Integrated economics system started")
        
    async def _report_metrics_loop(self):
        """Periodically report system metrics."""
        while self.is_running:
            try:
                # Get current status
                budget_status = self.budget.get_budget_status()
                billing_stats = self.billing.get_revenue_stats()
                reward_stats = self.rewards.get_reward_stats()
                
                # Log summary
                logger.info(f"Economics Status:")
                logger.info(f"  Pool Balance: {budget_status['pool_balance']:.2f} BLY")
                logger.info(f"  Budget Utilization: {budget_status['budget_utilization']:.1%}")
                logger.info(f"  Training Enabled: {budget_status['training_enabled']}")
                logger.info(f"  Demand Ratio: {budget_status['demand_ratio']:.2f}")
                logger.info(f"  Pending Queue: {budget_status['pending_queue_size']} requests")
                logger.info(f"  Total Burned: {billing_stats['total_burned']:.2f} BLY")
                logger.info(f"  Current BLY Price: ${billing_stats['current_bly_price']:.3f}")
                logger.info(f"  Inflation Rate: {reward_stats['inflation']['current_rate']:.2%}")
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in metrics reporting: {e}")
                await asyncio.sleep(300)
                
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status."""
        return {
            "metrics": self.metrics,
            "budget": self.budget.get_budget_status(),
            "billing": self.billing.get_revenue_stats(),
            "rewards": self.rewards.get_reward_stats(),
            "training": {
                "enabled": self.scheduler.training_paused == False,
                "pause_reason": self.scheduler.pause_reason
            }
        }

# Singleton
_integrated_system = None

def get_integrated_economics() -> IntegratedEconomicsSystem:
    """Get or create integrated economics system."""
    global _integrated_system
    if _integrated_system is None:
        _integrated_system = IntegratedEconomicsSystem()
    return _integrated_system

async def main():
    """Test the integrated system."""
    system = get_integrated_economics()
    
    print("=== Blyan Network Economics Simulation ===\n")
    
    # 1. Calculate breakeven
    print("1. Breakeven Analysis:")
    breakeven = system.calculate_breakeven_volume(
        target_learning_rewards_daily=5000,
        bly_price=0.10
    )
    print(f"  Required daily tokens: {breakeven['required_daily_tokens']:,.0f}")
    print(f"  Required daily revenue: ${breakeven['required_daily_revenue_usd']:,.2f}")
    print(f"  Daily BLY burn: {breakeven['daily_burn_bly']:,.0f}")
    
    # 2. Simulate 30 days
    print("\n2. 30-Day Economic Simulation:")
    simulation = await system.simulate_economics(
        daily_inference_volume=50_000_000,  # 50M tokens/day
        daily_learning_events=100,
        days=30,
        bly_price=0.10
    )
    
    print(f"  Total Revenue: ${simulation['results']['total_revenue_usd']:,.2f}")
    print(f"  Total BLY Burned: {simulation['results']['total_bly_burned']:,.0f}")
    print(f"  Total Rewards Paid: {simulation['sustainability']['total_rewards']:,.0f}")
    print(f"  Surplus/Deficit: {simulation['sustainability']['surplus_deficit']:,.0f} BLY")
    print(f"  Sustainable: {simulation['sustainability']['sustainable']}")
    print(f"  Learning Uptime: {simulation['sustainability']['learning_uptime']:.1%}")
    
    # 3. Model training costs
    print("\n3. Model Training Costs:")
    calc = TrainingCostCalculator()
    
    for model_size in [20, 120]:
        costs = calc.calculate_training_cost(model_size, 300)
        rewards = calc.calculate_bly_rewards(model_size, 300, 0.10)
        print(f"\n  GPT-{model_size}B:")
        print(f"    USD Cost: ${costs['total_cost_usd']:,.2f}")
        print(f"    BLY Required: {rewards['total_bly_required']:,.0f}")
        print(f"    Training Days: {costs['distributed_hours']/24:.1f}")
        print(f"    GPUs Needed: {costs['gpus_needed']}")

if __name__ == "__main__":
    asyncio.run(main())