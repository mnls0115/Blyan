#!/usr/bin/env python3
"""
Automatic Reward Distribution System
Distributes rewards based on dataset contributions and model improvements
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RewardDistribution:
    """Reward distribution record."""
    distribution_id: str
    distribution_type: str  # dataset, inference, learning, validation
    recipient_address: str
    amount: Decimal
    reason: str
    metrics: Dict[str, Any]
    distributed_at: datetime
    transaction_id: Optional[str] = None
    status: str = "pending"  # pending, distributed, failed

@dataclass
class RewardPolicy:
    """Reward distribution policy."""
    dataset_base_reward: Decimal = Decimal("100")
    dataset_quality_multiplier: Decimal = Decimal("10")
    inference_reward_per_token: Decimal = Decimal("0.001")
    learning_improvement_reward: Decimal = Decimal("500")
    validation_reward: Decimal = Decimal("50")
    daily_budget: Decimal = Decimal("100000")
    min_payout_threshold: Decimal = Decimal("10")

class AutomaticRewardDistributor:
    """
    Automatically distributes rewards based on contributions.
    """
    
    def __init__(self, distribution_interval_seconds: int = 3600):
        self.distribution_interval = distribution_interval_seconds
        self.policy = RewardPolicy()
        self.pending_rewards: Dict[str, List[RewardDistribution]] = {}
        self.distribution_history: List[RewardDistribution] = []
        self.is_running = False
        self.distribution_task = None
        
        # Load distribution history
        self._load_history()
    
    def _load_history(self):
        """Load distribution history from disk."""
        history_file = Path("./data/rewards/distribution_history.json")
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                for dist_data in data:
                    dist_data['amount'] = Decimal(dist_data['amount'])
                    dist_data['distributed_at'] = datetime.fromisoformat(dist_data['distributed_at'])
                    self.distribution_history.append(RewardDistribution(**dist_data))
    
    def _save_history(self):
        """Save distribution history to disk."""
        history_file = Path("./data/rewards/distribution_history.json")
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for dist in self.distribution_history[-1000:]:  # Keep last 1000 records
            dist_dict = asdict(dist)
            dist_dict['amount'] = str(dist_dict['amount'])
            dist_dict['distributed_at'] = dist_dict['distributed_at'].isoformat()
            data.append(dist_dict)
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def start(self):
        """Start automatic distribution loop."""
        if self.is_running:
            logger.warning("Distribution already running")
            return
        
        self.is_running = True
        self.distribution_task = asyncio.create_task(self._distribution_loop())
        logger.info("Automatic reward distribution started")
    
    async def stop(self):
        """Stop automatic distribution."""
        self.is_running = False
        if self.distribution_task:
            self.distribution_task.cancel()
            try:
                await self.distribution_task
            except asyncio.CancelledError:
                pass
        logger.info("Automatic reward distribution stopped")
    
    async def _distribution_loop(self):
        """Main distribution loop."""
        while self.is_running:
            try:
                await self._run_distribution_cycle()
                await asyncio.sleep(self.distribution_interval)
            except Exception as e:
                logger.error(f"Distribution cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _run_distribution_cycle(self):
        """Run a single distribution cycle."""
        logger.info("Starting reward distribution cycle")
        
        # Collect rewards from various sources
        dataset_rewards = await self._collect_dataset_rewards()
        inference_rewards = await self._collect_inference_rewards()
        learning_rewards = await self._collect_learning_rewards()
        validation_rewards = await self._collect_validation_rewards()
        
        # Aggregate rewards by recipient
        all_rewards = dataset_rewards + inference_rewards + learning_rewards + validation_rewards
        aggregated = self._aggregate_rewards(all_rewards)
        
        # Apply daily budget limits
        limited_rewards = await self._apply_budget_limits(aggregated)
        
        # Distribute rewards
        distributed_count = 0
        failed_count = 0
        
        for recipient, rewards in limited_rewards.items():
            total_amount = sum(r.amount for r in rewards)
            
            # Check minimum payout threshold
            if total_amount < self.policy.min_payout_threshold:
                # Add to pending for next cycle
                if recipient not in self.pending_rewards:
                    self.pending_rewards[recipient] = []
                self.pending_rewards[recipient].extend(rewards)
                continue
            
            # Distribute reward
            success = await self._distribute_reward(recipient, rewards, total_amount)
            
            if success:
                distributed_count += len(rewards)
            else:
                failed_count += len(rewards)
        
        logger.info(
            f"Distribution cycle complete: {distributed_count} distributed, "
            f"{failed_count} failed, {len(self.pending_rewards)} pending"
        )
    
    async def _collect_dataset_rewards(self) -> List[RewardDistribution]:
        """Collect rewards for dataset contributions."""
        rewards = []
        
        # Import PoDL recorder
        from backend.data.podl_score_system import get_podl_recorder
        recorder = get_podl_recorder()
        
        # Get recent validated contributions
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.distribution_interval)
        
        for contrib in recorder.contributions.values():
            if (contrib.validation_status == "validated" and 
                contrib.validated_at and 
                contrib.validated_at > cutoff_time and
                contrib.reward_amount):
                
                reward = RewardDistribution(
                    distribution_id=f"dataset_{contrib.dataset_id}_{int(time.time())}",
                    distribution_type="dataset",
                    recipient_address=contrib.contributor_address,
                    amount=contrib.reward_amount,
                    reason=f"Dataset contribution {contrib.dataset_id} with PoDL score {contrib.podl_score:.3f}",
                    metrics={
                        "dataset_id": contrib.dataset_id,
                        "podl_score": contrib.podl_score,
                        "quality_score": contrib.quality_score,
                        "samples": contrib.sample_count
                    },
                    distributed_at=datetime.utcnow()
                )
                rewards.append(reward)
        
        return rewards
    
    async def _collect_inference_rewards(self) -> List[RewardDistribution]:
        """Collect rewards for inference contributions."""
        rewards = []
        
        # Import usage tracker
        from backend.model.moe_infer import get_expert_usage_tracker
        tracker = get_expert_usage_tracker()
        
        # Get recent usage stats
        cutoff_time = time.time() - self.distribution_interval
        
        # Aggregate by provider
        provider_stats = {}
        for expert_name, usage in tracker.usage_stats.items():
            if usage['last_used'] > cutoff_time:
                provider = usage.get('provider_address')
                if provider:
                    if provider not in provider_stats:
                        provider_stats[provider] = {
                            'total_uses': 0,
                            'total_tokens': 0,
                            'avg_quality': []
                        }
                    provider_stats[provider]['total_uses'] += usage['use_count']
                    provider_stats[provider]['total_tokens'] += usage.get('tokens_generated', 0)
                    if usage['quality_score'] > 0:
                        provider_stats[provider]['avg_quality'].append(usage['quality_score'])
        
        # Create rewards
        for provider, stats in provider_stats.items():
            if stats['total_tokens'] > 0:
                token_reward = self.policy.inference_reward_per_token * Decimal(stats['total_tokens'])
                
                # Quality bonus
                avg_quality = sum(stats['avg_quality']) / len(stats['avg_quality']) if stats['avg_quality'] else 0.5
                quality_multiplier = Decimal(str(1 + (avg_quality - 0.5)))  # 0.5x to 1.5x
                
                total_reward = token_reward * quality_multiplier
                
                reward = RewardDistribution(
                    distribution_id=f"inference_{provider[:8]}_{int(time.time())}",
                    distribution_type="inference",
                    recipient_address=provider,
                    amount=total_reward,
                    reason=f"Inference services: {stats['total_tokens']} tokens, {stats['total_uses']} requests",
                    metrics={
                        "tokens": stats['total_tokens'],
                        "requests": stats['total_uses'],
                        "quality": round(avg_quality, 3)
                    },
                    distributed_at=datetime.utcnow()
                )
                rewards.append(reward)
        
        return rewards
    
    async def _collect_learning_rewards(self) -> List[RewardDistribution]:
        """Collect rewards for model improvements."""
        rewards = []
        
        # Check for model improvement submissions
        improvements_file = Path("./data/model_improvements.json")
        if improvements_file.exists():
            with open(improvements_file, 'r') as f:
                improvements = json.load(f)
            
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.distribution_interval)
            
            for improvement in improvements:
                submitted_at = datetime.fromisoformat(improvement['submitted_at'])
                if submitted_at > cutoff_time and improvement.get('validated'):
                    # Calculate reward based on improvement percentage
                    improvement_pct = improvement.get('improvement_percentage', 0)
                    if improvement_pct > 0:
                        reward_amount = self.policy.learning_improvement_reward * Decimal(str(improvement_pct / 100))
                        
                        reward = RewardDistribution(
                            distribution_id=f"learning_{improvement['id']}_{int(time.time())}",
                            distribution_type="learning",
                            recipient_address=improvement['contributor'],
                            amount=reward_amount,
                            reason=f"Model improvement: {improvement_pct}% on {improvement['metric']}",
                            metrics={
                                "improvement_id": improvement['id'],
                                "improvement_pct": improvement_pct,
                                "metric": improvement['metric']
                            },
                            distributed_at=datetime.utcnow()
                        )
                        rewards.append(reward)
        
        return rewards
    
    async def _collect_validation_rewards(self) -> List[RewardDistribution]:
        """Collect rewards for validation work."""
        rewards = []
        
        # Check validation log
        validation_log = Path("./data/validation_log.json")
        if validation_log.exists():
            with open(validation_log, 'r') as f:
                validations = json.load(f)
            
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.distribution_interval)
            
            for validation in validations:
                validated_at = datetime.fromisoformat(validation['timestamp'])
                if validated_at > cutoff_time:
                    reward = RewardDistribution(
                        distribution_id=f"validation_{validation['id']}_{int(time.time())}",
                        distribution_type="validation",
                        recipient_address=validation['validator'],
                        amount=self.policy.validation_reward,
                        reason=f"Validated {validation['target_type']}: {validation['target_id']}",
                        metrics={
                            "validation_id": validation['id'],
                            "target_type": validation['target_type'],
                            "target_id": validation['target_id']
                        },
                        distributed_at=datetime.utcnow()
                    )
                    rewards.append(reward)
        
        return rewards
    
    def _aggregate_rewards(self, rewards: List[RewardDistribution]) -> Dict[str, List[RewardDistribution]]:
        """Aggregate rewards by recipient."""
        aggregated = {}
        
        # Add pending rewards first
        for recipient, pending in self.pending_rewards.items():
            aggregated[recipient] = pending.copy()
        
        # Add new rewards
        for reward in rewards:
            if reward.recipient_address not in aggregated:
                aggregated[reward.recipient_address] = []
            aggregated[reward.recipient_address].append(reward)
        
        # Clear pending rewards (they're now in aggregated)
        self.pending_rewards.clear()
        
        return aggregated
    
    async def _apply_budget_limits(
        self,
        aggregated: Dict[str, List[RewardDistribution]]
    ) -> Dict[str, List[RewardDistribution]]:
        """Apply daily budget limits."""
        # Calculate total for this cycle
        total_amount = sum(
            sum(r.amount for r in rewards)
            for rewards in aggregated.values()
        )
        
        if total_amount <= self.policy.daily_budget:
            return aggregated
        
        # Apply proportional reduction
        scaling_factor = self.policy.daily_budget / total_amount
        
        limited = {}
        for recipient, rewards in aggregated.items():
            limited_rewards = []
            for reward in rewards:
                limited_reward = RewardDistribution(
                    distribution_id=reward.distribution_id,
                    distribution_type=reward.distribution_type,
                    recipient_address=reward.recipient_address,
                    amount=reward.amount * scaling_factor,
                    reason=reward.reason + f" (scaled {scaling_factor:.2%} due to budget)",
                    metrics=reward.metrics,
                    distributed_at=reward.distributed_at
                )
                limited_rewards.append(limited_reward)
            limited[recipient] = limited_rewards
        
        logger.info(f"Applied budget scaling: {scaling_factor:.2%}")
        return limited
    
    async def _distribute_reward(
        self,
        recipient: str,
        rewards: List[RewardDistribution],
        total_amount: Decimal
    ) -> bool:
        """Distribute reward to recipient."""
        try:
            # Import ledger
            from backend.accounting.postgres_ledger import get_postgres_ledger
            ledger = get_postgres_ledger()
            
            # Ensure connection
            if not ledger.pool:
                await ledger.connect()
            
            # Create aggregated description
            descriptions = [r.reason for r in rewards[:3]]  # First 3 reasons
            if len(rewards) > 3:
                descriptions.append(f"and {len(rewards) - 3} more")
            
            description = "; ".join(descriptions)
            
            # Add reward to ledger
            tx = await ledger.add_reward(
                user_address=recipient,
                amount=total_amount,
                description=f"Automatic distribution: {description}",
                metadata={
                    "distribution_ids": [r.distribution_id for r in rewards],
                    "distribution_types": list(set(r.distribution_type for r in rewards)),
                    "reward_count": len(rewards)
                }
            )
            
            # Update reward status
            for reward in rewards:
                reward.status = "distributed"
                reward.transaction_id = tx.id
            
            # Add to history
            self.distribution_history.extend(rewards)
            self._save_history()
            
            logger.info(f"Distributed {total_amount} BLY to {recipient} ({len(rewards)} rewards)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to distribute reward to {recipient}: {e}")
            
            # Mark as failed
            for reward in rewards:
                reward.status = "failed"
            
            # Add back to pending
            if recipient not in self.pending_rewards:
                self.pending_rewards[recipient] = []
            self.pending_rewards[recipient].extend(rewards)
            
            return False
    
    async def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution statistics."""
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        daily_distributions = [
            d for d in self.distribution_history
            if d.distributed_at > day_ago
        ]
        
        weekly_distributions = [
            d for d in self.distribution_history
            if d.distributed_at > week_ago
        ]
        
        # Calculate stats
        daily_total = sum(d.amount for d in daily_distributions)
        weekly_total = sum(d.amount for d in weekly_distributions)
        
        # Group by type
        daily_by_type = {}
        for d in daily_distributions:
            if d.distribution_type not in daily_by_type:
                daily_by_type[d.distribution_type] = {
                    "count": 0,
                    "total": Decimal(0)
                }
            daily_by_type[d.distribution_type]["count"] += 1
            daily_by_type[d.distribution_type]["total"] += d.amount
        
        # Top recipients
        recipient_totals = {}
        for d in weekly_distributions:
            if d.recipient_address not in recipient_totals:
                recipient_totals[d.recipient_address] = Decimal(0)
            recipient_totals[d.recipient_address] += d.amount
        
        top_recipients = sorted(
            recipient_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "daily": {
                "total_distributed": str(daily_total),
                "distribution_count": len(daily_distributions),
                "by_type": {
                    k: {"count": v["count"], "total": str(v["total"])}
                    for k, v in daily_by_type.items()
                }
            },
            "weekly": {
                "total_distributed": str(weekly_total),
                "distribution_count": len(weekly_distributions),
                "daily_average": str(weekly_total / 7)
            },
            "top_recipients": [
                {"address": addr, "total": str(total)}
                for addr, total in top_recipients
            ],
            "pending_rewards": {
                "recipient_count": len(self.pending_rewards),
                "total_pending": str(sum(
                    sum(r.amount for r in rewards)
                    for rewards in self.pending_rewards.values()
                ))
            },
            "policy": {
                "daily_budget": str(self.policy.daily_budget),
                "min_payout": str(self.policy.min_payout_threshold)
            }
        }

# Singleton instance
_distributor = None

def get_reward_distributor() -> AutomaticRewardDistributor:
    """Get or create reward distributor singleton."""
    global _distributor
    if _distributor is None:
        _distributor = AutomaticRewardDistributor()
    return _distributor