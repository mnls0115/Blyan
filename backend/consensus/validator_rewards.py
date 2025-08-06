"""
Validator Rewards System

Merit-based compensation for validators without requiring massive stakes.
Integrates with BLY token economics.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

from backend.core.reward_engine import RewardEngine


@dataclass
class ValidatorStats:
    """Statistics for a validator during an epoch"""
    validator_id: str
    epoch: int
    blocks_signed: int = 0
    blocks_proposed: int = 0
    invalid_signatures: int = 0
    missed_blocks: int = 0
    uptime_seconds: float = 0
    total_epoch_seconds: float = 3600  # 1 hour epochs
    avg_signing_latency: timedelta = field(default_factory=lambda: timedelta(0))
    slashing_penalties: Decimal = Decimal("0")
    
    @property
    def uptime_percent(self) -> float:
        """Calculate uptime percentage"""
        if self.total_epoch_seconds == 0:
            return 0
        return (self.uptime_seconds / self.total_epoch_seconds) * 100
    
    @property
    def participation_rate(self) -> float:
        """Calculate participation rate"""
        total_blocks = self.blocks_signed + self.missed_blocks
        if total_blocks == 0:
            return 0
        return self.blocks_signed / total_blocks


@dataclass
class ValidatorInfo:
    """Validator information and configuration"""
    validator_id: str
    public_key: str
    address: str  # BLY reward address
    joined_epoch: int
    reputation_score: float = 1.0
    total_rewards_earned: Decimal = Decimal("0")
    status: str = "active"  # active, inactive, slashed, banned
    ban_until_epoch: Optional[int] = None


@dataclass 
class EpochRewards:
    """Rewards distribution for an epoch"""
    epoch: int
    total_reward_pool: Decimal
    validator_rewards: Dict[str, Decimal]
    timestamp: datetime
    finalized: bool = False


class ValidatorRewards:
    """Manages validator reward calculation and distribution"""
    
    def __init__(self, data_dir: Path, reward_engine: RewardEngine):
        self.data_dir = data_dir
        self.reward_engine = reward_engine
        self.rewards_dir = data_dir / "validator_rewards"
        self.rewards_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.base_reward_per_epoch = Decimal("1000")  # 1000 BLY per epoch
        self.blocks_per_epoch = 3600  # ~1 hour at 1 block/sec
        
        # Validator registry
        self.validators: Dict[str, ValidatorInfo] = {}
        self.load_validators()
        
        # Current epoch tracking
        self.current_epoch = self.calculate_current_epoch()
        self.epoch_stats: Dict[str, ValidatorStats] = {}
        
        # Performance multipliers
        self.uptime_multiplier_max = 2.0  # 99%+ uptime = 2x rewards
        self.speed_bonus_percent = 10  # Fast signing = +10%
        self.correctness_bonus_percent = 20  # No invalid sigs = +20%
        
    def load_validators(self):
        """Load validator registry from disk"""
        validators_file = self.data_dir / "validators.json"
        if validators_file.exists():
            with open(validators_file, 'r') as f:
                data = json.load(f)
                for v in data:
                    validator = ValidatorInfo(
                        validator_id=v["validator_id"],
                        public_key=v["public_key"],
                        address=v["address"],
                        joined_epoch=v["joined_epoch"],
                        reputation_score=v.get("reputation_score", 1.0),
                        total_rewards_earned=Decimal(v.get("total_rewards_earned", "0")),
                        status=v.get("status", "active"),
                        ban_until_epoch=v.get("ban_until_epoch")
                    )
                    self.validators[validator.validator_id] = validator
    
    def save_validators(self):
        """Save validator registry to disk"""
        validators_file = self.data_dir / "validators.json"
        data = []
        for validator in self.validators.values():
            data.append({
                "validator_id": validator.validator_id,
                "public_key": validator.public_key,
                "address": validator.address,
                "joined_epoch": validator.joined_epoch,
                "reputation_score": validator.reputation_score,
                "total_rewards_earned": str(validator.total_rewards_earned),
                "status": validator.status,
                "ban_until_epoch": validator.ban_until_epoch
            })
        
        with open(validators_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_current_epoch(self) -> int:
        """Calculate current epoch number"""
        genesis_time = datetime(2025, 1, 1)  # Network genesis time
        current_time = datetime.utcnow()
        elapsed = current_time - genesis_time
        return int(elapsed.total_seconds() / 3600)  # 1 hour epochs
    
    def register_validator(
        self, 
        validator_id: str, 
        public_key: str, 
        address: str
    ) -> bool:
        """Register a new validator"""
        if validator_id in self.validators:
            return False
        
        validator = ValidatorInfo(
            validator_id=validator_id,
            public_key=public_key,
            address=address,
            joined_epoch=self.current_epoch
        )
        
        self.validators[validator_id] = validator
        self.save_validators()
        
        return True
    
    def record_block_signed(
        self, 
        validator_id: str, 
        block_height: int,
        signing_time: timedelta
    ):
        """Record that a validator signed a block"""
        if validator_id not in self.validators:
            return
        
        stats = self.get_or_create_stats(validator_id)
        stats.blocks_signed += 1
        
        # Update average signing latency
        total_latency = stats.avg_signing_latency * (stats.blocks_signed - 1)
        total_latency += signing_time
        stats.avg_signing_latency = total_latency / stats.blocks_signed
    
    def record_block_proposed(self, validator_id: str, block_height: int):
        """Record that a validator proposed a block"""
        if validator_id not in self.validators:
            return
        
        stats = self.get_or_create_stats(validator_id)
        stats.blocks_proposed += 1
    
    def record_missed_block(self, validator_id: str, block_height: int):
        """Record that a validator missed signing a block"""
        if validator_id not in self.validators:
            return
        
        stats = self.get_or_create_stats(validator_id)
        stats.missed_blocks += 1
    
    def record_invalid_signature(self, validator_id: str):
        """Record that a validator produced an invalid signature"""
        if validator_id not in self.validators:
            return
        
        stats = self.get_or_create_stats(validator_id)
        stats.invalid_signatures += 1
    
    def update_uptime(self, validator_id: str, online: bool):
        """Update validator uptime tracking"""
        if validator_id not in self.validators:
            return
        
        stats = self.get_or_create_stats(validator_id)
        if online:
            # Add 1 second of uptime (called every second)
            stats.uptime_seconds += 1
    
    def apply_slashing_penalty(self, validator_id: str, penalty: Decimal):
        """Apply slashing penalty to validator"""
        if validator_id not in self.validators:
            return
        
        stats = self.get_or_create_stats(validator_id)
        stats.slashing_penalties += penalty
        
        # Update validator status if needed
        validator = self.validators[validator_id]
        if stats.slashing_penalties > Decimal("100"):
            validator.status = "slashed"
    
    def get_or_create_stats(self, validator_id: str) -> ValidatorStats:
        """Get or create stats for current epoch"""
        key = f"{validator_id}_{self.current_epoch}"
        if key not in self.epoch_stats:
            self.epoch_stats[key] = ValidatorStats(
                validator_id=validator_id,
                epoch=self.current_epoch
            )
        return self.epoch_stats[key]
    
    def get_active_validators(self) -> List[str]:
        """Get list of active validators"""
        active = []
        for validator in self.validators.values():
            if validator.status == "active":
                if validator.ban_until_epoch is None or validator.ban_until_epoch <= self.current_epoch:
                    active.append(validator.validator_id)
        return active
    
    async def calculate_epoch_rewards(self, epoch: int) -> Dict[str, Decimal]:
        """Calculate rewards for all validators for an epoch"""
        rewards = {}
        
        # Get active validators for this epoch
        active_validators = self.get_active_validators()
        if not active_validators:
            return rewards
        
        # Calculate total reward pool from inflation
        total_pool = await self.get_epoch_reward_pool(epoch)
        
        # Calculate individual rewards
        for validator_id in active_validators:
            stats = self.get_validator_stats(validator_id, epoch)
            if not stats:
                continue
            
            # Base reward proportional to participation
            participation_rate = stats.blocks_signed / self.blocks_per_epoch
            reward = total_pool * Decimal(str(participation_rate))
            
            # Apply multipliers
            reward = self.apply_performance_multipliers(reward, stats)
            
            # Apply penalties
            reward -= stats.slashing_penalties
            
            # Ensure non-negative
            rewards[validator_id] = max(Decimal("0"), reward)
        
        # Normalize rewards to match total pool
        total_rewards = sum(rewards.values())
        if total_rewards > 0:
            normalization_factor = total_pool / total_rewards
            for validator_id in rewards:
                rewards[validator_id] *= normalization_factor
        
        return rewards
    
    def apply_performance_multipliers(
        self, 
        base_reward: Decimal, 
        stats: ValidatorStats
    ) -> Decimal:
        """Apply performance-based multipliers to base reward"""
        reward = base_reward
        
        # Uptime multiplier (linear from 1.0 to 2.0 based on uptime)
        uptime_multiplier = 1.0 + min(1.0, stats.uptime_percent / 100)
        reward *= Decimal(str(uptime_multiplier))
        
        # Speed bonus (fast signing)
        if stats.avg_signing_latency < timedelta(milliseconds=100):
            reward *= Decimal("1.1")  # +10%
        
        # Correctness bonus (no invalid signatures)
        if stats.invalid_signatures == 0 and stats.blocks_signed > 0:
            reward *= Decimal("1.2")  # +20%
        
        # Reputation multiplier
        validator = self.validators.get(stats.validator_id)
        if validator:
            reward *= Decimal(str(validator.reputation_score))
        
        return reward
    
    async def get_epoch_reward_pool(self, epoch: int) -> Decimal:
        """Get total reward pool for an epoch"""
        # Base reward pool
        base_pool = self.base_reward_per_epoch
        
        # Adjust based on network inflation target
        if self.reward_engine:
            # Use 10% of inflation allocation for validators
            inflation_rate = await self.reward_engine.get_current_inflation_rate()
            total_supply = await self.reward_engine.get_total_supply()
            
            annual_inflation = total_supply * Decimal(str(inflation_rate))
            hourly_inflation = annual_inflation / (365 * 24)
            validator_allocation = hourly_inflation * Decimal("0.1")  # 10% for validators
            
            base_pool = max(base_pool, validator_allocation)
        
        return base_pool
    
    def get_validator_stats(self, validator_id: str, epoch: int) -> Optional[ValidatorStats]:
        """Get validator stats for a specific epoch"""
        key = f"{validator_id}_{epoch}"
        return self.epoch_stats.get(key)
    
    async def finalize_epoch(self, epoch: int) -> EpochRewards:
        """Finalize rewards for an epoch and trigger distribution"""
        # Calculate rewards
        rewards = await self.calculate_epoch_rewards(epoch)
        
        # Create epoch rewards record
        epoch_rewards = EpochRewards(
            epoch=epoch,
            total_reward_pool=sum(rewards.values()),
            validator_rewards=rewards,
            timestamp=datetime.utcnow(),
            finalized=True
        )
        
        # Save epoch rewards
        self.save_epoch_rewards(epoch_rewards)
        
        # Distribute rewards
        await self.distribute_rewards(epoch_rewards)
        
        # Update validator totals
        for validator_id, amount in rewards.items():
            if validator_id in self.validators:
                self.validators[validator_id].total_rewards_earned += amount
        
        self.save_validators()
        
        # Clean up old stats
        self.cleanup_old_stats(epoch)
        
        return epoch_rewards
    
    async def distribute_rewards(self, epoch_rewards: EpochRewards):
        """Distribute rewards to validators"""
        if not self.reward_engine:
            return
        
        distributions = []
        
        for validator_id, amount in epoch_rewards.validator_rewards.items():
            validator = self.validators.get(validator_id)
            if not validator or amount <= 0:
                continue
            
            # Queue reward distribution
            distributions.append({
                "address": validator.address,
                "amount": amount,
                "reason": f"Validator rewards epoch {epoch_rewards.epoch}",
                "metadata": {
                    "validator_id": validator_id,
                    "epoch": epoch_rewards.epoch
                }
            })
        
        # Batch distribute through reward engine
        if distributions:
            await self.reward_engine.batch_distribute_rewards(distributions)
    
    def save_epoch_rewards(self, epoch_rewards: EpochRewards):
        """Save epoch rewards to disk"""
        epoch_file = self.rewards_dir / f"epoch_{epoch_rewards.epoch}.json"
        
        data = {
            "epoch": epoch_rewards.epoch,
            "total_reward_pool": str(epoch_rewards.total_reward_pool),
            "validator_rewards": {
                k: str(v) for k, v in epoch_rewards.validator_rewards.items()
            },
            "timestamp": epoch_rewards.timestamp.isoformat(),
            "finalized": epoch_rewards.finalized
        }
        
        with open(epoch_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_epoch_rewards(self, epoch: int) -> Optional[EpochRewards]:
        """Load epoch rewards from disk"""
        epoch_file = self.rewards_dir / f"epoch_{epoch}.json"
        if not epoch_file.exists():
            return None
        
        with open(epoch_file, 'r') as f:
            data = json.load(f)
        
        return EpochRewards(
            epoch=data["epoch"],
            total_reward_pool=Decimal(data["total_reward_pool"]),
            validator_rewards={
                k: Decimal(v) for k, v in data["validator_rewards"].items()
            },
            timestamp=datetime.fromisoformat(data["timestamp"]),
            finalized=data["finalized"]
        )
    
    def cleanup_old_stats(self, current_epoch: int):
        """Clean up stats older than 7 days"""
        cutoff_epoch = current_epoch - (7 * 24)  # 7 days
        
        keys_to_remove = []
        for key in self.epoch_stats:
            _, epoch_str = key.rsplit('_', 1)
            epoch = int(epoch_str)
            if epoch < cutoff_epoch:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.epoch_stats[key]
    
    def update_validator_reputation(self, validator_id: str, delta: float):
        """Update validator reputation score"""
        if validator_id not in self.validators:
            return
        
        validator = self.validators[validator_id]
        validator.reputation_score = max(0.1, min(2.0, validator.reputation_score + delta))
        self.save_validators()
    
    def ban_validator(self, validator_id: str, ban_duration_epochs: int):
        """Temporarily ban a validator"""
        if validator_id not in self.validators:
            return
        
        validator = self.validators[validator_id]
        validator.status = "banned"
        validator.ban_until_epoch = self.current_epoch + ban_duration_epochs
        self.save_validators()
    
    def get_validator_summary(self, validator_id: str) -> Dict:
        """Get summary of validator performance"""
        validator = self.validators.get(validator_id)
        if not validator:
            return {}
        
        current_stats = self.get_validator_stats(validator_id, self.current_epoch)
        
        return {
            "validator_id": validator_id,
            "address": validator.address,
            "status": validator.status,
            "joined_epoch": validator.joined_epoch,
            "reputation_score": validator.reputation_score,
            "total_rewards_earned": str(validator.total_rewards_earned),
            "current_epoch_stats": {
                "blocks_signed": current_stats.blocks_signed if current_stats else 0,
                "participation_rate": current_stats.participation_rate if current_stats else 0,
                "uptime_percent": current_stats.uptime_percent if current_stats else 0,
                "avg_signing_latency_ms": current_stats.avg_signing_latency.total_seconds() * 1000 if current_stats else 0
            }
        }


class ValidatorRewardScheduler:
    """Handles automatic epoch finalization and reward distribution"""
    
    def __init__(self, validator_rewards: ValidatorRewards):
        self.validator_rewards = validator_rewards
        self.running = False
        self.task = None
    
    async def start(self):
        """Start the reward scheduler"""
        self.running = True
        self.task = asyncio.create_task(self.run())
    
    async def stop(self):
        """Stop the reward scheduler"""
        self.running = False
        if self.task:
            await self.task
    
    async def run(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Check if current epoch has ended
                current_epoch = self.validator_rewards.calculate_current_epoch()
                
                # Check if we need to finalize previous epoch
                previous_epoch = current_epoch - 1
                if previous_epoch >= 0:
                    existing_rewards = self.validator_rewards.load_epoch_rewards(previous_epoch)
                    if not existing_rewards or not existing_rewards.finalized:
                        # Finalize previous epoch
                        await self.validator_rewards.finalize_epoch(previous_epoch)
                        print(f"Finalized rewards for epoch {previous_epoch}")
                
                # Sleep until next check (1 minute)
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"Error in reward scheduler: {e}")
                await asyncio.sleep(60)