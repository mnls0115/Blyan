"""
Consensus and Validator API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from backend.consensus.state_sync import StateSyncProtocol, SyncResult
from backend.consensus.validator_rewards import ValidatorRewards, ValidatorRewardScheduler
from backend.core.reward_engine import RewardEngine

router = APIRouter(prefix="/consensus", tags=["consensus"])

# Global instances (initialized in main server)
state_sync: Optional[StateSyncProtocol] = None
validator_rewards: Optional[ValidatorRewards] = None
reward_scheduler: Optional[ValidatorRewardScheduler] = None


def init_consensus_systems(data_dir: Path, reward_engine: RewardEngine):
    """Initialize consensus systems"""
    global state_sync, validator_rewards, reward_scheduler
    
    state_sync = StateSyncProtocol(data_dir)
    validator_rewards = ValidatorRewards(data_dir, reward_engine)
    reward_scheduler = ValidatorRewardScheduler(validator_rewards)


@router.post("/state_sync/fast_sync")
async def fast_sync_from_checkpoint(checkpoint_hash: str) -> Dict:
    """
    Fast sync blockchain state from a trusted checkpoint.
    
    This allows new nodes to sync in minutes instead of days.
    """
    if not state_sync:
        raise HTTPException(status_code=500, detail="State sync not initialized")
    
    result = await state_sync.fast_sync(checkpoint_hash)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=f"Sync failed: {result.error}")
    
    return {
        "success": result.success,
        "sync_time_seconds": result.sync_time_seconds,
        "blocks_synced": result.blocks_synced,
        "data_downloaded_gb": result.data_downloaded_gb,
        "message": f"Successfully synced {result.blocks_synced} blocks in {result.sync_time_seconds:.1f} seconds"
    }


@router.get("/state_sync/latest_checkpoint")
async def get_latest_checkpoint() -> Dict:
    """Get the latest checkpoint available for fast sync"""
    if not state_sync:
        raise HTTPException(status_code=500, detail="State sync not initialized")
    
    # Get latest checkpoint from snapshot providers
    # For now, return a mock checkpoint
    return {
        "checkpoint_hash": "abc123def456",
        "height": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
        "validator_signatures": 10,
        "state_size_gb": 5.2
    }


@router.get("/state_sync/providers")
async def get_snapshot_providers() -> Dict:
    """Get list of available snapshot providers"""
    if not state_sync:
        raise HTTPException(status_code=500, detail="State sync not initialized")
    
    return {
        "providers": state_sync.snapshot_providers,
        "recommended": state_sync.snapshot_providers[0] if state_sync.snapshot_providers else None
    }


@router.post("/validators/register")
async def register_validator(
    validator_id: str,
    public_key: str,
    reward_address: str
) -> Dict:
    """Register a new validator"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    success = validator_rewards.register_validator(validator_id, public_key, reward_address)
    
    if not success:
        raise HTTPException(status_code=400, detail="Validator already registered")
    
    return {
        "success": True,
        "validator_id": validator_id,
        "message": "Validator registered successfully"
    }


@router.get("/validators")
async def list_validators() -> Dict:
    """List all validators and their status"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    validators = []
    for validator in validator_rewards.validators.values():
        validators.append({
            "validator_id": validator.validator_id,
            "address": validator.address,
            "status": validator.status,
            "joined_epoch": validator.joined_epoch,
            "reputation_score": validator.reputation_score,
            "total_rewards_earned": str(validator.total_rewards_earned)
        })
    
    return {
        "validators": validators,
        "total_validators": len(validators),
        "active_validators": len([v for v in validators if v["status"] == "active"])
    }


@router.get("/validators/{validator_id}")
async def get_validator_details(validator_id: str) -> Dict:
    """Get detailed information about a specific validator"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    summary = validator_rewards.get_validator_summary(validator_id)
    
    if not summary:
        raise HTTPException(status_code=404, detail="Validator not found")
    
    return summary


@router.get("/validators/{validator_id}/rewards")
async def get_validator_rewards(validator_id: str, epochs: int = 10) -> Dict:
    """Get reward history for a validator"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    current_epoch = validator_rewards.calculate_current_epoch()
    reward_history = []
    
    for epoch in range(max(0, current_epoch - epochs), current_epoch):
        epoch_rewards = validator_rewards.load_epoch_rewards(epoch)
        if epoch_rewards and validator_id in epoch_rewards.validator_rewards:
            reward_history.append({
                "epoch": epoch,
                "reward": str(epoch_rewards.validator_rewards[validator_id]),
                "timestamp": epoch_rewards.timestamp.isoformat()
            })
    
    return {
        "validator_id": validator_id,
        "reward_history": reward_history,
        "total_epochs": len(reward_history)
    }


@router.get("/epochs/current")
async def get_current_epoch() -> Dict:
    """Get current epoch information"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    current_epoch = validator_rewards.calculate_current_epoch()
    
    # Calculate time until next epoch
    epoch_duration = 3600  # 1 hour
    epoch_start = current_epoch * epoch_duration
    next_epoch_start = (current_epoch + 1) * epoch_duration
    current_time = int(datetime.utcnow().timestamp())
    time_remaining = next_epoch_start - (current_time - 1704067200)  # From genesis
    
    return {
        "current_epoch": current_epoch,
        "epoch_duration_seconds": epoch_duration,
        "time_remaining_seconds": max(0, time_remaining),
        "next_epoch": current_epoch + 1
    }


@router.get("/epochs/{epoch}/rewards")
async def get_epoch_rewards(epoch: int) -> Dict:
    """Get reward distribution for a specific epoch"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    epoch_rewards = validator_rewards.load_epoch_rewards(epoch)
    
    if not epoch_rewards:
        raise HTTPException(status_code=404, detail="Epoch rewards not found")
    
    return {
        "epoch": epoch_rewards.epoch,
        "total_reward_pool": str(epoch_rewards.total_reward_pool),
        "validator_rewards": {
            k: str(v) for k, v in epoch_rewards.validator_rewards.items()
        },
        "timestamp": epoch_rewards.timestamp.isoformat(),
        "finalized": epoch_rewards.finalized,
        "num_validators_rewarded": len(epoch_rewards.validator_rewards)
    }


@router.post("/epochs/{epoch}/finalize")
async def finalize_epoch(epoch: int) -> Dict:
    """Manually finalize an epoch (admin only)"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    current_epoch = validator_rewards.calculate_current_epoch()
    
    if epoch >= current_epoch:
        raise HTTPException(status_code=400, detail="Cannot finalize current or future epoch")
    
    # Check if already finalized
    existing = validator_rewards.load_epoch_rewards(epoch)
    if existing and existing.finalized:
        raise HTTPException(status_code=400, detail="Epoch already finalized")
    
    # Finalize the epoch
    epoch_rewards = await validator_rewards.finalize_epoch(epoch)
    
    return {
        "success": True,
        "epoch": epoch,
        "total_rewards_distributed": str(epoch_rewards.total_reward_pool),
        "validators_rewarded": len(epoch_rewards.validator_rewards)
    }


@router.get("/rewards/pool")
async def get_reward_pool_info() -> Dict:
    """Get information about validator reward pool"""
    if not validator_rewards:
        raise HTTPException(status_code=500, detail="Validator rewards not initialized")
    
    current_epoch = validator_rewards.calculate_current_epoch()
    pool_size = await validator_rewards.get_epoch_reward_pool(current_epoch)
    
    return {
        "current_epoch": current_epoch,
        "epoch_reward_pool": str(pool_size),
        "base_reward_per_epoch": str(validator_rewards.base_reward_per_epoch),
        "blocks_per_epoch": validator_rewards.blocks_per_epoch
    }


@router.post("/rewards/scheduler/start")
async def start_reward_scheduler() -> Dict:
    """Start automatic reward distribution scheduler"""
    if not reward_scheduler:
        raise HTTPException(status_code=500, detail="Reward scheduler not initialized")
    
    await reward_scheduler.start()
    
    return {
        "success": True,
        "message": "Reward scheduler started"
    }


@router.post("/rewards/scheduler/stop")
async def stop_reward_scheduler() -> Dict:
    """Stop automatic reward distribution scheduler"""
    if not reward_scheduler:
        raise HTTPException(status_code=500, detail="Reward scheduler not initialized")
    
    await reward_scheduler.stop()
    
    return {
        "success": True,
        "message": "Reward scheduler stopped"
    }


@router.get("/health")
async def consensus_health_check() -> Dict:
    """Check health of consensus systems"""
    return {
        "state_sync": "initialized" if state_sync else "not_initialized",
        "validator_rewards": "initialized" if validator_rewards else "not_initialized",
        "reward_scheduler": "running" if reward_scheduler and reward_scheduler.running else "stopped",
        "timestamp": datetime.utcnow().isoformat()
    }