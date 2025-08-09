#!/usr/bin/env python3
"""
Rewards and Dataset API Endpoints
Handles PoDL scoring, reward distribution, and contribution tracking
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Create routers
dataset_router = APIRouter(prefix="/datasets", tags=["datasets"])
rewards_router = APIRouter(prefix="/rewards", tags=["rewards"])
contributor_router = APIRouter(prefix="/contributors", tags=["contributors"])

@dataset_router.post("/submit")
async def submit_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    contributor_address: str = Form(...),
    metadata: str = Form("{}")
):
    """
    Submit a dataset for PoDL scoring and rewards.
    """
    try:
        # Parse metadata
        metadata_dict = json.loads(metadata)
        
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Import PoDL recorder
        from backend.data.podl_score_system import get_podl_recorder
        recorder = get_podl_recorder()
        
        # Submit dataset for scoring
        contribution = await recorder.submit_dataset(
            contributor_address=contributor_address,
            dataset_path=str(temp_path),
            metadata=metadata_dict
        )
        
        # Clean up temp file
        temp_path.unlink()
        
        # If validated, queue reward distribution
        if contribution.validation_status == "validated":
            from backend.rewards.automatic_distribution import get_reward_distributor
            distributor = get_reward_distributor()
            
            # Queue for next distribution cycle
            background_tasks.add_task(
                distributor._run_distribution_cycle
            )
        
        return {
            "dataset_id": contribution.dataset_id,
            "status": contribution.validation_status,
            "podl_score": contribution.podl_score,
            "quality_score": contribution.quality_score,
            "diversity_score": contribution.diversity_score,
            "toxicity_score": contribution.toxicity_score,
            "usefulness_score": contribution.usefulness_score,
            "reward_amount": str(contribution.reward_amount) if contribution.reward_amount else "0"
        }
        
    except Exception as e:
        logger.error(f"Dataset submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dataset_router.get("/stats")
async def get_dataset_stats():
    """Get global dataset statistics."""
    try:
        from backend.data.podl_score_system import get_podl_recorder
        recorder = get_podl_recorder()
        
        total_datasets = len(recorder.contributions)
        validated_datasets = sum(
            1 for c in recorder.contributions.values()
            if c.validation_status == "validated"
        )
        
        podl_scores = [
            c.podl_score for c in recorder.contributions.values()
            if c.podl_score is not None
        ]
        
        avg_podl_score = sum(podl_scores) / len(podl_scores) if podl_scores else 0
        
        return {
            "total_datasets": total_datasets,
            "validated_datasets": validated_datasets,
            "average_podl_score": avg_podl_score,
            "total_samples": sum(c.sample_count for c in recorder.contributions.values()),
            "total_size_gb": sum(c.dataset_size for c in recorder.contributions.values()) / (1024**3)
        }
        
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dataset_router.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get top dataset contributors."""
    try:
        from backend.data.podl_score_system import get_podl_recorder
        recorder = get_podl_recorder()
        
        leaderboard = await recorder.get_leaderboard(limit)
        return leaderboard
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rewards_router.get("/stats")
async def get_reward_stats():
    """Get reward distribution statistics."""
    try:
        from backend.rewards.automatic_distribution import get_reward_distributor
        distributor = get_reward_distributor()
        
        stats = await distributor.get_distribution_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get reward stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rewards_router.get("/recent")
async def get_recent_rewards(limit: int = 20):
    """Get recent reward distributions."""
    try:
        from backend.rewards.automatic_distribution import get_reward_distributor
        distributor = get_reward_distributor()
        
        # Get recent distributions
        recent = distributor.distribution_history[-limit:]
        recent.reverse()  # Most recent first
        
        return [
            {
                "distribution_id": r.distribution_id,
                "distribution_type": r.distribution_type,
                "recipient": r.recipient_address,
                "amount": str(r.amount),
                "reason": r.reason,
                "distributed_at": r.distributed_at.isoformat(),
                "status": r.status
            }
            for r in recent
        ]
        
    except Exception as e:
        logger.error(f"Failed to get recent rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rewards_router.get("/history")
async def get_reward_history(days: int = 7):
    """Get reward distribution history for charts."""
    try:
        from backend.rewards.automatic_distribution import get_reward_distributor
        distributor = get_reward_distributor()
        
        # Calculate daily totals
        now = datetime.utcnow()
        daily_data = {}
        
        for i in range(days):
            date = (now - timedelta(days=i)).date()
            daily_data[str(date)] = {
                "dataset": Decimal(0),
                "inference": Decimal(0),
                "learning": Decimal(0),
                "validation": Decimal(0)
            }
        
        # Aggregate distributions
        cutoff = now - timedelta(days=days)
        for dist in distributor.distribution_history:
            if dist.distributed_at > cutoff:
                date_str = str(dist.distributed_at.date())
                if date_str in daily_data:
                    daily_data[date_str][dist.distribution_type] += dist.amount
        
        # Format for chart
        dates = sorted(daily_data.keys())
        return {
            "dates": dates,
            "dataset_rewards": [float(daily_data[d]["dataset"]) for d in dates],
            "inference_rewards": [float(daily_data[d]["inference"]) for d in dates],
            "learning_rewards": [float(daily_data[d]["learning"]) for d in dates],
            "validation_rewards": [float(daily_data[d]["validation"]) for d in dates]
        }
        
    except Exception as e:
        logger.error(f"Failed to get reward history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@rewards_router.post("/start_distribution")
async def start_distribution(background_tasks: BackgroundTasks):
    """Manually trigger reward distribution."""
    try:
        from backend.rewards.automatic_distribution import get_reward_distributor
        distributor = get_reward_distributor()
        
        # Start distribution in background
        background_tasks.add_task(distributor._run_distribution_cycle)
        
        return {"message": "Distribution cycle started"}
        
    except Exception as e:
        logger.error(f"Failed to start distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@contributor_router.get("/{address}/stats")
async def get_contributor_stats(address: str):
    """Get statistics for a specific contributor."""
    try:
        from backend.data.podl_score_system import get_podl_recorder
        recorder = get_podl_recorder()
        
        stats = await recorder.get_contributor_stats(address)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get contributor stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@contributor_router.get("/{address}/contributions")
async def get_contributor_contributions(address: str, limit: int = 50):
    """Get contributions for a specific contributor."""
    try:
        from backend.data.podl_score_system import get_podl_recorder
        recorder = get_podl_recorder()
        
        # Get contributions
        contributions = [
            c for c in recorder.contributions.values()
            if c.contributor_address == address
        ]
        
        # Sort by submission time
        contributions.sort(key=lambda x: x.submitted_at, reverse=True)
        
        return [
            {
                "dataset_id": c.dataset_id,
                "submitted_at": c.submitted_at.isoformat(),
                "validation_status": c.validation_status,
                "podl_score": c.podl_score,
                "quality_score": c.quality_score,
                "diversity_score": c.diversity_score,
                "sample_count": c.sample_count,
                "reward_amount": str(c.reward_amount) if c.reward_amount else "0"
            }
            for c in contributions[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to get contributions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@contributor_router.get("/{address}/rewards")
async def get_contributor_rewards(address: str, days: int = 30):
    """Get reward history for a specific contributor."""
    try:
        from backend.rewards.automatic_distribution import get_reward_distributor
        distributor = get_reward_distributor()
        
        # Filter rewards for this contributor
        cutoff = datetime.utcnow() - timedelta(days=days)
        rewards = [
            r for r in distributor.distribution_history
            if r.recipient_address == address and r.distributed_at > cutoff
        ]
        
        # Sort by date
        rewards.sort(key=lambda x: x.distributed_at, reverse=True)
        
        return [
            {
                "distribution_id": r.distribution_id,
                "distribution_type": r.distribution_type,
                "amount": str(r.amount),
                "reason": r.reason,
                "distributed_at": r.distributed_at.isoformat(),
                "status": r.status
            }
            for r in rewards
        ]
        
    except Exception as e:
        logger.error(f"Failed to get contributor rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize automatic distribution on module load
async def init_distribution():
    """Initialize automatic reward distribution."""
    try:
        from backend.rewards.automatic_distribution import get_reward_distributor
        distributor = get_reward_distributor()
        await distributor.start()
        logger.info("Automatic reward distribution initialized")
    except Exception as e:
        logger.error(f"Failed to initialize distribution: {e}")

# Export routers
__all__ = ['dataset_router', 'rewards_router', 'contributor_router', 'init_distribution']