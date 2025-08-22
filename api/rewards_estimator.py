#!/usr/bin/env python3
"""
Reward Estimation API
Provides endpoints for estimating rewards based on work metrics.
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import logging

from backend.rewards.policy import (
    calc_inference_bly,
    calc_learning_bly,
    calc_validation_bly,
    calc_dataset_bly,
    estimate_daily_rewards,
    RewardConfig
)
from backend.rewards.buckets import BucketAllocator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rewards", tags=["rewards"])

# Initialize shared instances
config = RewardConfig.from_yaml()
allocator = BucketAllocator()


class InferenceEstimate(BaseModel):
    """Inference reward estimate."""
    per_1k_tokens_bly: float = Field(..., description="BLY per 1000 tokens")
    estimate_bly: float = Field(..., description="Total estimated BLY")
    quality_applied: float = Field(..., description="Quality multiplier applied")
    budget_available: bool = Field(..., description="Whether budget is available")


class LearningEstimate(BaseModel):
    """Learning reward estimate."""
    per_1pct_improve_bly: float = Field(..., description="BLY per 1% improvement")
    estimate_bly: float = Field(..., description="Total estimated BLY")
    difficulty_applied: float = Field(..., description="Difficulty multiplier")
    applicability_applied: float = Field(..., description="Applicability multiplier")
    budget_available: bool = Field(..., description="Whether budget is available")


class ValidationEstimate(BaseModel):
    """Validation reward estimate."""
    per_task_bly: float = Field(..., description="BLY per validation task")
    estimate_bly: float = Field(..., description="Total estimated BLY")
    budget_available: bool = Field(..., description="Whether budget is available")


class DatasetEstimate(BaseModel):
    """Dataset reward estimate."""
    per_gb_bly: float = Field(..., description="BLY per GB of data")
    estimate_bly: float = Field(..., description="Total estimated BLY")
    quality_applied: float = Field(..., description="Quality score applied")
    diversity_applied: float = Field(..., description="Diversity score applied")
    budget_available: bool = Field(..., description="Whether budget is available")


class RewardEstimateResponse(BaseModel):
    """Complete reward estimate response."""
    inference: Optional[InferenceEstimate] = None
    learning: Optional[LearningEstimate] = None
    validation: Optional[ValidationEstimate] = None
    dataset: Optional[DatasetEstimate] = None
    notes: List[str] = Field(default_factory=list)
    budget_status: Dict = Field(default_factory=dict)


@router.get("/estimate", response_model=RewardEstimateResponse)
async def estimate_rewards(
    # Inference parameters
    tokens: Optional[int] = Query(None, ge=0, description="Number of tokens to process"),
    quality: Optional[float] = Query(1.0, ge=0.5, le=1.5, description="Quality score"),
    
    # Learning parameters
    improvement_pct: Optional[float] = Query(None, ge=0, description="Improvement percentage"),
    difficulty: Optional[float] = Query(1.5, ge=1.0, le=3.0, description="Difficulty factor"),
    applicability: Optional[float] = Query(1.0, ge=0.8, le=1.2, description="Applicability factor"),
    
    # Validation parameters
    validation_tasks: Optional[int] = Query(None, ge=0, description="Number of validation tasks"),
    
    # Dataset parameters
    dataset_gb: Optional[float] = Query(None, ge=0, description="Dataset size in GB"),
    dataset_quality: Optional[float] = Query(0.9, ge=0, le=1, description="Dataset quality score"),
    dataset_diversity: Optional[float] = Query(1.0, ge=0, le=2, description="Dataset diversity score")
):
    """
    Estimate rewards for various types of work.
    
    Provide parameters for the types of work you want to estimate.
    The response includes both policy-based estimates and current budget availability.
    """
    response = RewardEstimateResponse()
    notes = []
    
    # Get current bucket status
    bucket_status = allocator.get_bucket_status()
    response.budget_status = bucket_status
    
    # Estimate inference rewards
    if tokens is not None:
        inference_reward = calc_inference_bly(tokens, quality, config=config)
        
        # Check budget availability
        inference_budget = bucket_status['buckets']['inference']['available']
        budget_available = inference_budget >= inference_reward
        
        response.inference = InferenceEstimate(
            per_1k_tokens_bly=config.per_1k_tokens_bly,
            estimate_bly=inference_reward,
            quality_applied=quality,
            budget_available=budget_available
        )
        
        if not budget_available:
            notes.append(f"Inference budget limited. Available: {inference_budget:.2f} BLY")
    
    # Estimate learning rewards
    if improvement_pct is not None:
        # Use learning bucket allocation
        learning_pool = bucket_status['buckets']['learning']['available'] * 24  # Daily equivalent
        
        learning_reward = calc_learning_bly(
            learning_pool,
            improvement_pct,
            difficulty,
            applicability,
            hourslice_share=1/24,
            config=config
        )
        
        # Check budget availability
        learning_budget = bucket_status['buckets']['learning']['available']
        budget_available = learning_budget >= learning_reward
        
        response.learning = LearningEstimate(
            per_1pct_improve_bly=config.per_1pct_improvement_bly,
            estimate_bly=learning_reward,
            difficulty_applied=difficulty,
            applicability_applied=applicability,
            budget_available=budget_available
        )
        
        if not budget_available:
            notes.append(f"Learning budget limited. Available: {learning_budget:.2f} BLY")
    
    # Estimate validation rewards
    if validation_tasks is not None:
        validation_reward = calc_validation_bly(validation_tasks, config=config)
        
        # Check budget availability
        validation_budget = bucket_status['buckets']['validation']['available']
        budget_available = validation_budget >= validation_reward
        
        response.validation = ValidationEstimate(
            per_task_bly=config.per_validation_task_bly,
            estimate_bly=validation_reward,
            budget_available=budget_available
        )
        
        if not budget_available:
            notes.append(f"Validation budget limited. Available: {validation_budget:.2f} BLY")
    
    # Estimate dataset rewards
    if dataset_gb is not None:
        dataset_reward = calc_dataset_bly(
            dataset_gb,
            dataset_quality,
            dataset_diversity,
            config=config
        )
        
        # Check budget availability
        dataset_budget = bucket_status['buckets']['dataset']['available']
        budget_available = dataset_budget >= dataset_reward
        
        response.dataset = DatasetEstimate(
            per_gb_bly=config.per_gb_dataset_bly,
            estimate_bly=dataset_reward,
            quality_applied=dataset_quality,
            diversity_applied=dataset_diversity,
            budget_available=budget_available
        )
        
        if not budget_available:
            notes.append(f"Dataset budget limited. Available: {dataset_budget:.2f} BLY")
    
    # Add general notes
    notes.append("Estimates include active balancing and current bucket utilization.")
    notes.append("Actual rewards depend on network demand and validation.")
    
    response.notes = notes
    
    return response


@router.get("/policy")
async def get_reward_policy():
    """
    Get current reward policy configuration.
    
    Returns the per-unit rates and multiplier ranges.
    """
    return {
        "rates": {
            "per_1k_tokens_bly": config.per_1k_tokens_bly,
            "per_1pct_improvement_bly": config.per_1pct_improvement_bly,
            "per_validation_task_bly": config.per_validation_task_bly,
            "per_gb_dataset_bly": config.per_gb_dataset_bly
        },
        "multipliers": {
            "inference_quality": {
                "min": config.inference_quality_min,
                "max": config.inference_quality_max
            },
            "learning_difficulty": {
                "min": config.difficulty_min,
                "max": config.difficulty_max
            },
            "learning_applicability": {
                "min": config.applicability_min,
                "max": config.applicability_max
            }
        },
        "daily_budget": config.daily_budget_bly,
        "description": "All rewards are work-based. No GPU-specific multipliers."
    }


@router.get("/buckets/status")
async def get_bucket_status():
    """
    Get current status of budget buckets.
    
    Shows available budget, utilization, and backpay queues.
    """
    return allocator.get_bucket_status()


@router.get("/buckets/metrics")
async def get_bucket_metrics():
    """
    Get metrics for monitoring budget health.
    
    Includes utilization rates, queue sizes, and alerts.
    """
    return allocator.get_metrics()


@router.post("/buckets/allocate")
async def allocate_from_bucket(
    bucket_type: str = Query(..., description="Bucket type (inference/learning/validation/dataset)"),
    amount: float = Query(..., gt=0, description="Amount to allocate in BLY"),
    requester: str = Query("api", description="Requester identifier")
):
    """
    Allocate BLY from a specific bucket.
    
    Returns granted amount and request ID. If budget exhausted, request is queued for backpay.
    """
    try:
        granted, request_id = allocator.allocate(bucket_type, amount, requester)
        
        return {
            "granted_bly": granted,
            "requested_bly": amount,
            "request_id": request_id,
            "queued_bly": max(0, amount - granted),
            "bucket_type": bucket_type
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/calculator")
async def reward_calculator(
    # Common work scenarios
    scenario: str = Query(
        "custom",
        description="Scenario: 'inference_hour', 'learning_session', 'validation_batch', 'dataset_upload', 'custom'"
    ),
    
    # Scenario-specific defaults
    hours: float = Query(1.0, description="Hours of work"),
    tokens_per_hour: int = Query(100000, description="Tokens per hour for inference"),
    improvements_per_session: float = Query(2.0, description="% improvement for learning"),
    tasks_per_batch: int = Query(50, description="Validation tasks per batch"),
    gb_per_upload: float = Query(10.0, description="GB per dataset upload")
):
    """
    Interactive reward calculator for common scenarios.
    
    Helps users understand potential earnings for different types of work.
    """
    estimates = {}
    
    if scenario == "inference_hour" or scenario == "custom":
        total_tokens = int(tokens_per_hour * hours)
        estimates["inference"] = {
            "description": f"Processing {total_tokens:,} tokens over {hours} hours",
            "reward_bly": calc_inference_bly(total_tokens, quality=1.0, config=config),
            "per_hour": calc_inference_bly(tokens_per_hour, quality=1.0, config=config)
        }
    
    if scenario == "learning_session" or scenario == "custom":
        learning_pool = config.daily_budget_bly * 0.35  # Learning allocation
        estimates["learning"] = {
            "description": f"Achieving {improvements_per_session}% improvement",
            "reward_bly": calc_learning_bly(
                learning_pool,
                improvements_per_session,
                difficulty=1.5,
                applicability=1.0,
                hourslice_share=hours/24,
                config=config
            ),
            "factors": {
                "base": improvements_per_session * config.per_1pct_improvement_bly,
                "with_difficulty": improvements_per_session * config.per_1pct_improvement_bly * 1.5
            }
        }
    
    if scenario == "validation_batch" or scenario == "custom":
        total_tasks = int(tasks_per_batch * hours)
        estimates["validation"] = {
            "description": f"Completing {total_tasks} validation tasks",
            "reward_bly": calc_validation_bly(total_tasks, config=config),
            "per_task": config.per_validation_task_bly
        }
    
    if scenario == "dataset_upload" or scenario == "custom":
        estimates["dataset"] = {
            "description": f"Contributing {gb_per_upload} GB of quality data",
            "reward_bly": calc_dataset_bly(gb_per_upload, 0.9, 1.0, config=config),
            "per_gb": config.per_gb_dataset_bly
        }
    
    # Add summary
    total_reward = sum(est.get("reward_bly", 0) for est in estimates.values())
    
    return {
        "scenario": scenario,
        "estimates": estimates,
        "total_reward_bly": round(total_reward, 2),
        "work_duration_hours": hours,
        "hourly_rate_bly": round(total_reward / hours, 2) if hours > 0 else 0,
        "note": "Estimates assume standard quality and current budget availability"
    }


if __name__ == "__main__":
    # For testing
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Reward Estimation API")
    app.include_router(router)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)