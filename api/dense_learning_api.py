"""
Dense Learning API Endpoints
=============================
API integration for dense model distributed training.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import logging
import asyncio

from backend.learning.dense_round_service import DenseRoundService
from backend.learning.dense_partition_planner import DeviceProfile, TrainingMode
from backend.dense.hybrid_scheduler import SchedulingPolicy

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/learning/dense", tags=["dense_learning"])

# Global coordinator instance (initialized by server)
dense_coordinator: Optional[DenseRoundService] = None


# Request/Response Models
class StartRoundRequest(BaseModel):
    """Request to start a dense learning round."""
    model_name: str = Field(default="Qwen3-8B", description="Model to train")
    dataset_id: str = Field(..., description="Dataset identifier")
    hyperparams: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters")
    target_improvement: float = Field(default=0.01, description="Target loss improvement")
    max_workers: Optional[int] = Field(default=None, description="Maximum workers to use")
    scheduling_policy: str = Field(default="adaptive", description="Scheduling policy")


class WorkerRegistrationRequest(BaseModel):
    """Worker registration request."""
    worker_id: str = Field(..., description="Unique worker identifier")
    vram_gb: float = Field(..., description="Available GPU memory in GB")
    flops_tflops: float = Field(default=10.0, description="Compute capability in TFLOPS")
    gpu_model: str = Field(default="Unknown", description="GPU model name")
    capabilities: List[str] = Field(default_factory=list, description="Supported features")


class SubmitDeltaRequest(BaseModel):
    """Delta submission from worker."""
    round_id: str = Field(..., description="Round identifier")
    worker_id: str = Field(..., description="Worker identifier")
    layer_name: str = Field(..., description="Layer that was updated")
    delta_hash: str = Field(..., description="Hash of delta block on chain")
    base_hash: str = Field(..., description="Hash of base parameters")
    metrics: Dict[str, float] = Field(..., description="Training metrics")
    validation_score: Optional[float] = Field(default=None, description="Validation score")


class WorkerPlanRequest(BaseModel):
    """Request for worker assignment plan."""
    worker_id: str = Field(..., description="Worker identifier")
    round_id: Optional[str] = Field(default=None, description="Specific round")


# API Endpoints
@router.post("/start")
async def start_dense_round(
    request: StartRoundRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Start a new dense learning round.
    
    This endpoint initiates distributed dense model training across registered workers.
    The round will automatically partition the model based on worker capabilities.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        # Convert policy string to enum
        policy = SchedulingPolicy[request.scheduling_policy.upper()]
        
        # Start round
        round_info = await dense_coordinator.start_round(
            model_name=request.model_name,
            dataset_id=request.dataset_id,
            hyperparams=request.hyperparams,
            target_improvement=request.target_improvement,
            max_workers=request.max_workers
        )
        
        # Schedule background monitoring
        background_tasks.add_task(
            monitor_round_progress,
            round_info["round_id"]
        )
        
        return {
            "status": "success",
            "round_id": round_info["round_id"],
            "model": request.model_name,
            "dataset": request.dataset_id,
            "expected_workers": round_info.get("expected_workers", 0),
            "partitions": round_info.get("partitions", []),
            "scheduling_policy": request.scheduling_policy,
            "message": f"Dense learning round {round_info['round_id']} started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start dense round: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{round_id}")
async def get_round_status(round_id: str) -> Dict[str, Any]:
    """
    Get status of a dense learning round.
    
    Returns detailed information about round progress, worker participation,
    and delta collection status.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        status = await dense_coordinator.get_round_status(round_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Round {round_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get round status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_all_rounds_status() -> Dict[str, Any]:
    """
    Get status of all dense learning rounds.
    
    Returns summary of active, pending, and completed rounds.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        all_rounds = await dense_coordinator.get_all_rounds_status()
        
        return {
            "active_rounds": all_rounds.get("active", []),
            "pending_rounds": all_rounds.get("pending", []),
            "completed_rounds": all_rounds.get("completed", [])[-10:],  # Last 10
            "total_completed": len(all_rounds.get("completed", [])),
            "statistics": all_rounds.get("statistics", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get all rounds status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/worker/register")
async def register_worker(request: WorkerRegistrationRequest) -> Dict[str, Any]:
    """
    Register a worker for dense learning.
    
    Workers must register their capabilities before participating in rounds.
    The system will automatically assign appropriate training modes and layers.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        # Create device profile
        device_profile = DeviceProfile(
            device_id=request.worker_id,
            device_type="cuda",
            vram_gb=request.vram_gb,
            flops_tflops=request.flops_tflops,
            gpu_model=request.gpu_model,
            capabilities=request.capabilities
        )
        
        # Register worker
        result = await dense_coordinator.register_worker(
            worker_id=request.worker_id,
            device_profile=device_profile
        )
        
        return {
            "status": "success",
            "worker_id": request.worker_id,
            "assigned_mode": result.get("mode", "pending"),
            "capabilities": {
                "vram_gb": request.vram_gb,
                "flops_tflops": request.flops_tflops,
                "gpu_model": request.gpu_model,
                "supported_modes": result.get("supported_modes", [])
            },
            "message": f"Worker {request.worker_id} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delta/submit")
async def submit_delta(request: SubmitDeltaRequest) -> Dict[str, Any]:
    """
    Submit a trained delta from a worker.
    
    Workers submit layer deltas after local training. Deltas must be
    validated through Proof of Learning before being accepted.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        # Submit delta
        result = await dense_coordinator.submit_delta(
            round_id=request.round_id,
            worker_id=request.worker_id,
            layer_name=request.layer_name,
            delta_hash=request.delta_hash,
            base_hash=request.base_hash,
            metrics=request.metrics,
            validation_score=request.validation_score
        )
        
        return {
            "status": "success",
            "round_id": request.round_id,
            "worker_id": request.worker_id,
            "layer": request.layer_name,
            "delta_hash": request.delta_hash,
            "validation_status": result.get("validation_status", "pending"),
            "acceptance": result.get("accepted", False),
            "message": "Delta submitted for validation"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit delta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worker/plan/{worker_id}")
async def get_worker_plan(worker_id: str, round_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get training plan for a specific worker.
    
    Returns the assigned layers, training mode, and hyperparameters
    for the worker in the current or specified round.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        # Get worker assignment
        if round_id:
            plan = await dense_coordinator.get_worker_assignment(worker_id, round_id)
        else:
            # Get latest active round assignment
            plan = await dense_coordinator.get_latest_worker_assignment(worker_id)
        
        if not plan:
            raise HTTPException(status_code=404, detail=f"No assignment found for worker {worker_id}")
        
        return {
            "worker_id": worker_id,
            "round_id": plan.get("round_id"),
            "assignment": {
                "mode": plan.get("mode", "unknown"),
                "layers": plan.get("layers", []),
                "precision": plan.get("precision", "fp16"),
                "batch_size": plan.get("batch_size", 1),
                "gradient_accumulation": plan.get("gradient_accumulation", 1),
                "learning_rate": plan.get("learning_rate", 1e-5),
                "dataset_shard": plan.get("dataset_shard", [0, 0])
            },
            "memory_budget_gb": plan.get("memory_budget_gb", 0),
            "expected_throughput": plan.get("expected_throughput", 0),
            "status": plan.get("status", "pending")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get worker plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/round/{round_id}/finalize")
async def finalize_round(round_id: str) -> Dict[str, Any]:
    """
    Finalize a dense learning round.
    
    Triggers delta aggregation, consensus validation, and blockchain commit.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        result = await dense_coordinator.finalize_round(round_id)
        
        return {
            "status": "success",
            "round_id": round_id,
            "deltas_collected": result.get("deltas_collected", 0),
            "deltas_validated": result.get("deltas_validated", 0),
            "deltas_accepted": result.get("deltas_accepted", 0),
            "consensus_reached": result.get("consensus_reached", False),
            "blockchain_commits": result.get("blockchain_commits", []),
            "total_improvement": result.get("total_improvement", 0.0),
            "message": f"Round {round_id} finalized successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to finalize round: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_learning_metrics() -> Dict[str, Any]:
    """
    Get overall dense learning metrics.
    
    Returns system-wide statistics including throughput, quality improvements,
    and resource utilization.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        metrics = await dense_coordinator.get_system_metrics()
        
        return {
            "total_rounds": metrics.get("total_rounds", 0),
            "active_workers": metrics.get("active_workers", 0),
            "total_workers": metrics.get("total_workers", 0),
            "aggregate_metrics": {
                "total_deltas": metrics.get("total_deltas", 0),
                "accepted_deltas": metrics.get("accepted_deltas", 0),
                "average_improvement": metrics.get("average_improvement", 0.0),
                "total_tokens_processed": metrics.get("total_tokens_processed", 0)
            },
            "resource_utilization": {
                "total_vram_gb": metrics.get("total_vram_gb", 0),
                "used_vram_gb": metrics.get("used_vram_gb", 0),
                "total_flops": metrics.get("total_flops", 0),
                "utilization_rate": metrics.get("utilization_rate", 0.0)
            },
            "quality_metrics": {
                "average_loss_reduction": metrics.get("average_loss_reduction", 0.0),
                "consensus_rate": metrics.get("consensus_rate", 0.0),
                "validation_success_rate": metrics.get("validation_success_rate", 0.0)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/worker/{worker_id}")
async def unregister_worker(worker_id: str) -> Dict[str, Any]:
    """
    Unregister a worker from dense learning.
    
    Removes worker from active pool and triggers rebalancing if needed.
    """
    if not dense_coordinator:
        raise HTTPException(status_code=503, detail="Dense learning coordinator not initialized")
    
    try:
        result = await dense_coordinator.unregister_worker(worker_id)
        
        return {
            "status": "success",
            "worker_id": worker_id,
            "rounds_affected": result.get("rounds_affected", []),
            "rebalancing_triggered": result.get("rebalancing_triggered", False),
            "message": f"Worker {worker_id} unregistered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to unregister worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def monitor_round_progress(round_id: str):
    """Background task to monitor round progress."""
    try:
        # Monitor for up to 1 hour
        timeout = 3600
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if not dense_coordinator:
                break
                
            status = await dense_coordinator.get_round_status(round_id)
            
            if status and status.get("status") in ["completed", "failed", "cancelled"]:
                # Round finished
                logger.info(f"Round {round_id} finished with status: {status.get('status')}")
                break
            
            # Check every 30 seconds
            await asyncio.sleep(30)
            
    except Exception as e:
        logger.error(f"Error monitoring round {round_id}: {e}")


# Initialization function for server
def initialize_dense_learning(model_name: str = "Qwen3-8B", num_layers: int = 36):
    """Initialize dense learning coordinator."""
    global dense_coordinator
    
    try:
        dense_coordinator = DenseRoundService(
            model_name=model_name,
            num_layers=num_layers
        )
        logger.info("Dense learning coordinator initialized successfully")
        return dense_coordinator
    except Exception as e:
        logger.error(f"Failed to initialize dense learning: {e}")
        return None