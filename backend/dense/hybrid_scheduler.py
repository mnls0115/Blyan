"""
Hybrid Training Scheduler with Dynamic Thresholds
==================================================
Intelligently schedules dense vs LoRA training based on GPU capabilities,
model requirements, and network conditions. Model-agnostic design.
"""

import asyncio
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Literal
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

from backend.learning.dense_partition_planner import (
    DensePartitionPlanner,
    DeviceProfile,
    TrainingMode,
    PartitionPlan
)

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    """Scheduling policies for training assignment."""
    CAPABILITY_FIRST = "capability_first"  # Prioritize by GPU capability
    EFFICIENCY_FIRST = "efficiency_first"  # Prioritize by efficiency metrics
    FAIRNESS = "fairness"  # Round-robin with capability constraints
    ADAPTIVE = "adaptive"  # Dynamic based on network state
    COST_OPTIMIZED = "cost_optimized"  # Minimize compute cost


@dataclass
class WorkerMetrics:
    """Performance metrics for a worker."""
    worker_id: str
    device_profile: DeviceProfile
    
    # Performance history
    avg_throughput: float = 0.0  # tokens/sec
    avg_loss_reduction: float = 0.0  # Average loss improvement
    success_rate: float = 1.0  # Successful rounds ratio
    
    # Current state
    current_mode: Optional[TrainingMode] = None
    assigned_layers: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.utcnow)
    
    # Resource usage
    memory_utilization: float = 0.0  # 0-1
    compute_utilization: float = 0.0  # 0-1
    network_bandwidth: float = 0.0  # MB/s
    
    # Quality metrics
    validation_scores: List[float] = field(default_factory=list)
    drift_warnings: int = 0
    
    def update_metrics(self, throughput: float, loss_reduction: float, success: bool):
        """Update worker metrics with exponential moving average."""
        alpha = 0.3  # EMA factor
        
        self.avg_throughput = alpha * throughput + (1 - alpha) * self.avg_throughput
        self.avg_loss_reduction = alpha * loss_reduction + (1 - alpha) * self.avg_loss_reduction
        self.success_rate = alpha * float(success) + (1 - alpha) * self.success_rate
        self.last_active = datetime.utcnow()
    
    def get_efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        # Weighted combination of metrics
        throughput_norm = min(self.avg_throughput / 1000, 1.0)  # Normalize to 1000 tok/s
        
        score = (
            0.3 * throughput_norm +
            0.3 * self.avg_loss_reduction +
            0.2 * self.success_rate +
            0.1 * (1 - self.memory_utilization) +
            0.1 * (1 - self.compute_utilization)
        )
        
        # Penalty for drift
        score *= (0.95 ** self.drift_warnings)
        
        return score


@dataclass
class NetworkState:
    """Current state of the distributed network."""
    total_workers: int = 0
    active_workers: int = 0
    
    # Aggregate capabilities
    total_vram_gb: float = 0.0
    total_flops: float = 0.0
    
    # Mode distribution
    dense_workers: int = 0
    lora_workers: int = 0
    qlora_workers: int = 0
    idle_workers: int = 0
    
    # Network health
    avg_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    
    # Training progress
    current_epoch: int = 0
    global_loss: float = float('inf')
    improvement_rate: float = 0.0
    
    def update_from_workers(self, workers: List[WorkerMetrics]):
        """Update state from worker metrics."""
        self.total_workers = len(workers)
        self.active_workers = sum(
            1 for w in workers 
            if (datetime.utcnow() - w.last_active).seconds < 60
        )
        
        self.total_vram_gb = sum(w.device_profile.vram_gb for w in workers)
        self.total_flops = sum(w.device_profile.flops_tflops for w in workers)
        
        # Count by mode
        self.dense_workers = sum(1 for w in workers if w.current_mode == TrainingMode.DENSE)
        self.lora_workers = sum(1 for w in workers if w.current_mode == TrainingMode.LORA)
        self.qlora_workers = sum(1 for w in workers if w.current_mode == TrainingMode.QLORA)
        self.idle_workers = self.total_workers - (self.dense_workers + self.lora_workers + self.qlora_workers)


@dataclass
class DynamicThresholds:
    """Dynamic thresholds that adapt based on network conditions."""
    # Mode transition thresholds
    dense_min_vram_gb: float = 8.0
    lora_min_vram_gb: float = 4.0
    qlora_min_vram_gb: float = 2.0
    
    # Performance thresholds
    min_throughput: float = 10.0  # tokens/sec
    min_loss_reduction: float = 0.001
    max_drift: float = 0.1
    
    # Resource thresholds
    max_memory_util: float = 0.9
    max_compute_util: float = 0.95
    
    # Adaptation rates
    vram_adapt_rate: float = 0.1
    perf_adapt_rate: float = 0.05
    
    def adapt_to_conditions(self, network_state: NetworkState, worker_metrics: List[WorkerMetrics]):
        """Adapt thresholds based on network conditions."""
        # If many workers are OOM, increase VRAM requirements
        oom_rate = sum(1 for w in worker_metrics if w.memory_utilization > 0.95) / len(worker_metrics)
        if oom_rate > 0.2:
            self.dense_min_vram_gb *= (1 + self.vram_adapt_rate)
            self.lora_min_vram_gb *= (1 + self.vram_adapt_rate)
        
        # If network is congested, prefer fewer updates
        if network_state.avg_latency_ms > 100:
            self.min_throughput *= (1 - self.perf_adapt_rate)
        
        # If quality is suffering, be more strict
        if network_state.improvement_rate < 0.01:
            self.min_loss_reduction *= (1 + self.perf_adapt_rate)
            self.max_drift *= (1 - self.perf_adapt_rate)
    
    def get_mode_for_device(self, device: DeviceProfile) -> TrainingMode:
        """Determine training mode based on device and thresholds."""
        if device.vram_gb >= self.dense_min_vram_gb:
            return TrainingMode.DENSE
        elif device.vram_gb >= self.lora_min_vram_gb:
            return TrainingMode.LORA
        elif device.vram_gb >= self.qlora_min_vram_gb:
            return TrainingMode.QLORA
        else:
            return TrainingMode.SPECULATIVE


class HybridScheduler:
    """Hybrid scheduler for dense and LoRA training with dynamic thresholds."""
    
    def __init__(
        self,
        model_name: str,
        num_layers: int,
        policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE
    ):
        self.model_name = model_name
        self.num_layers = num_layers
        self.policy = policy
        
        # Components
        self.planner = DensePartitionPlanner()
        self.thresholds = DynamicThresholds()
        
        # State
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.network_state = NetworkState()
        self.assignment_history: List[Dict[str, Any]] = []
        
        # Scheduling queue
        self.pending_rounds: asyncio.Queue = asyncio.Queue()
        self.active_assignments: Dict[str, Dict] = {}
        
        # Monitoring
        self.last_rebalance = datetime.utcnow()
        self.rebalance_interval = timedelta(minutes=5)
    
    async def register_worker(self, worker_id: str, device_profile: DeviceProfile) -> Dict[str, Any]:
        """Register a new worker with the scheduler."""
        # Create or update metrics
        if worker_id not in self.worker_metrics:
            self.worker_metrics[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                device_profile=device_profile
            )
        else:
            self.worker_metrics[worker_id].device_profile = device_profile
        
        # Determine initial mode
        mode = self.thresholds.get_mode_for_device(device_profile)
        self.worker_metrics[worker_id].current_mode = mode
        
        # Update network state
        self._update_network_state()
        
        # Trigger rebalancing if needed
        if self._should_rebalance():
            await self._rebalance_assignments()
        
        return {
            "worker_id": worker_id,
            "assigned_mode": mode.value,
            "status": "registered"
        }
    
    async def schedule_round(
        self,
        round_id: str,
        dataset_size: int,
        target_batch_size: int = 32
    ) -> Dict[str, Any]:
        """Schedule a training round across available workers."""
        # Get active workers
        active_workers = self._get_active_workers()
        if not active_workers:
            return {"status": "error", "message": "No active workers"}
        
        # Create assignments based on policy
        assignments = await self._create_assignments(
            active_workers,
            dataset_size,
            target_batch_size
        )
        
        # Store assignments
        self.active_assignments[round_id] = assignments
        
        # Record in history
        self.assignment_history.append({
            "round_id": round_id,
            "timestamp": datetime.utcnow(),
            "assignments": assignments,
            "policy": self.policy.value
        })
        
        return {
            "round_id": round_id,
            "assignments": assignments,
            "total_workers": len(assignments),
            "dense_workers": sum(1 for a in assignments.values() if a["mode"] == "dense"),
            "lora_workers": sum(1 for a in assignments.values() if a["mode"] == "lora")
        }
    
    async def _create_assignments(
        self,
        workers: List[WorkerMetrics],
        dataset_size: int,
        target_batch_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """Create training assignments based on scheduling policy."""
        assignments = {}
        
        if self.policy == SchedulingPolicy.CAPABILITY_FIRST:
            # Sort by capability
            workers = sorted(workers, key=lambda w: w.device_profile.vram_gb, reverse=True)
            
        elif self.policy == SchedulingPolicy.EFFICIENCY_FIRST:
            # Sort by efficiency score
            workers = sorted(workers, key=lambda w: w.get_efficiency_score(), reverse=True)
            
        elif self.policy == SchedulingPolicy.FAIRNESS:
            # Round-robin (already in order)
            pass
            
        elif self.policy == SchedulingPolicy.ADAPTIVE:
            # Adaptive assignment based on current conditions
            workers = await self._adaptive_sort(workers)
            
        elif self.policy == SchedulingPolicy.COST_OPTIMIZED:
            # Sort by cost efficiency
            workers = sorted(
                workers,
                key=lambda w: w.get_efficiency_score() / max(w.device_profile.cost_per_hour, 0.01),
                reverse=True
            )
        
        # Create partition plan
        devices = [w.device_profile for w in workers]
        partition_plan = self.planner.plan_partition(
            devices=devices,
            model_name=self.model_name,
            num_layers=self.num_layers,
            optimize_for="balanced"
        )
        
        # Assign based on plan
        for i, (worker, assignment) in enumerate(zip(workers, partition_plan.assignments)):
            worker_id = worker.worker_id
            
            # Determine data shard
            shard_size = dataset_size // len(workers)
            shard_start = i * shard_size
            shard_end = shard_start + shard_size if i < len(workers) - 1 else dataset_size
            
            assignments[worker_id] = {
                "mode": assignment.mode.value,
                "layers": assignment.layer_indices,
                "precision": assignment.precision,
                "memory_budget_gb": assignment.memory_budget_gb,
                "dataset_shard": (shard_start, shard_end),
                "batch_size": min(
                    target_batch_size,
                    int(assignment.memory_budget_gb * 100)  # Rough estimate
                ),
                "gradient_accumulation": max(
                    1,
                    target_batch_size // min(target_batch_size, int(assignment.memory_budget_gb * 100))
                )
            }
            
            # Update worker state
            worker.current_mode = assignment.mode
            worker.assigned_layers = [f"layer_{i}" for i in assignment.layer_indices]
        
        return assignments
    
    async def _adaptive_sort(self, workers: List[WorkerMetrics]) -> List[WorkerMetrics]:
        """Adaptively sort workers based on current conditions."""
        scores = []
        
        for worker in workers:
            score = 0.0
            
            # Base capability score
            score += worker.device_profile.vram_gb / 100  # Normalize
            score += worker.device_profile.flops_tflops / 10
            
            # Performance history
            score += worker.get_efficiency_score()
            
            # Current utilization (prefer less loaded)
            score += (1 - worker.memory_utilization)
            score += (1 - worker.compute_utilization)
            
            # Network conditions
            if self.network_state.avg_latency_ms > 50:
                # Prefer workers with better bandwidth
                score += worker.network_bandwidth / 1000
            
            # Mode balance (prefer diversity)
            if worker.current_mode == TrainingMode.DENSE and self.network_state.dense_workers > self.network_state.lora_workers:
                score *= 0.9  # Slight penalty for over-represented mode
            
            scores.append((worker, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scores]
    
    async def update_worker_performance(
        self,
        worker_id: str,
        throughput: float,
        loss_reduction: float,
        success: bool,
        memory_util: float = 0.0,
        compute_util: float = 0.0
    ):
        """Update worker performance metrics."""
        if worker_id not in self.worker_metrics:
            return
        
        worker = self.worker_metrics[worker_id]
        worker.update_metrics(throughput, loss_reduction, success)
        worker.memory_utilization = memory_util
        worker.compute_utilization = compute_util
        
        # Check for drift
        if loss_reduction < self.thresholds.min_loss_reduction:
            worker.drift_warnings += 1
        else:
            worker.drift_warnings = max(0, worker.drift_warnings - 1)
        
        # Update network state
        self._update_network_state()
        
        # Adapt thresholds
        if len(self.worker_metrics) > 5:  # Need enough data
            self.thresholds.adapt_to_conditions(
                self.network_state,
                list(self.worker_metrics.values())
            )
    
    async def _rebalance_assignments(self):
        """Rebalance assignments across workers."""
        logger.info("Rebalancing worker assignments")
        
        # Get current active assignments
        if not self.active_assignments:
            return
        
        # Re-evaluate all workers
        active_workers = self._get_active_workers()
        
        # Check if any workers switched capability tier
        for worker in active_workers:
            new_mode = self.thresholds.get_mode_for_device(worker.device_profile)
            if new_mode != worker.current_mode:
                logger.info(f"Worker {worker.worker_id} transitioning from {worker.current_mode} to {new_mode}")
                worker.current_mode = new_mode
        
        # Update last rebalance time
        self.last_rebalance = datetime.utcnow()
    
    def _get_active_workers(self) -> List[WorkerMetrics]:
        """Get list of active workers."""
        cutoff = datetime.utcnow() - timedelta(seconds=60)
        return [
            w for w in self.worker_metrics.values()
            if w.last_active > cutoff
        ]
    
    def _update_network_state(self):
        """Update network state from worker metrics."""
        self.network_state.update_from_workers(list(self.worker_metrics.values()))
    
    def _should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        # Time-based rebalancing
        if datetime.utcnow() - self.last_rebalance > self.rebalance_interval:
            return True
        
        # Significant change in worker count
        active_count = len(self._get_active_workers())
        if abs(active_count - self.network_state.active_workers) > 2:
            return True
        
        # Performance degradation
        failing_workers = sum(
            1 for w in self.worker_metrics.values()
            if w.success_rate < 0.8 or w.drift_warnings > 3
        )
        if failing_workers > len(self.worker_metrics) * 0.2:
            return True
        
        return False
    
    async def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get current scheduling statistics."""
        active_workers = self._get_active_workers()
        
        return {
            "policy": self.policy.value,
            "total_workers": len(self.worker_metrics),
            "active_workers": len(active_workers),
            "network_state": {
                "total_vram_gb": self.network_state.total_vram_gb,
                "total_flops": self.network_state.total_flops,
                "dense_workers": self.network_state.dense_workers,
                "lora_workers": self.network_state.lora_workers,
                "avg_latency_ms": self.network_state.avg_latency_ms
            },
            "thresholds": {
                "dense_min_vram": self.thresholds.dense_min_vram_gb,
                "lora_min_vram": self.thresholds.lora_min_vram_gb,
                "min_throughput": self.thresholds.min_throughput,
                "min_loss_reduction": self.thresholds.min_loss_reduction
            },
            "worker_efficiency": {
                worker_id: {
                    "mode": w.current_mode.value if w.current_mode else "idle",
                    "efficiency_score": w.get_efficiency_score(),
                    "throughput": w.avg_throughput,
                    "success_rate": w.success_rate
                }
                for worker_id, w in self.worker_metrics.items()
            },
            "active_rounds": len(self.active_assignments),
            "completed_rounds": len(self.assignment_history)
        }
    
    async def handle_worker_failure(self, worker_id: str, round_id: str):
        """Handle worker failure during training."""
        if worker_id not in self.worker_metrics:
            return
        
        # Mark worker as failed
        worker = self.worker_metrics[worker_id]
        worker.success_rate *= 0.5  # Heavy penalty
        worker.drift_warnings += 2
        
        # Find replacement if critical
        if round_id in self.active_assignments:
            failed_assignment = self.active_assignments[round_id].get(worker_id)
            if failed_assignment and failed_assignment["mode"] == "dense":
                # Critical path - need replacement
                replacement = await self._find_replacement_worker(failed_assignment)
                if replacement:
                    logger.info(f"Replacing failed worker {worker_id} with {replacement}")
                    self.active_assignments[round_id][replacement] = failed_assignment
                    del self.active_assignments[round_id][worker_id]
    
    async def _find_replacement_worker(self, assignment: Dict) -> Optional[str]:
        """Find replacement worker for failed assignment."""
        required_memory = assignment["memory_budget_gb"]
        required_mode = TrainingMode(assignment["mode"])
        
        # Find idle workers with sufficient capacity
        candidates = [
            w for w in self._get_active_workers()
            if w.current_mode is None or w.current_mode == TrainingMode.SPECULATIVE
            if w.device_profile.vram_gb >= required_memory
        ]
        
        if not candidates:
            return None
        
        # Select best candidate
        best = max(candidates, key=lambda w: w.get_efficiency_score())
        best.current_mode = required_mode
        return best.worker_id