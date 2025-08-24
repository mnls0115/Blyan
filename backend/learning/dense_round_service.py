"""
Dense Learning Round Service
=============================
Orchestrates distributed dense model training rounds across heterogeneous GPUs.
Handles worker registration, dataset sharding, delta collection, and PoL validation.
Model-agnostic design that scales with any model size.

Core responsibilities:
- Round lifecycle management
- Worker health monitoring
- Dataset distribution
- Delta aggregation
- Cost accounting
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import torch
import numpy as np

from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex
from backend.core.delta_index import DeltaIndex
from backend.learning.dense_partition_planner import (
    DensePartitionPlanner, 
    DeviceProfile,
    TrainingMode,
    create_partition_plan
)

logger = logging.getLogger(__name__)


class RoundStatus(Enum):
    """Status of a training round."""
    PLANNING = "planning"
    RECRUITING = "recruiting"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    VALIDATING = "validating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerRegistration:
    """Registration info for a worker node."""
    worker_id: str
    device_profile: DeviceProfile
    assigned_layers: List[str]
    training_mode: TrainingMode
    last_heartbeat: float
    status: str  # active, failed, disconnected
    current_step: int
    metrics: Dict[str, Any]
    
    def is_healthy(self, timeout: float = 60) -> bool:
        """Check if worker is healthy."""
        return (
            self.status == "active" and 
            time.time() - self.last_heartbeat < timeout
        )


@dataclass
class RoundConfig:
    """Configuration for a training round."""
    round_id: str
    model_name: str
    model_profile_path: str
    dataset_id: str
    dataset_size: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    checkpoint_interval: int
    validation_interval: int
    optimizer_type: str
    mixed_precision: bool
    activation_checkpointing: bool
    min_workers: int
    max_workers: int
    worker_timeout: float
    cost_per_token: float
    max_cost: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class RoundMetrics:
    """Metrics for a training round."""
    round_id: str
    status: RoundStatus
    start_time: float
    end_time: Optional[float]
    total_steps: int
    total_tokens: int
    average_loss: float
    best_validation_score: float
    total_cost: float
    num_workers: int
    num_deltas_submitted: int
    num_deltas_accepted: int
    failures: List[str]
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "status": self.status.value,
            "duration": (self.end_time or time.time()) - self.start_time
        }


class DenseRoundService:
    """
    Service for orchestrating dense model training rounds.
    Coordinates distributed training across heterogeneous GPUs.
    """
    
    def __init__(
        self,
        root_dir: Path,
        chain_a: Optional[Chain] = None,
        chain_b: Optional[Chain] = None,
        enable_pol: bool = True
    ):
        """
        Initialize round service.
        
        Args:
            root_dir: Root directory for data
            chain_a: Meta chain (optional)
            chain_b: Parameter chain (optional)
            enable_pol: Enable Proof of Learning validation
        """
        self.root_dir = Path(root_dir)
        self.chain_a = chain_a or Chain(root_dir, "A")
        self.chain_b = chain_b or Chain(root_dir, "B")
        self.param_index = ParameterIndex(root_dir / "param_index.json")
        self.delta_index = DeltaIndex(root_dir / "delta_index.json")
        self.enable_pol = enable_pol
        
        # Active rounds
        self.active_rounds: Dict[str, RoundConfig] = {}
        self.round_metrics: Dict[str, RoundMetrics] = {}
        
        # Worker management
        self.registered_workers: Dict[str, WorkerRegistration] = {}
        self.worker_assignments: Dict[str, str] = {}  # worker_id -> round_id
        
        # Partition planner
        self.planner = DensePartitionPlanner()
        
        # Delta storage
        self.pending_deltas: Dict[str, List[Dict]] = defaultdict(list)
        self.validated_deltas: Dict[str, List[Dict]] = defaultdict(list)
        
        # Background tasks
        self.health_monitor_task = None
        self.aggregation_task = None
        
    async def start_round(
        self,
        model_name: str,
        dataset_id: str,
        hyperparams: Dict[str, Any],
        min_workers: int = 1,
        max_workers: int = 100,
        max_cost: float = 1000.0
    ) -> str:
        """
        Start a new training round.
        
        Args:
            model_name: Model identifier
            dataset_id: Dataset identifier
            hyperparams: Training hyperparameters
            min_workers: Minimum workers required
            max_workers: Maximum workers allowed
            max_cost: Maximum cost budget
            
        Returns:
            Round ID
        """
        # Generate round ID
        round_id = f"round_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        # Load model profile
        try:
            from config.model_profile import get_model_config
            model_config = get_model_config()
        except ImportError:
            logger.error("Failed to load model profile")
            raise ValueError("Model profile not found")
        
        # Create round configuration
        config = RoundConfig(
            round_id=round_id,
            model_name=model_name,
            model_profile_path=str(self.root_dir / "config" / "model_profile.py"),
            dataset_id=dataset_id,
            dataset_size=hyperparams.get("dataset_size", 1000000),
            batch_size=hyperparams.get("batch_size", 8),
            gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
            learning_rate=hyperparams.get("learning_rate", 1.5e-5),
            max_steps=hyperparams.get("max_steps", 1000),
            checkpoint_interval=hyperparams.get("checkpoint_interval", 100),
            validation_interval=hyperparams.get("validation_interval", 50),
            optimizer_type=hyperparams.get("optimizer", "adamw"),
            mixed_precision=hyperparams.get("mixed_precision", True),
            activation_checkpointing=hyperparams.get("activation_checkpointing", True),
            min_workers=min_workers,
            max_workers=max_workers,
            worker_timeout=hyperparams.get("worker_timeout", 60.0),
            cost_per_token=hyperparams.get("cost_per_token", 0.0001),
            max_cost=max_cost
        )
        
        # Initialize metrics
        metrics = RoundMetrics(
            round_id=round_id,
            status=RoundStatus.PLANNING,
            start_time=time.time(),
            end_time=None,
            total_steps=0,
            total_tokens=0,
            average_loss=0.0,
            best_validation_score=0.0,
            total_cost=0.0,
            num_workers=0,
            num_deltas_submitted=0,
            num_deltas_accepted=0,
            failures=[]
        )
        
        # Store round
        self.active_rounds[round_id] = config
        self.round_metrics[round_id] = metrics
        
        # Start background tasks
        if not self.health_monitor_task:
            self.health_monitor_task = asyncio.create_task(self._monitor_health())
        
        logger.info(f"Started training round {round_id} for {model_name}")
        
        # Wait for workers to join
        metrics.status = RoundStatus.RECRUITING
        asyncio.create_task(self._recruit_workers(round_id))
        
        return round_id
    
    async def register_worker(
        self,
        worker_id: str,
        device_specs: Dict[str, Any],
        capabilities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Register a worker for training.
        
        Args:
            worker_id: Worker identifier
            device_specs: Device specifications
            capabilities: Worker capabilities
            
        Returns:
            Registration response with assignment
        """
        # Create device profile
        device_profile = DeviceProfile(
            device_id=worker_id,
            vram_gb=device_specs.get("vram_gb", 4.0),
            tflops=device_specs.get("tflops", 10.0),
            net_bandwidth_gbps=device_specs.get("bandwidth", 1.0),
            compute_capability=tuple(device_specs.get("compute_capability", [7, 0])),
            is_nvlink=device_specs.get("nvlink", False),
            reliability_score=device_specs.get("reliability", 1.0)
        )
        
        # Find active round needing workers
        round_id = self._find_round_for_worker(device_profile)
        if not round_id:
            return {
                "status": "no_rounds",
                "message": "No active rounds requiring workers"
            }
        
        config = self.active_rounds[round_id]
        
        # Get partition plan
        plan = await self._get_or_create_partition_plan(round_id)
        if not plan:
            return {
                "status": "planning_failed",
                "message": "Failed to create partition plan"
            }
        
        # Find assignment for this worker
        assignment = None
        for assign in plan["assignments"]:
            if assign["device_id"] == worker_id:
                assignment = assign
                break
        
        if not assignment:
            # Create new assignment
            remaining_devices = [device_profile]
            new_plan = self.planner.plan_partition(
                remaining_devices,
                target_batch_size=config.batch_size,
                gradient_accumulation=config.gradient_accumulation_steps,
                activation_checkpointing=config.activation_checkpointing
            )
            
            if new_plan.assignments:
                assignment = new_plan.assignments[0]
                plan["assignments"].append(asdict(assignment))
        
        if not assignment:
            return {
                "status": "no_assignment",
                "message": "Could not assign layers to worker"
            }
        
        # Register worker
        registration = WorkerRegistration(
            worker_id=worker_id,
            device_profile=device_profile,
            assigned_layers=assignment.layer_names if hasattr(assignment, 'layer_names') else assignment["layer_names"],
            training_mode=TrainingMode(assignment.training_mode if hasattr(assignment, 'training_mode') else assignment["training_mode"]),
            last_heartbeat=time.time(),
            status="active",
            current_step=0,
            metrics={}
        )
        
        self.registered_workers[worker_id] = registration
        self.worker_assignments[worker_id] = round_id
        
        # Update metrics
        metrics = self.round_metrics[round_id]
        metrics.num_workers = len([w for w in self.registered_workers.values() 
                                   if self.worker_assignments.get(w.worker_id) == round_id])
        
        # Check if we can start training
        if metrics.num_workers >= config.min_workers and metrics.status == RoundStatus.RECRUITING:
            metrics.status = RoundStatus.TRAINING
            asyncio.create_task(self._start_training(round_id))
        
        logger.info(f"Registered worker {worker_id} for round {round_id} with {len(registration.assigned_layers)} layers")
        
        return {
            "status": "registered",
            "round_id": round_id,
            "assignment": assignment if isinstance(assignment, dict) else asdict(assignment),
            "config": {
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "checkpoint_interval": config.checkpoint_interval
            }
        }
    
    async def submit_delta(
        self,
        round_id: str,
        worker_id: str,
        layer_name: str,
        delta_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit a layer delta from a worker.
        
        Args:
            round_id: Round identifier
            worker_id: Worker identifier
            layer_name: Layer name
            delta_data: Delta data and metadata
            
        Returns:
            Submission response
        """
        # Validate round
        if round_id not in self.active_rounds:
            return {"status": "error", "message": "Invalid round ID"}
        
        # Validate worker
        if worker_id not in self.registered_workers:
            return {"status": "error", "message": "Unregistered worker"}
        
        registration = self.registered_workers[worker_id]
        if layer_name not in registration.assigned_layers:
            return {"status": "error", "message": f"Layer {layer_name} not assigned to worker"}
        
        # Store delta
        delta_record = {
            "round_id": round_id,
            "worker_id": worker_id,
            "layer_name": layer_name,
            "delta_hash": delta_data.get("delta_hash"),
            "base_hash": delta_data.get("base_layer_hash"),
            "metrics": delta_data.get("metrics", {}),
            "timestamp": time.time(),
            "validated": False
        }
        
        self.pending_deltas[round_id].append(delta_record)
        
        # Update metrics
        metrics = self.round_metrics[round_id]
        metrics.num_deltas_submitted += 1
        
        # Trigger validation if enough deltas
        if len(self.pending_deltas[round_id]) >= metrics.num_workers:
            asyncio.create_task(self._validate_deltas(round_id))
        
        logger.info(f"Received delta for {layer_name} from {worker_id}")
        
        return {
            "status": "accepted",
            "delta_id": f"{round_id}:{layer_name}:{worker_id}"
        }
    
    async def heartbeat(self, worker_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process worker heartbeat.
        
        Args:
            worker_id: Worker identifier
            metrics: Current worker metrics
            
        Returns:
            Heartbeat response
        """
        if worker_id not in self.registered_workers:
            return {"status": "error", "message": "Unregistered worker"}
        
        registration = self.registered_workers[worker_id]
        registration.last_heartbeat = time.time()
        registration.metrics = metrics
        registration.current_step = metrics.get("step", 0)
        
        # Check if worker needs instructions
        round_id = self.worker_assignments.get(worker_id)
        if not round_id:
            return {"status": "idle"}
        
        config = self.active_rounds.get(round_id)
        if not config:
            return {"status": "round_ended"}
        
        # Check if training should stop
        if registration.current_step >= config.max_steps:
            return {
                "status": "stop_training",
                "message": "Max steps reached"
            }
        
        # Check cost limit
        metrics_obj = self.round_metrics[round_id]
        if metrics_obj.total_cost >= config.max_cost:
            return {
                "status": "stop_training",
                "message": "Cost limit reached"
            }
        
        return {
            "status": "continue",
            "current_step": registration.current_step,
            "max_steps": config.max_steps
        }
    
    async def finalize_round(self, round_id: str) -> Dict[str, Any]:
        """
        Finalize a training round.
        
        Args:
            round_id: Round identifier
            
        Returns:
            Finalization result
        """
        if round_id not in self.active_rounds:
            return {"status": "error", "message": "Invalid round ID"}
        
        config = self.active_rounds[round_id]
        metrics = self.round_metrics[round_id]
        
        # Update status
        metrics.status = RoundStatus.FINALIZING
        
        # Validate all pending deltas
        await self._validate_deltas(round_id)
        
        # Write accepted deltas to blockchain
        accepted_deltas = self.validated_deltas.get(round_id, [])
        
        for delta in accepted_deltas:
            try:
                # Add to delta index
                self.delta_index.add_delta(
                    layer_name=delta["layer_name"],
                    delta_hash=delta["delta_hash"],
                    base_hash=delta["base_hash"],
                    training_round_id=round_id,
                    validation_score=delta.get("validation_score", 0.0),
                    trainer_id=delta["worker_id"],
                    compression="dense",
                    size_bytes=delta.get("size_bytes", 0),
                    metadata=delta.get("metrics", {})
                )
                
                metrics.num_deltas_accepted += 1
                
            except Exception as e:
                logger.error(f"Failed to commit delta: {e}")
                metrics.failures.append(str(e))
        
        # Update parameter index if all deltas accepted
        if metrics.num_deltas_accepted > 0:
            logger.info(f"Committed {metrics.num_deltas_accepted} deltas for round {round_id}")
        
        # Mark round as completed
        metrics.status = RoundStatus.COMPLETED
        metrics.end_time = time.time()
        
        # Clean up
        del self.active_rounds[round_id]
        self.pending_deltas.pop(round_id, None)
        self.validated_deltas.pop(round_id, None)
        
        # Unassign workers
        for worker_id in list(self.worker_assignments.keys()):
            if self.worker_assignments[worker_id] == round_id:
                del self.worker_assignments[worker_id]
        
        return {
            "status": "completed",
            "metrics": metrics.to_dict()
        }
    
    def get_round_status(self, round_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a training round.
        
        Args:
            round_id: Round identifier
            
        Returns:
            Round status or None
        """
        if round_id not in self.round_metrics:
            return None
        
        metrics = self.round_metrics[round_id]
        config = self.active_rounds.get(round_id)
        
        # Get worker status
        workers = []
        for worker_id, registration in self.registered_workers.items():
            if self.worker_assignments.get(worker_id) == round_id:
                workers.append({
                    "worker_id": worker_id,
                    "status": registration.status,
                    "assigned_layers": registration.assigned_layers,
                    "current_step": registration.current_step,
                    "is_healthy": registration.is_healthy()
                })
        
        return {
            "round_id": round_id,
            "status": metrics.status.value,
            "config": config.to_dict() if config else None,
            "metrics": metrics.to_dict(),
            "workers": workers,
            "pending_deltas": len(self.pending_deltas.get(round_id, [])),
            "validated_deltas": len(self.validated_deltas.get(round_id, []))
        }
    
    async def _find_round_for_worker(self, device_profile: DeviceProfile) -> Optional[str]:
        """Find suitable round for worker."""
        for round_id, config in self.active_rounds.items():
            metrics = self.round_metrics[round_id]
            
            # Check if round needs workers
            if metrics.status != RoundStatus.RECRUITING:
                continue
            
            # Check if worker limit reached
            if metrics.num_workers >= config.max_workers:
                continue
            
            return round_id
        
        return None
    
    async def _get_or_create_partition_plan(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Get or create partition plan for round."""
        # Collect all registered workers for this round
        devices = []
        for worker_id, registration in self.registered_workers.items():
            if self.worker_assignments.get(worker_id) == round_id:
                devices.append(registration.device_profile)
        
        if not devices:
            return None
        
        config = self.active_rounds[round_id]
        
        # Create partition plan
        plan = self.planner.plan_partition(
            devices,
            target_batch_size=config.batch_size,
            gradient_accumulation=config.gradient_accumulation_steps,
            activation_checkpointing=config.activation_checkpointing
        )
        
        return plan.to_dict()
    
    async def _recruit_workers(self, round_id: str):
        """Wait for minimum workers to join."""
        config = self.active_rounds[round_id]
        metrics = self.round_metrics[round_id]
        
        # Wait for minimum workers
        start_time = time.time()
        timeout = 300  # 5 minutes
        
        while metrics.num_workers < config.min_workers:
            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for workers for round {round_id}")
                metrics.status = RoundStatus.FAILED
                metrics.failures.append("Insufficient workers joined")
                return
            
            await asyncio.sleep(5)
        
        logger.info(f"Recruited {metrics.num_workers} workers for round {round_id}")
    
    async def _start_training(self, round_id: str):
        """Start training phase."""
        logger.info(f"Starting training for round {round_id}")
        
        # Training is managed by workers
        # Service just monitors progress
        
    async def _validate_deltas(self, round_id: str):
        """Validate pending deltas."""
        if not self.enable_pol:
            # Move all pending to validated without PoL
            self.validated_deltas[round_id] = self.pending_deltas[round_id]
            self.pending_deltas[round_id] = []
            return
        
        # TODO: Implement PoL validation
        # For now, accept all deltas
        self.validated_deltas[round_id] = self.pending_deltas[round_id]
        self.pending_deltas[round_id] = []
        
        metrics = self.round_metrics[round_id]
        metrics.status = RoundStatus.VALIDATING
    
    async def _monitor_health(self):
        """Monitor worker health."""
        while True:
            try:
                current_time = time.time()
                
                for worker_id, registration in list(self.registered_workers.items()):
                    # Check heartbeat timeout
                    if current_time - registration.last_heartbeat > 60:
                        if registration.status == "active":
                            registration.status = "disconnected"
                            logger.warning(f"Worker {worker_id} disconnected")
                            
                            # Check if round needs replacement
                            round_id = self.worker_assignments.get(worker_id)
                            if round_id:
                                metrics = self.round_metrics.get(round_id)
                                if metrics:
                                    metrics.failures.append(f"Worker {worker_id} disconnected")
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)
    
    def get_active_rounds(self) -> List[Dict[str, Any]]:
        """Get all active rounds."""
        rounds = []
        for round_id in self.active_rounds:
            status = self.get_round_status(round_id)
            if status:
                rounds.append(status)
        return rounds
    
    def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific worker."""
        if worker_id not in self.registered_workers:
            return None
        
        registration = self.registered_workers[worker_id]
        round_id = self.worker_assignments.get(worker_id)
        
        return {
            "worker_id": worker_id,
            "status": registration.status,
            "assigned_layers": registration.assigned_layers,
            "training_mode": registration.training_mode.value,
            "current_step": registration.current_step,
            "round_id": round_id,
            "last_heartbeat": registration.last_heartbeat,
            "is_healthy": registration.is_healthy(),
            "metrics": registration.metrics
        }