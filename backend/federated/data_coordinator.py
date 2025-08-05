#!/usr/bin/env python3
"""Federated learning data coordination for Blyan."""

from __future__ import annotations

import json
import time
import hashlib
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from backend.core.scheduler import PreemptiveScheduler, Metrics, SchedulerState

@dataclass
class BenchmarkTask:
    """Standard benchmark task for expert evaluation."""
    task_id: str
    task_type: str  # "classification", "generation", "reasoning"
    input_data: List[str]
    expected_outputs: List[str]
    evaluation_metric: str  # "accuracy", "bleu", "perplexity"
    
@dataclass
class LocalDataContribution:
    """Represents a node's local data contribution."""
    node_id: str
    data_hash: str  # Hash of local data (for verification without sharing)
    sample_count: int
    data_quality_score: float
    privacy_level: str  # "public", "federated", "private"

class FederatedDataCoordinator:
    """Coordinates federated learning without centralizing private data."""
    
    def __init__(self, benchmark_path: Path):
        self.benchmark_path = benchmark_path
        self.benchmark_tasks: Dict[str, BenchmarkTask] = {}
        self.node_contributions: Dict[str, LocalDataContribution] = {}
        self.load_standard_benchmarks()
    
    def load_standard_benchmarks(self):
        """Load standardized benchmark tasks for expert evaluation."""
        # Common sense reasoning tasks
        reasoning_tasks = [
            "The sky is usually what color? (A) Blue (B) Green (C) Red",
            "What do people use to write? (A) Hammer (B) Pen (C) Fork",
            "Where do fish live? (A) Trees (B) Sky (C) Water"
        ]
        
        reasoning_answers = ["A", "B", "C"]
        
        self.benchmark_tasks["reasoning"] = BenchmarkTask(
            task_id="common_reasoning",
            task_type="classification", 
            input_data=reasoning_tasks,
            expected_outputs=reasoning_answers,
            evaluation_metric="accuracy"
        )
        
        # Text generation tasks
        generation_prompts = [
            "Complete the sentence: The weather today is",
            "Write a short greeting: Hello, my name is", 
            "Explain in simple terms: Artificial intelligence is"
        ]
        
        # Note: In real implementation, these would be high-quality reference outputs
        generation_references = [
            "sunny and pleasant",
            "AI Assistant, and I'm here to help you",
            "a technology that enables computers to perform tasks that typically require human intelligence"
        ]
        
        self.benchmark_tasks["generation"] = BenchmarkTask(
            task_id="text_generation",
            task_type="generation",
            input_data=generation_prompts,
            expected_outputs=generation_references,
            evaluation_metric="semantic_similarity"
        )
        
        print(f"✅ Loaded {len(self.benchmark_tasks)} standard benchmark tasks")
    
    def register_node_data(self, contribution: LocalDataContribution):
        """Register a node's local data contribution without accessing the data."""
        self.node_contributions[contribution.node_id] = contribution
        print(f"📊 Registered data contribution from {contribution.node_id}")
        print(f"   • Sample count: {contribution.sample_count}")
        print(f"   • Quality score: {contribution.data_quality_score:.2f}")
        print(f"   • Privacy level: {contribution.privacy_level}")
    
    def evaluate_expert_on_benchmarks(self, expert_name: str, expert_weights: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate expert performance on standard benchmarks."""
        results = {}
        
        for task_name, task in self.benchmark_tasks.items():
            print(f"🧪 Evaluating {expert_name} on {task_name} benchmark...")
            
            # Simulate expert evaluation (in real implementation, load expert and run inference)
            if task.task_type == "classification":
                # Simulate classification accuracy
                accuracy = self._simulate_classification_performance(expert_weights, task)
                results[f"{task_name}_accuracy"] = accuracy
                
            elif task.task_type == "generation":
                # Simulate generation quality
                quality_score = self._simulate_generation_performance(expert_weights, task)
                results[f"{task_name}_quality"] = quality_score
        
        return results
    
    def _simulate_classification_performance(self, expert_weights: Dict[str, Any], task: BenchmarkTask) -> float:
        """Simulate classification performance based on expert weights."""
        # In real implementation, this would run actual inference
        # For now, simulate based on expert complexity/quality
        
        param_count = sum(w.numel() if hasattr(w, 'numel') else len(str(w)) for w in expert_weights.values())
        
        # Simulate: larger, more complex experts tend to perform better (with some randomness)
        base_accuracy = min(0.95, 0.3 + (param_count / 1000000) * 0.5)  # Scale with parameter count
        noise = np.random.normal(0, 0.1)  # Add some realistic variance
        
        return max(0.1, min(0.99, base_accuracy + noise))
    
    def _simulate_generation_performance(self, expert_weights: Dict[str, Any], task: BenchmarkTask) -> float:
        """Simulate text generation quality based on expert weights."""
        # In real implementation, this would use BLEU, ROUGE, or semantic similarity metrics
        
        param_count = sum(w.numel() if hasattr(w, 'numel') else len(str(w)) for w in expert_weights.values())
        
        # Simulate generation quality
        base_quality = min(0.9, 0.2 + (param_count / 2000000) * 0.6)
        noise = np.random.normal(0, 0.15)
        
        return max(0.05, min(0.95, base_quality + noise))
    
    def calculate_federated_improvement_score(
        self, 
        old_expert_weights: Dict[str, Any], 
        new_expert_weights: Dict[str, Any],
        expert_name: str
    ) -> float:
        """Calculate improvement score using federated evaluation approach."""
        
        print(f"🔍 Calculating federated improvement for {expert_name}...")
        
        # Evaluate old expert
        old_scores = self.evaluate_expert_on_benchmarks(f"{expert_name}_old", old_expert_weights)
        
        # Evaluate new expert  
        new_scores = self.evaluate_expert_on_benchmarks(f"{expert_name}_new", new_expert_weights)
        
        # Calculate improvement across all benchmarks
        improvements = []
        for task_name in self.benchmark_tasks.keys():
            for metric in ["accuracy", "quality"]:
                old_key = f"{task_name}_{metric}"
                new_key = f"{task_name}_{metric}"
                
                if old_key in old_scores and new_key in new_scores:
                    improvement = new_scores[new_key] - old_scores[old_key]
                    improvements.append(improvement)
                    print(f"   • {old_key}: {old_scores[old_key]:.3f} → {new_scores[new_key]:.3f} ({improvement:+.3f})")
        
        # Average improvement across all tasks
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # Convert to 0-1 scale (negative improvements get score 0)
        improvement_score = max(0.0, min(1.0, 0.5 + avg_improvement * 2))
        
        print(f"📈 Overall improvement score: {improvement_score:.3f}")
        return improvement_score
    
    def get_privacy_preserving_data_summary(self, node_id: str) -> Dict[str, Any]:
        """Get summary of node's data contribution without exposing raw data."""
        if node_id not in self.node_contributions:
            return {}
        
        contribution = self.node_contributions[node_id]
        
        return {
            "node_id": node_id,
            "data_fingerprint": contribution.data_hash[:16],  # Truncated hash for privacy
            "contribution_score": contribution.data_quality_score,
            "sample_count_tier": self._get_sample_count_tier(contribution.sample_count),
            "privacy_level": contribution.privacy_level,
            "last_updated": time.time()
        }
    
    def _get_sample_count_tier(self, sample_count: int) -> str:
        """Convert sample count to privacy-preserving tier."""
        if sample_count < 100:
            return "small"
        elif sample_count < 1000:
            return "medium" 
        elif sample_count < 10000:
            return "large"
        else:
            return "very_large"

class PrivacyPreservingTraining:
    """Handles privacy-preserving aspects of federated learning."""
    
    @staticmethod
    def add_differential_privacy_noise(gradients: Dict[str, Any], privacy_budget: float = 1.0) -> Dict[str, Any]:
        """Add calibrated noise to gradients for differential privacy."""
        noisy_gradients = {}
        
        for param_name, grad_value in gradients.items():
            if hasattr(grad_value, 'shape'):  # PyTorch tensor
                noise_scale = 2.0 / privacy_budget  # Calibrate noise to privacy budget
                noise = np.random.laplace(0, noise_scale, grad_value.shape)
                noisy_gradients[param_name] = grad_value + noise
            else:
                # Handle scalar gradients
                noise = np.random.laplace(0, 2.0 / privacy_budget)
                noisy_gradients[param_name] = grad_value + noise
        
        return noisy_gradients
    
    @staticmethod
    def create_data_fingerprint(local_data: List[str]) -> str:
        """Create privacy-preserving fingerprint of local data."""
        # Create hash that represents data distribution without exposing content
        data_concat = "".join(sorted(local_data))  # Sort to ensure consistency
        return hashlib.sha256(data_concat.encode()).hexdigest()


class MicroSteppingLearningLoop:
    """Learning loop with micro-stepping support for preemptive scheduling."""
    
    def __init__(self, 
                 step_duration_ms: int = 300,
                 checkpoint_interval_sec: int = 90,
                 gradient_accumulation_steps: int = 4,
                 scheduler: Optional[PreemptiveScheduler] = None):
        self.step_duration_ms = step_duration_ms
        self.checkpoint_interval_sec = checkpoint_interval_sec
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Control flags
        self._paused = False
        self._target_gpu_util = 0.7
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        
        # Scheduler integration
        self.scheduler = scheduler
        self.learning_metrics = {"active_learners": 0, "avg_step_time": 0.0, "total_steps": 0}
        
        # State tracking
        self._last_checkpoint_time = time.time()
        self._current_step = 0
        self._total_steps_completed = 0
        self._accumulated_gradients = 0
        
        # Performance metrics
        self._step_times: List[float] = []
        self._gpu_utilization_samples: List[float] = []
        
    def pause(self, grace_ms: int = 100) -> bool:
        """Pause learning at the next micro-step boundary."""
        self._paused = True
        
        # Wait for graceful pause within grace period
        start_time = time.time()
        while self._is_in_step() and (time.time() - start_time) * 1000 < grace_ms:
            time.sleep(0.01)  # 10ms polling
            
        return not self._is_in_step()  # True if successfully paused
    
    def resume(self, target_gpu_util: float = 0.7):
        """Resume learning with specified GPU utilization target."""
        self._target_gpu_util = target_gpu_util
        self._paused = False
        
        if not self._running:
            self.start_learning_loop()
            
    def throttle(self, target_gpu_util: float = 0.3):
        """Throttle learning to specified GPU utilization."""
        self._target_gpu_util = target_gpu_util
        # Note: Actual GPU throttling would be implemented in the training framework
        
    def start_learning_loop(self):
        """Start the learning loop in a separate thread."""
        if self._running:
            return
            
        self._running = True
        self._loop_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._loop_thread.start()
        
    def stop_learning_loop(self):
        """Stop the learning loop."""
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
            
    def _learning_loop(self):
        """Main learning loop with micro-stepping."""
        while self._running:
            if self._paused:
                time.sleep(0.1)  # Sleep while paused
                continue
                
            step_start = time.time()
            
            try:
                # Execute micro-step (200-500ms target)
                self._execute_micro_step()
                
                # Track step timing
                step_duration = (time.time() - step_start) * 1000
                self._step_times.append(step_duration)
                
                # Keep only recent samples for monitoring
                if len(self._step_times) > 100:
                    self._step_times = self._step_times[-50:]
                    
                # Checkpoint periodically
                if time.time() - self._last_checkpoint_time > self.checkpoint_interval_sec:
                    self._async_checkpoint()
                    
            except Exception as e:
                print(f"❌ Learning step failed: {e}")
                time.sleep(1.0)  # Back off on error
                
            # Yield at step boundary for preemption
            self._yield_at_boundary()
            
    def _execute_micro_step(self):
        """Execute a single micro-step of learning with scheduler integration."""
        step_start_time = time.time()
        
        # Check scheduler state before learning step
        if self.scheduler:
            current_metrics = Metrics(
                p95_latency_ms=100.0,  # Mock inference latency
                p50_latency_ms=80.0,   # Mock inference latency
                queue_depth=0,         # No inference queue in learning loop
                queue_wait_ms=0.0,     # No queue waiting
                gpu_utilization=self._target_gpu_util,
                memory_free_gb=6.0,    # Mock value
                arrival_rate_per_sec=0.0,  # No arrivals during learning
                warm_pool_hit_ratio=1.0,   # Not relevant for learning
                learning_step_duration_ms=self.learning_metrics["avg_step_time"] * 1000
            )
            
            scheduler_state = self.scheduler.tick(current_metrics)
            
            # Respond to scheduler state
            if scheduler_state == SchedulerState.RED:
                print(f"🔴 Scheduler demands pause: yielding to inference")
                self.pause()
                time.sleep(0.1)  # Brief pause to allow inference to catch up
                return
            elif scheduler_state == SchedulerState.YELLOW:
                print(f"🟡 Scheduler caution: reducing learning intensity")
                self._target_gpu_util = 0.5  # Reduce GPU utilization to make room for inference
            else:
                self._target_gpu_util = 0.7  # Normal GPU utilization
        
        # Simulate gradient computation (replace with actual training code)
        target_duration = self.step_duration_ms / 1000.0
        
        # Simulate work based on target GPU utilization
        work_duration = target_duration * self._target_gpu_util
        
        # Mock training step
        while time.time() - step_start_time < work_duration:
            # Simulate computation (replace with actual forward/backward pass)
            _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
            
        self._accumulated_gradients += 1
        
        # Apply gradients after accumulation
        if self._accumulated_gradients >= self.gradient_accumulation_steps:
            self._apply_accumulated_gradients()
            self._accumulated_gradients = 0
            self._current_step += 1
            
        self._total_steps_completed += 1
        
        # Update learning metrics for scheduler
        step_time = time.time() - step_start_time
        self.learning_metrics["total_steps"] += 1
        if self.learning_metrics["total_steps"] > 0:
            self.learning_metrics["avg_step_time"] = (
                (self.learning_metrics["avg_step_time"] * (self.learning_metrics["total_steps"] - 1) + step_time) / 
                self.learning_metrics["total_steps"]
            )
        
    def _apply_accumulated_gradients(self):
        """Apply accumulated gradients to model parameters."""
        # Simulate gradient application
        # In real implementation: optimizer.step(), optimizer.zero_grad()
        pass
        
    def _async_checkpoint(self):
        """Create lightweight checkpoint asynchronously."""
        # Simulate async checkpoint (replace with actual checkpoint logic)
        self._last_checkpoint_time = time.time()
        
        # In real implementation:
        # - Save model state_dict (FP16 to reduce size)
        # - Save optimizer state (compressed)
        # - Use background thread for I/O
        print(f"📁 Checkpoint created at step {self._current_step}")
        
    def _yield_at_boundary(self):
        """Yield control at micro-step boundary for preemption."""
        # Allow preemption by sleeping briefly
        time.sleep(0.001)  # 1ms yield
        
    def _is_in_step(self) -> bool:
        """Check if currently executing a micro-step."""
        # In real implementation, this would check training state
        return not self._paused
        
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status for monitoring."""
        avg_step_time = np.mean(self._step_times) if self._step_times else 0
        
        return {
            "running": self._running,
            "paused": self._paused,
            "target_gpu_util": self._target_gpu_util,
            "current_step": self._current_step,
            "total_steps_completed": self._total_steps_completed,
            "accumulated_gradients": self._accumulated_gradients,
            "avg_step_duration_ms": avg_step_time,
            "recent_step_times": self._step_times[-10:] if self._step_times else [],
            "last_checkpoint_time": self._last_checkpoint_time,
            "checkpoint_interval_sec": self.checkpoint_interval_sec
        }


# Global learning loop instance  
_global_learning_loop: Optional[MicroSteppingLearningLoop] = None

def get_learning_loop() -> MicroSteppingLearningLoop:
    """Get global learning loop instance."""
    global _global_learning_loop
    if _global_learning_loop is None:
        _global_learning_loop = MicroSteppingLearningLoop()
    return _global_learning_loop


# Example usage and testing
def demo_federated_coordination():
    """Demonstrate federated learning coordination."""
    coordinator = FederatedDataCoordinator(Path("./benchmarks"))
    
    # Simulate nodes registering their local data
    node1_data = LocalDataContribution(
        node_id="node1",
        data_hash="abc123...",
        sample_count=5000,
        data_quality_score=0.85,
        privacy_level="federated"
    )
    
    node2_data = LocalDataContribution(
        node_id="node2", 
        data_hash="def456...",
        sample_count=12000,
        data_quality_score=0.92,
        privacy_level="private"
    )
    
    coordinator.register_node_data(node1_data)
    coordinator.register_node_data(node2_data)
    
    # Simulate expert improvement evaluation
    old_expert = {"layer.weight": np.random.randn(100, 50)}
    new_expert = {"layer.weight": np.random.randn(100, 50) * 1.1}  # Slightly different
    
    improvement_score = coordinator.calculate_federated_improvement_score(
        old_expert, new_expert, "layer0.expert1"
    )
    
    print(f"\n🎯 Final improvement score: {improvement_score:.3f}")

if __name__ == "__main__":
    demo_federated_coordination()