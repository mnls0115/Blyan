#!/usr/bin/env python3
"""SLO-based preemptive scheduler for Blyan inference and learning coordination."""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SchedulerState(Enum):
    GREEN = "green"    # Normal operations - learning can use full allocation
    YELLOW = "yellow"  # Approaching SLO violation - throttle learning
    RED = "red"        # SLO violation - preempt learning immediately

@dataclass
class Metrics:
    """Real-time system metrics for scheduling decisions."""
    p95_latency_ms: float
    p50_latency_ms: float
    queue_depth: int
    queue_wait_ms: float
    gpu_utilization: float
    memory_free_gb: float
    arrival_rate_per_sec: float
    warm_pool_hit_ratio: float
    learning_step_duration_ms: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class SLOConfig:
    """Service Level Objective configuration."""
    target_p95_ms: float = 300
    target_queue_wait_ms: float = 150
    yellow_threshold_ratio: float = 0.7  # 70% of SLO before throttling
    warm_slots_count: int = 2
    learning_util_green: float = 0.7     # 70% GPU for learning in GREEN
    learning_util_yellow: float = 0.3    # 30% GPU for learning in YELLOW  
    learning_util_red: float = 0.0       # 0% GPU for learning in RED
    preemption_grace_ms: int = 100       # Max time to wait for graceful pause

class PreemptiveScheduler:
    """SLO-based preemptive scheduler for inference/learning coordination."""
    
    def __init__(self, config: SLOConfig = None):
        self.config = config or SLOConfig()
        self.current_state = SchedulerState.GREEN
        self.state_transitions: Dict[SchedulerState, int] = {state: 0 for state in SchedulerState}
        self.last_metrics: Optional[Metrics] = None
        
        # Hook functions - to be wired to actual components
        self.learning_pause_hook: Optional[Callable] = None
        self.learning_resume_hook: Optional[Callable[[float], None]] = None
        self.learning_throttle_hook: Optional[Callable[[float], None]] = None
        self.warm_pool_ensure_hook: Optional[Callable[[int], None]] = None
        
        # State transition history for debugging
        self.transition_history: List[Dict[str, Any]] = []
        
    def set_hooks(self,
                  learning_pause: Callable = None,
                  learning_resume: Callable[[float], None] = None, 
                  learning_throttle: Callable[[float], None] = None,
                  warm_pool_ensure: Callable[[int], None] = None):
        """Wire scheduler to actual learning and inference components."""
        self.learning_pause_hook = learning_pause
        self.learning_resume_hook = learning_resume
        self.learning_throttle_hook = learning_throttle
        self.warm_pool_ensure_hook = warm_pool_ensure
        
    def tick(self, metrics: Metrics) -> SchedulerState:
        """Main scheduling decision point - called every few seconds."""
        old_state = self.current_state
        new_state = self._compute_target_state(metrics)
        
        if new_state != old_state:
            self._transition_to_state(new_state, metrics)
            
        self.last_metrics = metrics
        return self.current_state
    
    def _compute_target_state(self, metrics: Metrics) -> SchedulerState:
        """Determine target state based on current metrics and SLO."""
        # Critical SLO violations - immediate preemption
        if (metrics.p95_latency_ms >= self.config.target_p95_ms or 
            metrics.queue_wait_ms > self.config.target_queue_wait_ms):
            return SchedulerState.RED
            
        # Approaching SLO violation - throttle learning
        yellow_p95_threshold = self.config.target_p95_ms * self.config.yellow_threshold_ratio
        yellow_queue_threshold = self.config.target_queue_wait_ms * self.config.yellow_threshold_ratio
        
        if (metrics.p95_latency_ms > yellow_p95_threshold or
            metrics.queue_wait_ms > yellow_queue_threshold):
            return SchedulerState.YELLOW
            
        # Normal operations
        return SchedulerState.GREEN
    
    def _transition_to_state(self, new_state: SchedulerState, metrics: Metrics):
        """Execute state transition with appropriate actions."""
        old_state = self.current_state
        self.current_state = new_state
        self.state_transitions[new_state] += 1
        
        # Log transition
        logger.info(f"Scheduler transition: {old_state.value} → {new_state.value} "
                   f"(p95: {metrics.p95_latency_ms:.1f}ms, queue: {metrics.queue_wait_ms:.1f}ms)")
        
        # Record transition history
        self.transition_history.append({
            "timestamp": time.time(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "trigger_p95": metrics.p95_latency_ms,
            "trigger_queue_wait": metrics.queue_wait_ms,
            "gpu_util": metrics.gpu_utilization
        })
        
        # Execute state-specific actions
        if new_state == SchedulerState.RED:
            self._handle_red_state(metrics)
        elif new_state == SchedulerState.YELLOW:
            self._handle_yellow_state(metrics) 
        else:  # GREEN
            self._handle_green_state(metrics)
    
    def _handle_red_state(self, metrics: Metrics):
        """RED state: Immediate learning preemption + warm pool boost."""
        logger.warning(f"🚨 RED state: SLO violation detected! "
                      f"p95={metrics.p95_latency_ms:.1f}ms (target: {self.config.target_p95_ms}ms)")
        
        # Immediate learning pause
        if self.learning_pause_hook:
            try:
                self.learning_pause_hook()
                logger.info("✅ Learning preempted successfully")
            except Exception as e:
                logger.error(f"❌ Learning preemption failed: {e}")
        
        # Ensure warm slots for immediate inference capacity
        if self.warm_pool_ensure_hook:
            try:
                self.warm_pool_ensure_hook(self.config.warm_slots_count)
                logger.info(f"✅ Ensured {self.config.warm_slots_count} warm slots")
            except Exception as e:
                logger.error(f"❌ Warm pool expansion failed: {e}")
                
    def _handle_yellow_state(self, metrics: Metrics):
        """YELLOW state: Throttle learning to reduce resource contention."""
        logger.warning(f"⚠️ YELLOW state: Approaching SLO limits "
                      f"p95={metrics.p95_latency_ms:.1f}ms")
        
        if self.learning_throttle_hook:
            try:
                self.learning_throttle_hook(self.config.learning_util_yellow)
                logger.info(f"✅ Learning throttled to {self.config.learning_util_yellow:.0%} GPU")
            except Exception as e:
                logger.error(f"❌ Learning throttling failed: {e}")
    
    def _handle_green_state(self, metrics: Metrics):
        """GREEN state: Normal operations - learning can use full allocation."""
        logger.info(f"✅ GREEN state: Normal operations "
                   f"p95={metrics.p95_latency_ms:.1f}ms")
        
        if self.learning_resume_hook:
            try:
                self.learning_resume_hook(self.config.learning_util_green)
                logger.info(f"✅ Learning resumed at {self.config.learning_util_green:.0%} GPU")
            except Exception as e:
                logger.error(f"❌ Learning resume failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status for monitoring."""
        return {
            "current_state": self.current_state.value,
            "state_transitions": dict(self.state_transitions),
            "config": {
                "target_p95_ms": self.config.target_p95_ms,
                "target_queue_wait_ms": self.config.target_queue_wait_ms,
                "warm_slots": self.config.warm_slots_count
            },
            "last_metrics": {
                "p95_latency_ms": self.last_metrics.p95_latency_ms if self.last_metrics else None,
                "queue_wait_ms": self.last_metrics.queue_wait_ms if self.last_metrics else None,
                "gpu_utilization": self.last_metrics.gpu_utilization if self.last_metrics else None,
                "timestamp": self.last_metrics.timestamp if self.last_metrics else None
            },
            "recent_transitions": self.transition_history[-10:] if self.transition_history else []
        }
    
    def force_state(self, target_state: SchedulerState, reason: str = "manual"):
        """Force state transition for testing/debugging."""
        if self.last_metrics is None:
            # Create dummy metrics for forced transition
            dummy_metrics = Metrics(
                p95_latency_ms=0, p50_latency_ms=0, queue_depth=0,
                queue_wait_ms=0, gpu_utilization=0, memory_free_gb=0,
                arrival_rate_per_sec=0, warm_pool_hit_ratio=0,
                learning_step_duration_ms=0
            )
        else:
            dummy_metrics = self.last_metrics
            
        logger.info(f"🔧 Forcing scheduler state to {target_state.value} (reason: {reason})")
        self._transition_to_state(target_state, dummy_metrics)


# Global scheduler instance (singleton pattern)
_global_scheduler: Optional[PreemptiveScheduler] = None

def get_scheduler() -> PreemptiveScheduler:
    """Get global scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = PreemptiveScheduler()
    return _global_scheduler

def initialize_scheduler(config: SLOConfig = None) -> PreemptiveScheduler:
    """Initialize global scheduler with configuration."""
    global _global_scheduler
    _global_scheduler = PreemptiveScheduler(config)
    return _global_scheduler


# Example usage and testing
if __name__ == "__main__":
    # Demo scheduler behavior
    scheduler = PreemptiveScheduler()
    
    # Simulate normal metrics (GREEN state)
    normal_metrics = Metrics(
        p95_latency_ms=150,   # Well below 300ms target
        p50_latency_ms=80,
        queue_depth=5,
        queue_wait_ms=50,     # Well below 150ms target
        gpu_utilization=0.6,
        memory_free_gb=8.0,
        arrival_rate_per_sec=10.0,
        warm_pool_hit_ratio=0.85,
        learning_step_duration_ms=300
    )
    
    # Simulate approaching SLO (YELLOW state)
    yellow_metrics = Metrics(
        p95_latency_ms=250,   # 70%+ of 300ms target
        p50_latency_ms=120,
        queue_depth=15,
        queue_wait_ms=120,    # 70%+ of 150ms target
        gpu_utilization=0.85,
        memory_free_gb=4.0,
        arrival_rate_per_sec=25.0,
        warm_pool_hit_ratio=0.70,
        learning_step_duration_ms=400
    )
    
    # Simulate SLO violation (RED state)
    red_metrics = Metrics(
        p95_latency_ms=350,   # Above 300ms target
        p50_latency_ms=200,
        queue_depth=30,
        queue_wait_ms=180,    # Above 150ms target
        gpu_utilization=0.95,
        memory_free_gb=1.0,
        arrival_rate_per_sec=50.0,
        warm_pool_hit_ratio=0.45,
        learning_step_duration_ms=600
    )
    
    print("=== Scheduler Behavior Demo ===")
    print(f"Initial state: {scheduler.tick(normal_metrics)}")
    print(f"High load state: {scheduler.tick(yellow_metrics)}")  
    print(f"SLO violation state: {scheduler.tick(red_metrics)}")
    print(f"Recovery state: {scheduler.tick(normal_metrics)}")
    
    print("\n=== Scheduler Status ===")
    import json
    print(json.dumps(scheduler.get_status(), indent=2))