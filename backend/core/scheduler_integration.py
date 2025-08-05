#!/usr/bin/env python3
"""Scheduler integration utilities for connecting inference and learning components."""

from typing import Optional
from .scheduler import PreemptiveScheduler

# Global scheduler instance for system-wide coordination
_global_scheduler: Optional[PreemptiveScheduler] = None

def get_global_scheduler() -> PreemptiveScheduler:
    """Get the global scheduler instance, creating it if needed."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = PreemptiveScheduler()
    return _global_scheduler

def initialize_scheduler() -> PreemptiveScheduler:
    """Initialize the global scheduler for system-wide use."""
    global _global_scheduler
    _global_scheduler = PreemptiveScheduler()
    return _global_scheduler

def reset_scheduler():
    """Reset the global scheduler (mainly for testing)."""
    global _global_scheduler
    _global_scheduler = None

class SchedulerIntegration:
    """Helper class for integrating scheduler with system components."""
    
    def __init__(self, scheduler: Optional[PreemptiveScheduler] = None):
        self.scheduler = scheduler or get_global_scheduler()
        
    def connect_inference_coordinator(self, coordinator):
        """Connect scheduler to inference coordinator."""
        coordinator.scheduler = self.scheduler
        return coordinator
        
    def connect_learning_loop(self, learning_loop):
        """Connect scheduler to learning loop."""
        learning_loop.scheduler = self.scheduler
        return learning_loop
        
    def get_scheduler_status(self) -> dict:
        """Get current scheduler status for monitoring."""
        return {
            "current_state": str(self.scheduler.current_state.value),
            "state_transitions": self.scheduler.state_transitions,
            "transition_history": self.scheduler.transition_history[-5:] if self.scheduler.transition_history else [],
            "last_metrics": self.scheduler.last_metrics.__dict__ if self.scheduler.last_metrics else None
        }

# Example usage for wiring components together
def wire_system_components(inference_coordinator, learning_loop):
    """Wire inference coordinator and learning loop with shared scheduler."""
    
    # Initialize shared scheduler
    scheduler = initialize_scheduler()
    
    # Connect components
    integration = SchedulerIntegration(scheduler)
    integration.connect_inference_coordinator(inference_coordinator)
    integration.connect_learning_loop(learning_loop)
    
    print("✅ Scheduler wired to inference coordinator and learning loop")
    
    return {
        "scheduler": scheduler,
        "integration": integration,
        "inference_coordinator": inference_coordinator,
        "learning_loop": learning_loop
    }