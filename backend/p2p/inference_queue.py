#!/usr/bin/env python3
"""Async priority queue system for concurrent inference/learning management."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Priority levels for different task types."""
    CRITICAL = 0    # System-critical tasks
    HIGH = 1       # User inference requests
    NORMAL = 2     # Regular operations
    LOW = 3        # Background learning tasks
    
class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class InferenceMetrics:
    """Real-time metrics for priority calculation."""
    p95_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    def calculate_dynamic_priority(self, base_priority: TaskPriority) -> float:
        """Calculate dynamic priority based on current system state."""
        # Higher p95 latency = higher priority for new requests
        latency_factor = min(self.p95_latency_ms / 300.0, 2.0)  # Cap at 2x
        
        # Queue depth factor - prioritize when queue is getting full
        queue_factor = min(self.queue_depth / 50.0, 1.5)
        
        # Base priority weight
        base_weight = base_priority.value
        
        # Lower value = higher priority
        return base_weight - (latency_factor * 0.5) - (queue_factor * 0.3)

@dataclass
class QueuedTask:
    """Task wrapper for priority queue."""
    task_id: str
    task_type: str  # 'inference', 'learning', 'maintenance'
    priority: TaskPriority
    created_at: float
    future: asyncio.Future
    payload: Dict[str, Any]
    dynamic_priority: float = field(init=False)
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        self.dynamic_priority = self.priority.value
        
    def __lt__(self, other):
        """Lower dynamic_priority value = higher priority."""
        return self.dynamic_priority < other.dynamic_priority

class InferenceQueue:
    """Priority queue manager for inference and learning tasks."""
    
    def __init__(self, max_size: int = 100, num_workers: int = 3):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self.num_workers = num_workers
        self.workers: List[asyncio.Task] = []
        self.metrics = InferenceMetrics()
        self.active_tasks: Dict[str, QueuedTask] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self._shutdown = False
        self._task_counter = 0
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.max_history_size = 1000
        
    def register_handler(self, task_type: str, handler: Callable):
        """Register async handler for specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
        
    async def submit_task(
        self, 
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> Any:
        """Submit task to queue and wait for result."""
        if self._shutdown:
            raise RuntimeError("Queue is shutting down")
            
        # Generate task ID
        self._task_counter += 1
        task_id = f"{task_type}_{self._task_counter}_{time.time()}"
        
        # Create future for result
        future = asyncio.get_event_loop().create_future()
        
        # Create task
        task = QueuedTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            created_at=time.time(),
            future=future,
            payload=payload
        )
        
        # Calculate dynamic priority
        task.dynamic_priority = self.metrics.calculate_dynamic_priority(priority)
        
        # Add to queue
        try:
            await self.queue.put(task)
            self.metrics.queue_depth = self.queue.qsize()
            logger.debug(f"Submitted task {task_id} with priority {task.dynamic_priority:.2f}")
        except asyncio.QueueFull:
            future.set_exception(Exception("Queue is full"))
            raise
            
        # Wait for result with timeout
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            return result
        except asyncio.TimeoutError:
            # Cancel task if possible
            task.future.cancel()
            raise
            
    async def _worker(self, worker_id: int):
        """Worker coroutine to process tasks from queue."""
        logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get task with timeout to allow shutdown check
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Update metrics
                self.metrics.active_workers += 1
                self.active_tasks[task.task_id] = task
                
                # Process task
                start_time = time.time()
                try:
                    # Get handler
                    handler = self.task_handlers.get(task.task_type)
                    if not handler:
                        raise ValueError(f"No handler for task type: {task.task_type}")
                        
                    # Execute handler
                    logger.debug(f"Worker {worker_id} processing {task.task_id}")
                    result = await handler(task.payload)
                    
                    # Set result
                    if not task.future.done():
                        task.future.set_result(result)
                        
                    # Update metrics
                    latency_ms = (time.time() - start_time) * 1000
                    self._update_latency_metrics(latency_ms)
                    self.metrics.completed_tasks += 1
                    
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    
                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.dynamic_priority -= 0.5  # Boost priority for retry
                        await self.queue.put(task)
                        logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                    else:
                        if not task.future.done():
                            task.future.set_exception(e)
                        self.metrics.failed_tasks += 1
                        
                finally:
                    # Cleanup
                    self.metrics.active_workers -= 1
                    self.active_tasks.pop(task.task_id, None)
                    self.metrics.queue_depth = self.queue.qsize()
                    
            except asyncio.TimeoutError:
                # Normal timeout, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} stopped")
        
    def _update_latency_metrics(self, latency_ms: float):
        """Update latency metrics with new measurement."""
        self.latency_history.append(latency_ms)
        
        # Maintain history size
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
            
        # Calculate percentiles
        if self.latency_history:
            sorted_latencies = sorted(self.latency_history)
            p50_idx = int(len(sorted_latencies) * 0.5)
            p95_idx = int(len(sorted_latencies) * 0.95)
            
            self.metrics.p50_latency_ms = sorted_latencies[p50_idx]
            self.metrics.p95_latency_ms = sorted_latencies[p95_idx]
            
    async def start(self):
        """Start worker tasks."""
        if self.workers:
            raise RuntimeError("Queue already started")
            
        self._shutdown = False
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
            
        logger.info(f"Started {self.num_workers} workers")
        
    async def stop(self, timeout: float = 30.0):
        """Gracefully stop all workers."""
        logger.info("Stopping inference queue...")
        self._shutdown = True
        
        # Wait for workers to finish current tasks
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Some workers did not stop gracefully")
            
        # Cancel remaining tasks
        for task in self.active_tasks.values():
            if not task.future.done():
                task.future.cancel()
                
        self.workers.clear()
        logger.info("Inference queue stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_depth": self.metrics.queue_depth,
            "active_workers": self.metrics.active_workers,
            "completed_tasks": self.metrics.completed_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "p50_latency_ms": self.metrics.p50_latency_ms,
            "p95_latency_ms": self.metrics.p95_latency_ms,
            "active_task_ids": list(self.active_tasks.keys()),
            "handlers": list(self.task_handlers.keys())
        }

# Example handlers for testing
async def example_inference_handler(payload: Dict[str, Any]) -> str:
    """Example inference handler."""
    prompt = payload.get("prompt", "")
    model_name = payload.get("model_name", "default")
    
    # Simulate inference
    await asyncio.sleep(0.1)  # 100ms inference time
    return f"Inference result for '{prompt}' using {model_name}"

async def example_learning_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Example learning handler with micro-steps."""
    num_steps = payload.get("num_steps", 10)
    step_duration = payload.get("step_duration", 0.05)  # 50ms per step
    
    results = []
    for i in range(num_steps):
        # Simulate micro-step
        await asyncio.sleep(step_duration)
        results.append(f"Step {i+1} completed")
        
        # Check if we should yield (simplified check)
        if i % 4 == 0:  # Every 4 steps
            await asyncio.sleep(0)  # Yield control
            
    return {
        "status": "completed",
        "steps": results,
        "total_time": num_steps * step_duration
    }

# Demo usage
async def demo_queue():
    """Demonstrate queue functionality."""
    # Create queue
    queue = InferenceQueue(max_size=50, num_workers=3)
    
    # Register handlers
    queue.register_handler("inference", example_inference_handler)
    queue.register_handler("learning", example_learning_handler)
    
    # Start queue
    await queue.start()
    
    try:
        # Submit mixed tasks
        tasks = []
        
        # High priority inference
        for i in range(5):
            task = queue.submit_task(
                "inference",
                {"prompt": f"Query {i}", "model_name": "moe_model"},
                priority=TaskPriority.HIGH
            )
            tasks.append(task)
            
        # Low priority learning
        learning_task = queue.submit_task(
            "learning",
            {"num_steps": 20, "step_duration": 0.05},
            priority=TaskPriority.LOW
        )
        tasks.append(learning_task)
        
        # More inference (should preempt learning)
        for i in range(5, 10):
            task = queue.submit_task(
                "inference", 
                {"prompt": f"Urgent query {i}"},
                priority=TaskPriority.HIGH
            )
            tasks.append(task)
            
        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print results
        for i, result in enumerate(results):
            print(f"Task {i}: {result}")
            
        # Print final status
        print("\nFinal queue status:")
        status = queue.get_status()
        for key, value in status.items():
            if key != "active_task_ids":  # Skip long list
                print(f"  {key}: {value}")
                
    finally:
        await queue.stop()

if __name__ == "__main__":
    asyncio.run(demo_queue())