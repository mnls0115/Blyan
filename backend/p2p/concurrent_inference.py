#!/usr/bin/env python3
"""Concurrent inference system with learning/inference coordination."""

import asyncio
import time
import torch
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from .inference_queue import InferenceQueue, TaskPriority, InferenceMetrics
from backend.model.moe_infer import MoEModelManager, ExpertUsageTracker
from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex
from backend.learning.micro_step_trainer import MicroStepTrainer, MicroStepConfig
from backend.learning.dual_model_manager import DualModelManager, StreamConfig
from backend.inference.batch_manager import BatchManager, BatchConfig, BatchedInferenceEngine

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for concurrent inference system."""
    max_queue_size: int = 100
    num_inference_workers: int = 3
    num_learning_workers: int = 1
    micro_step_duration_ms: int = 100
    max_batch_size: int = 8
    memory_fraction: float = 0.85
    enable_dual_instances: bool = True
    cache_size: int = 100

class ConcurrentInferenceManager:
    """Manages concurrent inference and learning with intelligent scheduling."""
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        
        # Initialize queue system
        self.inference_queue = InferenceQueue(
            max_size=self.config.max_queue_size,
            num_workers=self.config.num_inference_workers
        )
        
        # Model managers (dual instance support)
        self.inference_model: Optional[MoEModelManager] = None
        self.learning_model: Optional[MoEModelManager] = None
        
        # Learning control
        self.learning_pause_event = asyncio.Event()
        self.learning_pause_event.set()  # Start resumed
        self.learning_active = False
        self.micro_step_trainer: Optional[MicroStepTrainer] = None
        self.dual_model_manager: Optional[DualModelManager] = None
        
        # Batch accumulator
        self.batch_accumulator: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        
        # Performance tracking
        self.inference_count = 0
        self.learning_steps = 0
        self.cache_hits = 0
        
        # Result cache (simple LRU)
        self.result_cache: Dict[str, Tuple[str, float]] = {}
        
        # Batch manager for efficient inference
        self.batch_manager: Optional[BatchManager] = None
        self.batched_engine: Optional[BatchedInferenceEngine] = None
        
    async def initialize(self, root_dir: Path):
        """Initialize model managers and chains."""
        try:
            # Load blockchain chains
            meta_chain = Chain(root_dir, "A")
            param_chain = Chain(root_dir, "B")
            param_index = ParameterIndex(param_chain)
            usage_tracker = ExpertUsageTracker(root_dir / "expert_usage.json")
            
            # Initialize inference model
            self.inference_model = MoEModelManager(
                meta_chain=meta_chain,
                param_chain=param_chain,
                param_index=param_index,
                usage_tracker=usage_tracker
            )
            
            # Optionally create separate learning instance
            if self.config.enable_dual_instances:
                self.learning_model = MoEModelManager(
                    meta_chain=meta_chain,
                    param_chain=param_chain,
                    param_index=param_index,
                    usage_tracker=usage_tracker
                )
                
            # Set memory limits
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
                
            # Initialize batch manager
            batch_config = BatchConfig(
                max_batch_size=self.config.max_batch_size,
                max_wait_time_ms=50,
                enable_dynamic_batching=True,
                cache_size=self.config.cache_size
            )
            self.batch_manager = BatchManager(batch_config)
            
            # Register handlers
            self.inference_queue.register_handler("inference", self._handle_inference)
            self.inference_queue.register_handler("batch_inference", self._handle_batch_inference)
            self.inference_queue.register_handler("learning_step", self._handle_learning_step)
            self.inference_queue.register_handler("dual_inference", self._handle_dual_inference)
            self.inference_queue.register_handler("dual_learning", self._handle_dual_learning)
            self.inference_queue.register_handler("batched_inference", self._handle_batched_inference)
            
            # Start queue
            await self.inference_queue.start()
            
            logger.info("Concurrent inference manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
            
    async def submit_inference(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        required_experts: Optional[List[str]] = None,
        use_cache: bool = True,
        priority: TaskPriority = TaskPriority.HIGH
    ) -> str:
        """Submit inference request with automatic batching and caching."""
        
        # Check cache first
        if use_cache:
            cache_key = f"{prompt}_{max_new_tokens}_{required_experts}"
            if cache_key in self.result_cache:
                result, timestamp = self.result_cache[cache_key]
                if time.time() - timestamp < 300:  # 5 min cache
                    self.cache_hits += 1
                    return result
                    
        # Accumulate for batching
        async with self.batch_lock:
            self.batch_accumulator.append({
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "required_experts": required_experts,
                "cache_key": cache_key if use_cache else None
            })
            
            # Check if we should trigger batch
            if len(self.batch_accumulator) >= self.config.max_batch_size:
                return await self._trigger_batch(priority)
                
        # For single requests or timeout, submit directly
        result = await self.inference_queue.submit_task(
            "inference",
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "required_experts": required_experts
            },
            priority=priority
        )
        
        # Update cache
        if use_cache:
            self.result_cache[cache_key] = (result, time.time())
            self._prune_cache()
            
        return result
        
    async def _trigger_batch(self, priority: TaskPriority) -> str:
        """Trigger batch inference for accumulated requests."""
        batch = self.batch_accumulator.copy()
        self.batch_accumulator.clear()
        
        # Submit batch task
        results = await self.inference_queue.submit_task(
            "batch_inference",
            {"batch": batch},
            priority=priority
        )
        
        # Return first result (caller's result)
        # In production, would need proper result routing
        return results[0] if results else "Batch processing failed"
        
    async def _handle_inference(self, payload: Dict[str, Any]) -> str:
        """Handle single inference request."""
        if not self.inference_model:
            raise RuntimeError("Inference model not initialized")
            
        # Request learning yield if using micro-step trainer
        if self.micro_step_trainer and self.learning_active:
            self.micro_step_trainer.request_yield(priority=0.8)
            
        # Pause learning if active (legacy path)
        elif self.learning_active:
            self.learning_pause_event.clear()
            await asyncio.sleep(0.01)  # Brief yield
            
        try:
            # Perform inference
            start_time = time.time()
            result, expert_usage = self.inference_model.selective_generate(
                prompt=payload["prompt"],
                max_new_tokens=payload.get("max_new_tokens", 64),
                required_experts=payload.get("required_experts")
            )
            
            inference_time = time.time() - start_time
            self.inference_count += 1
            
            logger.debug(f"Inference completed in {inference_time:.2f}s")
            
            return result
            
        finally:
            # Resume learning
            if self.learning_active:
                self.learning_pause_event.set()
                
    async def _handle_batch_inference(self, payload: Dict[str, Any]) -> List[str]:
        """Handle batched inference requests."""
        batch = payload["batch"]
        if not batch:
            return []
            
        # Group by similar parameters for efficiency
        grouped = {}
        for item in batch:
            key = (item["max_new_tokens"], tuple(item.get("required_experts") or []))
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
            
        results = []
        
        # Process each group
        for (max_tokens, experts), group_items in grouped.items():
            prompts = [item["prompt"] for item in group_items]
            
            # Batch inference (simplified - would need proper batching in model)
            for prompt in prompts:
                result = await self._handle_inference({
                    "prompt": prompt,
                    "max_new_tokens": max_tokens,
                    "required_experts": list(experts) if experts else None
                })
                results.append(result)
                
                # Update caches
                for item in group_items:
                    if item.get("cache_key"):
                        self.result_cache[item["cache_key"]] = (result, time.time())
                        
        self._prune_cache()
        return results
        
    async def _handle_learning_step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle micro-step learning with interruption support."""
        if not self.learning_model:
            return {"status": "no_learning_model"}
            
        # Use micro-step trainer if available
        if payload.get("use_micro_step_trainer", True) and self.inference_model:
            return await self._handle_micro_step_learning(payload)
            
        # Legacy learning path
        num_steps = payload.get("num_steps", 10)
        step_duration = payload.get("step_duration", 0.1)
        
        self.learning_active = True
        completed_steps = []
        
        try:
            for i in range(num_steps):
                # Wait for resume signal
                await self.learning_pause_event.wait()
                
                # Simulate learning step (in real implementation, actual training)
                start = time.time()
                
                # Check queue pressure
                queue_depth = self.inference_queue.metrics.queue_depth
                if queue_depth > 10:
                    # Yield more frequently under pressure
                    step_duration = min(step_duration, 0.05)
                    
                # Actual learning work would go here
                await asyncio.sleep(step_duration)
                
                step_time = time.time() - start
                completed_steps.append({
                    "step": i + 1,
                    "duration": step_time,
                    "interrupted": not self.learning_pause_event.is_set()
                })
                
                self.learning_steps += 1
                
                # Micro-yield point
                if i % 4 == 0 or queue_depth > 5:
                    await asyncio.sleep(0)  # Yield to event loop
                    
        finally:
            self.learning_active = False
            
        return {
            "status": "completed",
            "total_steps": len(completed_steps),
            "completed_steps": completed_steps,
            "average_step_time": sum(s["duration"] for s in completed_steps) / len(completed_steps)
        }
        
    async def _handle_micro_step_learning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle learning with micro-step trainer."""
        import torch.nn as nn
        import torch.optim as optim
        
        # Initialize trainer if needed
        if not self.micro_step_trainer:
            # Use actual model or create placeholder
            if self.inference_model and hasattr(self.inference_model, 'model'):
                model = self.inference_model.model
            else:
                # Fallback model - TODO: replace with actual model loading
                hidden_size = payload.get("hidden_size", 512)
                model = nn.Linear(hidden_size, hidden_size)
            optimizer = optim.Adam(model.parameters(), lr=payload.get("learning_rate", 1e-4))
            
            config = MicroStepConfig(
                min_step_duration_ms=payload.get("min_step_ms", 50),
                max_step_duration_ms=payload.get("max_step_ms", 200),
                adaptive_step_sizing=True,
                gradient_accumulation_steps=payload.get("gradient_accumulation", 4)
            )
            
            self.micro_step_trainer = MicroStepTrainer(model, optimizer, config)
            
        trainer = self.micro_step_trainer
        num_steps = payload.get("num_steps", 10)
        
        self.learning_active = True
        
        try:
            # Create dummy batches for demo
            import torch
            results = []
            
            for step in range(num_steps):
                if trainer.stop_event.is_set():
                    break
                    
                # Create training batch from payload or use dummy data
                batch_size = payload.get("batch_size", 4)
                seq_length = payload.get("sequence_length", 512)
                batch = {
                    'input_ids': torch.randn(batch_size, seq_length),
                    'labels': torch.randint(0, 2, (batch_size,))
                }
                
                # Monitor queue pressure and adjust
                queue_depth = self.inference_queue.metrics.queue_depth
                trainer.inference_pressure = min(queue_depth / 20.0, 1.0)
                
                # Execute training step
                step_result = await trainer.train_step(batch, step)
                results.append(step_result)
                
                self.learning_steps += 1
                
            # Get final metrics
            final_metrics = trainer.get_metrics()
            
            return {
                "status": "completed",
                "total_steps": len(results),
                "final_metrics": final_metrics,
                "step_results": results[-5:],  # Last 5 steps
                "average_loss": final_metrics["average_loss"],
                "total_yields": final_metrics["yield_count"]
            }
            
        finally:
            self.learning_active = False
            
    async def _handle_batched_inference(self, payload: Dict[str, Any]) -> Any:
        """Handle inference using advanced batch manager."""
        if not self.batch_manager:
            raise RuntimeError("Batch manager not initialized")
            
        # Initialize batched engine if needed
        if not self.batched_engine:
            self.batched_engine = BatchedInferenceEngine(
                self.inference_model,
                self.batch_manager.config
            )
            await self.batched_engine.start()
            
        # Submit to batch manager
        prompt = payload.get("prompt", "")
        max_new_tokens = payload.get("max_new_tokens", 64)
        temperature = payload.get("temperature", 1.0)
        top_p = payload.get("top_p", 1.0)
        
        result = await self.batched_engine.infer(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        self.inference_count += 1
        
        return result
            
    async def _handle_dual_inference(self, payload: Dict[str, Any]) -> str:
        """Handle inference using dual model manager with stream separation."""
        if not self.dual_model_manager:
            # Initialize dual model manager on first use
            if self.inference_model and hasattr(self.inference_model, 'model'):
                base_model = self.inference_model.model
                self.dual_model_manager = DualModelManager(
                    base_model,
                    stream_config=StreamConfig(
                        inference_stream_priority=-1,
                        learning_stream_priority=1
                    )
                )
            else:
                raise RuntimeError("No base model available for dual model manager")
                
        # Prepare inference input data
        input_shape = payload.get("input_shape", [1, 512])
        input_data = {
            "input_ids": torch.randn(*input_shape)  # TODO: Replace with actual tokenized input
        }
        
        result = await self.dual_model_manager.inference(
            input_data,
            use_graph=payload.get("use_cuda_graph", False)
        )
        
        self.inference_count += 1
        
        return f"Dual inference completed in {result['inference_time_ms']:.1f}ms on {result['stream']} stream"
        
    async def _handle_dual_learning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle learning using dual model manager with stream separation."""
        if not self.dual_model_manager:
            return {"status": "no_dual_model_manager"}
            
        import torch.optim as optim
        
        # Create optimizer if needed
        optimizer = optim.Adam(
            self.dual_model_manager.learning_model.parameters(),
            lr=payload.get("learning_rate", 1e-4)
        )
        
        num_steps = payload.get("num_steps", 10)
        self.learning_active = True
        results = []
        
        try:
            for step in range(num_steps):
                # Create training batch
                batch_size = payload.get("batch_size", 32)
                seq_length = payload.get("sequence_length", 512)
                num_classes = payload.get("num_classes", 10)
                batch = {
                    "input_ids": torch.randn(batch_size, seq_length),
                    "labels": torch.randint(0, num_classes, (batch_size,))
                }
                
                # Run learning on dedicated stream
                result = await self.dual_model_manager.learning_step(
                    batch,
                    optimizer,
                    accumulation_steps=payload.get("gradient_accumulation", 1)
                )
                
                results.append(result)
                self.learning_steps += 1
                
                # Periodic model sync
                if step % 10 == 0:
                    self.dual_model_manager.synchronize_models()
                    
            return {
                "status": "completed",
                "total_steps": len(results),
                "average_loss": sum(r["loss"] for r in results) / len(results),
                "stream_stats": self.dual_model_manager.get_stream_statistics()
            }
            
        finally:
            self.learning_active = False
        
    def _prune_cache(self):
        """Prune cache to maintain size limit."""
        if len(self.result_cache) > self.config.cache_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self.result_cache.keys(),
                key=lambda k: self.result_cache[k][1]
            )
            for key in sorted_keys[:len(sorted_keys) - self.config.cache_size]:
                del self.result_cache[key]
                
    async def submit_learning_task(self, num_steps: int = 100, step_duration: float = 0.1):
        """Submit learning task with micro-stepping."""
        return await self.inference_queue.submit_task(
            "learning_step",
            {
                "num_steps": num_steps,
                "step_duration": step_duration
            },
            priority=TaskPriority.LOW
        )
        
    async def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        queue_status = self.inference_queue.get_status()
        
        status = {
            "queue": queue_status,
            "inference_count": self.inference_count,
            "learning_steps": self.learning_steps,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.result_cache),
            "learning_active": self.learning_active,
            "learning_paused": not self.learning_pause_event.is_set()
        }
        
        # Add dual model statistics if available
        if self.dual_model_manager:
            status["dual_model"] = {
                "enabled": True,
                "memory_usage": self.dual_model_manager.get_memory_usage(),
                "stream_stats": self.dual_model_manager.get_stream_statistics()
            }
        else:
            status["dual_model"] = {"enabled": False}
            
        return status
        
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down concurrent inference manager...")
        
        # Stop learning
        self.learning_active = False
        self.learning_pause_event.set()
        
        # Stop queue
        await self.inference_queue.stop()
        
        # Clear cache
        self.result_cache.clear()
        
        logger.info("Shutdown complete")

# Integration with existing ExpertNodeServer
class ConcurrentExpertNodeServer:
    """Enhanced expert node server with concurrent inference support."""
    
    def __init__(
        self,
        node_id: str,
        available_experts: List[str],
        port: int = 8001,
        config: InferenceConfig = None
    ):
        self.node_id = node_id
        self.available_experts = available_experts
        self.port = port
        self.config = config or InferenceConfig()
        
        # Initialize concurrent manager
        self.concurrent_manager = ConcurrentInferenceManager(self.config)
        
        # Web app
        from aiohttp import web
        self.app = web.Application()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post('/inference', self.handle_inference)
        self.app.router.add_post('/batch_inference', self.handle_batch_inference)
        self.app.router.add_post('/learning/start', self.handle_start_learning)
        self.app.router.add_get('/status', self.handle_status)
        
    async def handle_inference(self, request):
        """Handle inference request with queueing."""
        try:
            data = await request.json()
            
            result = await self.concurrent_manager.submit_inference(
                prompt=data.get("prompt", ""),
                max_new_tokens=data.get("max_new_tokens", 64),
                required_experts=data.get("required_experts"),
                priority=TaskPriority.HIGH
            )
            
            return web.json_response({
                "result": result,
                "node_id": self.node_id
            })
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
            
    async def handle_batch_inference(self, request):
        """Handle batch inference request."""
        try:
            data = await request.json()
            prompts = data.get("prompts", [])
            
            # Submit all prompts
            tasks = []
            for prompt in prompts:
                task = self.concurrent_manager.submit_inference(
                    prompt=prompt,
                    max_new_tokens=data.get("max_new_tokens", 64),
                    priority=TaskPriority.HIGH
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return web.json_response({
                "results": [str(r) for r in results],
                "node_id": self.node_id
            })
            
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
            
    async def handle_start_learning(self, request):
        """Start background learning task."""
        try:
            data = await request.json()
            
            result = await self.concurrent_manager.submit_learning_task(
                num_steps=data.get("num_steps", 100),
                step_duration=data.get("step_duration", 0.1)
            )
            
            return web.json_response({
                "status": "learning_started",
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Learning error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
            
    async def handle_status(self, request):
        """Get server status."""
        status = await self.concurrent_manager.get_status()
        status["node_id"] = self.node_id
        status["available_experts"] = self.available_experts
        
        return web.json_response(status)
        
    async def start(self):
        """Start the server."""
        # Initialize concurrent manager
        await self.concurrent_manager.initialize(Path("./data"))
        
        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Concurrent expert node server started on port {self.port}")
        
# Demo
async def demo_concurrent_system():
    """Demonstrate concurrent inference and learning."""
    import aiohttp
    
    # Start server
    server = ConcurrentExpertNodeServer(
        node_id="concurrent_node1",
        available_experts=["layer0.expert0", "layer1.expert0"],
        port=8003
    )
    
    await server.start()
    
    # Simulate mixed workload
    async with aiohttp.ClientSession() as session:
        # Start learning
        async with session.post(
            "http://localhost:8003/learning/start",
            json={"num_steps": 50, "step_duration": 0.1}
        ) as resp:
            print("Learning started:", await resp.json())
            
        # Submit inference requests while learning
        inference_tasks = []
        for i in range(10):
            async with session.post(
                "http://localhost:8003/inference",
                json={"prompt": f"Query {i}", "max_new_tokens": 32}
            ) as resp:
                result = await resp.json()
                print(f"Inference {i}: {result.get('result', 'error')[:50]}...")
                
        # Check status
        async with session.get("http://localhost:8003/status") as resp:
            status = await resp.json()
            print("\nFinal status:")
            print(f"  Inference count: {status['inference_count']}")
            print(f"  Learning steps: {status['learning_steps']}")
            print(f"  Cache hits: {status['cache_hits']}")
            print(f"  Queue status: {status['queue']}")

if __name__ == "__main__":
    asyncio.run(demo_concurrent_system())