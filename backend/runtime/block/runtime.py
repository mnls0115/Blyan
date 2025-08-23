"""
Block Runtime Implementation

Main runtime that orchestrates expert loading, execution, and streaming.
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable, Dict, Any
import logging
from pathlib import Path

from .types import (
    RequestSpec, 
    RuntimeConfig, 
    CacheConfig,
    FetchStrategy,
    StreamToken
)
from .expert_store import ExpertStore
from .execution_engine import ExecutionEngine
from .streamer import Streamer
from .metrics import MetricsCollector
from .errors import InvalidRequestError, SessionNotFoundError

logger = logging.getLogger(__name__)


class BlockRuntime:
    """
    Unified runtime for standardized inference with expert management.
    """
    
    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        manifest_path: Optional[Path] = None,
        peers: Optional[list[str]] = None,
        chains: Optional[Dict[str, Any]] = None,
        model = None,
        tokenizer = None
    ):
        # Use default config if not provided
        self.config = config or RuntimeConfig(
            cache_config=CacheConfig(),
            fetch_strategy=FetchStrategy.STANDARD
        )
        
        # Store chains reference
        self.chains = chains or {}
        
        # Initialize components
        self.expert_store = ExpertStore(
            cache_config=self.config.cache_config,
            manifest_path=manifest_path,
            peers=peers,
            fetch_timeout_ms=self.config.fetch_timeout_ms,
            enable_verification=self.config.enable_verification,
            chain_b=self.chains.get('B'),  # Pass parameter chain
            fetch_strategy=self.config.fetch_strategy,
            max_concurrent_fetches=self.config.max_concurrent_fetches,
            hedged_delay_ms=self.config.hedged_delay_ms
        )
        
        self.execution_engine = ExecutionEngine(
            expert_store=self.expert_store,
            model=model,
            tokenizer=tokenizer
        )
        
        self.streamer = Streamer()
        
        self.metrics = MetricsCollector(
            enable_detailed=self.config.enable_metrics
        )
        
        # Session tracking
        self.sessions: Dict[str, Dict] = {}
        
        # Feature flags from config
        self.enable_verification = self.config.enable_verification
        self.enable_prefetch = self.config.prefetch_early_layers > 0
        self.enable_hedged_fetch = self.config.fetch_strategy == FetchStrategy.HEDGED
        
        logger.info(
            f"BlockRuntime initialized with config: "
            f"verification={self.enable_verification}, "
            f"prefetch={self.enable_prefetch}, "
            f"hedged={self.enable_hedged_fetch}"
        )
    
    async def run_inference(
        self,
        req: RequestSpec,
        stream_cb: Callable[[StreamToken], Awaitable[None]]
    ) -> None:
        """
        Run inference with streaming callback.
        
        This is the main entry point for inference requests.
        """
        session_id = req["session_id"]
        start_time = time.time()
        
        # Track session
        self.sessions[session_id] = {
            "request": req,
            "start_time": start_time,
            "status": "initializing"
        }
        
        # Update metrics
        self.metrics.increment("inference_requests")
        self.metrics.update_resource("active_sessions", len(self.sessions))
        
        try:
            # Validate request
            if not self._validate_request(req):
                raise InvalidRequestError("Invalid request specification")
            
            # Prefetch early layer experts if enabled
            if self.enable_prefetch:
                await self._prefetch_experts(req)
            
            # Create stream session
            stream_session = await self.streamer.create_stream(
                session_id=session_id,
                callback=stream_cb
            )
            
            self.sessions[session_id]["stream"] = stream_session
            self.sessions[session_id]["status"] = "running"
            
            first_token_time = None
            tokens_generated = 0
            
            # Generate tokens
            async for token in self.execution_engine.generate(req):
                # Record first token latency
                if first_token_time is None:
                    first_token_time = time.time()
                    first_token_latency_ms = (first_token_time - start_time) * 1000
                    self.metrics.record_latency("first_token_latency_ms", first_token_latency_ms)
                    logger.info(f"First token latency: {first_token_latency_ms:.2f}ms")
                
                # Stream token
                await stream_session.send_token(token)
                
                tokens_generated += 1
                self.metrics.increment("tokens_generated")
                
                # Update session info
                self.sessions[session_id]["tokens_generated"] = tokens_generated
            
            # Mark as completed
            self.sessions[session_id]["status"] = "completed"
            
            # Calculate throughput
            total_time = time.time() - start_time
            if total_time > 0:
                tokens_per_second = tokens_generated / total_time
                self.metrics.record_throughput("tokens_per_second", tokens_per_second)
                logger.info(
                    f"Session {session_id} completed: "
                    f"{tokens_generated} tokens in {total_time:.2f}s "
                    f"({tokens_per_second:.1f} tok/s)"
                )
            
        except Exception as e:
            self.sessions[session_id]["status"] = "failed"
            self.sessions[session_id]["error"] = str(e)
            self.metrics.record_error(type(e).__name__)
            logger.error(f"Inference failed for session {session_id}: {e}")
            raise
            
        finally:
            # Clean up
            if session_id in self.sessions:
                # Close stream
                if "stream" in self.sessions[session_id]:
                    await self.streamer.close_stream(session_id)
                
                del self.sessions[session_id]
                self.metrics.update_resource("active_sessions", len(self.sessions))
    
    def _validate_request(self, req: RequestSpec) -> bool:
        """Validate request specification."""
        required_fields = ["model_id", "input_ids", "layer_plan", "sampling", "session_id"]
        
        for field in required_fields:
            if field not in req:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate layer plan format
        if not isinstance(req["layer_plan"], dict):
            logger.error("Invalid layer_plan format")
            return False
        
        # Validate sampling config
        sampling = req["sampling"]
        if not isinstance(sampling, dict):
            logger.error("Invalid sampling config")
            return False
        
        return True
    
    async def _prefetch_experts(self, req: RequestSpec) -> None:
        """Prefetch early layer experts."""
        if not self.enable_prefetch:
            return
        
        layer_plan = req["layer_plan"]
        early_layers = {}
        
        # Get first N layers to prefetch
        for layer_id in sorted(layer_plan.keys())[:self.config.prefetch_early_layers]:
            early_layers[layer_id] = layer_plan[layer_id]
        
        if early_layers:
            logger.info(f"Prefetching experts for layers: {list(early_layers.keys())}")
            await self.expert_store.prefetch_experts(early_layers)
    
    async def cancel(self, session_id: str) -> None:
        """Cancel an ongoing inference session."""
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        
        logger.info(f"Cancelling session {session_id}")
        
        # Cancel execution
        await self.execution_engine.cancel_session(session_id)
        
        # Cancel streaming
        if "stream" in self.sessions[session_id]:
            stream = self.sessions[session_id]["stream"]
            await stream.cancel()
        
        self.sessions[session_id]["status"] = "cancelled"
        self.metrics.increment("cancelled_sessions")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics."""
        # Combine metrics from all components
        metrics = {
            "runtime": self.metrics.get_metrics(),
            "expert_store": self.expert_store.get_metrics(),
            "execution_engine": self.execution_engine.get_metrics(),
            "streamer": self.streamer.get_metrics(),
            "sessions": {
                "active": len(self.sessions),
                "by_status": self._get_session_stats()
            }
        }
        
        return metrics
    
    def _get_session_stats(self) -> Dict[str, int]:
        """Get session statistics by status."""
        stats = {}
        for session in self.sessions.values():
            status = session.get("status", "unknown")
            stats[status] = stats.get(status, 0) + 1
        return stats
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics.export_prometheus()
    
    async def warmup(self, sample_request: Optional[RequestSpec] = None) -> None:
        """
        Warmup the runtime with a sample request.
        
        This helps pre-load models and establish connections.
        """
        logger.info("Starting runtime warmup...")
        
        if sample_request is None:
            # Create a minimal sample request
            import torch
            sample_request = RequestSpec(
                model_id="warmup_model",
                input_ids=torch.tensor([[1, 2, 3]]),
                layer_plan={0: [0]},
                sampling={"temperature": 1.0},
                session_id="warmup_session",
                max_tokens=1,
                stream=False
            )
        
        # Run a single inference
        tokens_received = []
        
        async def warmup_callback(token: StreamToken) -> None:
            tokens_received.append(token)
        
        try:
            await self.run_inference(sample_request, warmup_callback)
            logger.info(f"Warmup completed successfully, generated {len(tokens_received)} tokens")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the runtime."""
        logger.info("Shutting down BlockRuntime...")
        
        # Cancel all active sessions
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            try:
                await self.cancel(session_id)
            except Exception as e:
                logger.error(f"Error cancelling session {session_id}: {e}")
        
        logger.info("BlockRuntime shutdown complete")