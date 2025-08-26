"""
Batch Inference Support for Distributed Pipeline
Processes multiple requests together for higher throughput
"""

import asyncio
import torch
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass 
class BatchedRequest:
    """Batched inference requests."""
    requests: List[InferenceRequest]
    batch_size: int
    padded_input: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    
    @property
    def request_ids(self) -> List[str]:
        return [r.request_id for r in self.requests]

class BatchInferenceQueue:
    """
    Queue for batching inference requests.
    Collects requests and processes them in batches for efficiency.
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,  # 100ms max wait
        min_batch_size: int = 2
    ):
        """
        Initialize batch queue.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_wait_time: Maximum time to wait for batch to fill
            min_batch_size: Minimum batch size to process
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        self._queue: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
        self._batch_ready = asyncio.Event()
        self._processing = False
    
    async def add_request(self, request: InferenceRequest) -> None:
        """Add request to queue."""
        async with self._lock:
            self._queue.append(request)
            
            # Check if we should process immediately
            if len(self._queue) >= self.max_batch_size:
                self._batch_ready.set()
    
    async def get_batch(self) -> Optional[BatchedRequest]:
        """
        Get a batch of requests to process.
        
        Returns:
            BatchedRequest or None if no requests available
        """
        # Wait for batch conditions
        try:
            # Wait for either batch full or timeout
            await asyncio.wait_for(
                self._wait_for_batch(),
                timeout=self.max_wait_time
            )
        except asyncio.TimeoutError:
            # Process whatever we have if timeout
            pass
        
        async with self._lock:
            if not self._queue:
                return None
            
            # Get batch (up to max_batch_size)
            batch_requests = self._queue[:self.max_batch_size]
            self._queue = self._queue[self.max_batch_size:]
            
            # Clear event if queue is now below min size
            if len(self._queue) < self.min_batch_size:
                self._batch_ready.clear()
            
            return BatchedRequest(
                requests=batch_requests,
                batch_size=len(batch_requests)
            )
    
    async def _wait_for_batch(self) -> None:
        """Wait for batch to be ready."""
        while True:
            async with self._lock:
                # Check if we have enough for a batch
                if len(self._queue) >= self.min_batch_size:
                    return
            
            # Wait for more requests
            await asyncio.sleep(0.01)

class BatchedPipelineProcessor:
    """
    Process batched requests through pipeline stages.
    """
    
    def __init__(self, coordinator):
        """
        Initialize batch processor.
        
        Args:
            coordinator: DensePipelineCoordinator instance
        """
        self.coordinator = coordinator
        self.queue = BatchInferenceQueue()
        self._processing_task = None
    
    async def submit_request(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        request_id: Optional[str] = None
    ) -> str:
        """
        Submit request for batched processing.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            request_id: Optional request ID
            
        Returns:
            Generated response text
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}"
        
        request = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Add to queue
        await self.queue.add_request(request)
        
        # Start processing if not already running
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_batches())
        
        # Wait for result
        return await self._wait_for_result(request_id)
    
    async def _process_batches(self) -> None:
        """Process batches from queue."""
        while True:
            batch = await self.queue.get_batch()
            if batch is None:
                break
            
            try:
                await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Mark all requests as failed
                for req in batch.requests:
                    await self._store_result(req.request_id, f"Error: {str(e)}")
    
    async def _process_batch(self, batch: BatchedRequest) -> None:
        """
        Process a batch of requests.
        
        Args:
            batch: Batched requests to process
        """
        logger.info(f"Processing batch of {batch.batch_size} requests")
        
        # Tokenize all prompts together
        from transformers import AutoTokenizer
        import os
        
        model_name = os.getenv('MODEL_NAME', 'Qwen/Qwen3-8B')
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Batch tokenization with padding
        prompts = [req.prompt for req in batch.requests]
        encoded = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        batch.padded_input = encoded['input_ids']
        batch.attention_mask = encoded.get('attention_mask')
        
        # Process through pipeline stages
        hidden_states = batch.padded_input
        
        # Get pipeline nodes
        pipeline = self.coordinator.active_pipelines.get("default")
        if not pipeline:
            # Create pipeline
            nodes = self.coordinator.registry.get_healthy_nodes()
            if not nodes:
                raise RuntimeError("No healthy nodes available")
            
            # Simple round-robin assignment
            stages = []
            for i, node in enumerate(nodes):
                stages.append({
                    'node': node,
                    'stage_id': i,
                    'layer_range': [i * 8, (i + 1) * 8]  # Distribute layers
                })
            pipeline = {'stages': stages}
            self.coordinator.active_pipelines["default"] = pipeline
        
        # Process through stages with batching
        for stage_info in pipeline['stages']:
            node = stage_info['node']
            stage = stage_info
            
            # Send batched tensor to node
            hidden_states = await self._process_stage_batched(
                node, stage, hidden_states, batch.attention_mask
            )
        
        # Decode outputs for each request
        outputs = []
        for i, req in enumerate(batch.requests):
            # Extract output for this request
            if hidden_states is not None and i < hidden_states.size(0):
                output_ids = hidden_states[i]
                text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
                outputs.append(text)
            else:
                outputs.append("Error: No output generated")
        
        # Store results
        for req, output in zip(batch.requests, outputs):
            await self._store_result(req.request_id, output)
    
    async def _process_stage_batched(
        self,
        node,
        stage: Dict,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Process batched tensors through a stage."""
        import httpx
        import base64
        import io
        
        # Serialize batch
        buffer = io.BytesIO()
        torch.save({
            'tensor': hidden_states.cpu(),
            'attention_mask': attention_mask.cpu() if attention_mask is not None else None,
            'batch_size': hidden_states.size(0),
            'dtype': str(hidden_states.dtype),
            'shape': list(hidden_states.shape)
        }, buffer)
        payload = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{node.endpoint}/inference/stage",
                json={
                    "stage": stage,
                    "hidden_states": payload,
                    "serialization": "binary",
                    "is_batched": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("output")
                
                # Deserialize batch
                if result.get("serialization") == "binary":
                    buffer = io.BytesIO(base64.b64decode(output))
                    loaded = torch.load(buffer)
                    return loaded['tensor']
                
                return torch.tensor(output)
            else:
                raise RuntimeError(f"Stage processing failed: {response.text}")
    
    # Result storage (simple in-memory for now)
    _results: Dict[str, str] = {}
    _result_events: Dict[str, asyncio.Event] = {}
    
    async def _store_result(self, request_id: str, result: str) -> None:
        """Store result for request."""
        self._results[request_id] = result
        if request_id in self._result_events:
            self._result_events[request_id].set()
    
    async def _wait_for_result(self, request_id: str) -> str:
        """Wait for result of request."""
        event = asyncio.Event()
        self._result_events[request_id] = event
        
        # Wait for result with timeout
        await asyncio.wait_for(event.wait(), timeout=30.0)
        
        result = self._results.pop(request_id, "Timeout")
        self._result_events.pop(request_id, None)
        
        return result

# Export classes
__all__ = [
    'InferenceRequest',
    'BatchedRequest', 
    'BatchInferenceQueue',
    'BatchedPipelineProcessor'
]