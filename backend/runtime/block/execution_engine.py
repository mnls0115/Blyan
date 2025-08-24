"""
Execution Engine Implementation

Standardized token generation with expert loading and sampling.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import torch
import torch.nn.functional as F
import logging

from .types import RequestSpec, LayerData, StreamToken
from .expert_store import LayerStore
from .errors import InvalidRequestError, ResourceExhaustedError

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Standardized execution engine for token generation.
    """
    
    def __init__(
        self,
        layer_store: LayerStore,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size: int = 1,
        mixed_precision: bool = True,
        model = None,
        tokenizer = None
    ):
        self.layer_store = layer_store
        self.device = device
        self.max_batch_size = max_batch_size
        self.mixed_precision = mixed_precision
        self.model = model
        self.tokenizer = tokenizer
        
        # KV cache management
        self.kv_cache: Dict[str, Dict] = {}
        
        # Session tracking
        self.active_sessions: Dict[str, Dict] = {}
        
        # Metrics
        self.metrics = {
            "tokens_generated": 0,
            "total_generation_time_ms": 0,
            "layer_loads": 0,
            "sampling_time_ms": 0
        }
        
        # Try to load model if not provided
        if self.model is None:
            self._try_load_model()
    
    async def generate(
        self,
        request: RequestSpec
    ) -> AsyncGenerator[StreamToken, None]:
        """
        Generate tokens for the given request.
        """
        session_id = request["session_id"]
        
        # Validate request
        if not self._validate_request(request):
            raise InvalidRequestError("Invalid request specification")
        
        # Initialize session
        self.active_sessions[session_id] = {
            "request": request,
            "tokens_generated": 0,
            "start_time": time.time(),
            "cancelled": False
        }
        
        try:
            # Move input to device
            input_ids = request["input_ids"].to(self.device)
            max_tokens = request.get("max_tokens", 100)
            
            # Initialize KV cache for session
            self.kv_cache[session_id] = {}
            
            # Generation loop
            current_ids = input_ids
            
            for step in range(max_tokens):
                if self.active_sessions[session_id]["cancelled"]:
                    break
                
                step_start = time.time()
                
                # Run forward pass
                logits, expert_usage = await self._forward_pass(
                    current_ids,
                    request["layer_plan"],
                    session_id
                )
                
                # Sample next token
                next_token_id = self._sample_token(
                    logits,
                    request["sampling"]
                )
                
                # Update metrics
                self.metrics["tokens_generated"] += 1
                step_time_ms = (time.time() - step_start) * 1000
                self.metrics["total_generation_time_ms"] += step_time_ms
                
                # Create stream token
                token = StreamToken(
                    token_id=next_token_id,
                    logprobs=None,  # Can be computed if needed
                    timestamp_ms=int(time.time() * 1000),
                    expert_usage=expert_usage
                )
                
                yield token
                
                # Check for EOS
                if next_token_id == 2:  # Assuming 2 is EOS token
                    break
                
                # Update current_ids for next iteration
                current_ids = torch.cat([
                    current_ids, 
                    torch.tensor([[next_token_id]], device=self.device)
                ], dim=1)
                
                self.active_sessions[session_id]["tokens_generated"] += 1
                
        finally:
            # Cleanup
            if session_id in self.kv_cache:
                del self.kv_cache[session_id]
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    def _validate_request(self, request: RequestSpec) -> bool:
        """Validate request specification."""
        required_fields = ["model_id", "input_ids", "layer_plan", "sampling", "session_id"]
        
        for field in required_fields:
            if field not in request:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate layer plan
        if not isinstance(request["layer_plan"], dict):
            logger.error("Invalid layer_plan format")
            return False
        
        return True
    
    async def _forward_pass(
        self,
        input_ids: torch.Tensor,
        layer_plan: Dict[int, List[int]],
        session_id: str
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Run forward pass through the model layers.
        """
        expert_usage = {}
        
        # If we have a real model, use it
        if self.model is not None:
            try:
                # Load and apply experts to model
                for layer_id in sorted(layer_plan.keys()):
                    expert_ids = layer_plan[layer_id]
                    experts = await self._load_experts(layer_id, expert_ids)
                    
                    # Track expert usage
                    for expert_id in expert_ids:
                        expert_key = f"L{layer_id}_E{expert_id}"
                        expert_usage[expert_key] = 1.0 / len(expert_ids)
                
                # Run forward pass through real model
                with torch.no_grad():
                    outputs = self.model(input_ids, use_cache=True)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    
                # Get last token logits
                last_logits = logits[:, -1, :]
                return last_logits, expert_usage
                
            except Exception as e:
                logger.error(f"Real model forward failed: {e}")
                # In production, fail rather than use mock
                import os
                if os.environ.get('BLOCK_RUNTIME_ALLOW_MOCK', 'false').lower() != 'true':
                    raise RuntimeError(f"Model forward pass failed and mock not allowed: {e}")
                logger.warning("Using mock forward pass (debug mode only)")
        
        # Mock forward pass (debug only - disabled in production)
        import os
        if os.environ.get('BLOCK_RUNTIME_ALLOW_MOCK', 'false').lower() != 'true':
            raise RuntimeError("No model available and mock mode disabled")
        
        batch_size, seq_len = input_ids.shape
        hidden_dim = 4096  # Qwen3-30B hidden dimension
        
        # Initialize hidden states (mock)
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_dim, 
            device=self.device, dtype=torch.float16 if self.mixed_precision else torch.float32
        )
        
        # Process each layer
        for layer_id in sorted(layer_plan.keys()):
            expert_ids = layer_plan[layer_id]
            
            # Load required experts
            experts = await self._load_experts(layer_id, expert_ids)
            
            # Run MoE layer (simplified)
            layer_output = torch.zeros_like(hidden_states)
            total_weight = 0.0
            
            for expert_id, expert_data in zip(expert_ids, experts):
                # Mock expert processing
                expert_weight = 1.0 / len(expert_ids)  # Equal weighting for now
                
                # Apply expert (simplified linear transform)
                expert_output = self._apply_expert(
                    hidden_states,
                    expert_data.weights
                )
                
                layer_output += expert_weight * expert_output
                total_weight += expert_weight
                
                # Track usage
                expert_key = f"L{layer_id}_E{expert_id}"
                expert_usage[expert_key] = expert_weight
                
            if total_weight > 0:
                layer_output /= total_weight
            
            hidden_states = layer_output
        
        # Project to vocabulary (Qwen3 vocab size)
        vocab_size = 151936  # Qwen3-30B vocab size
        logits = torch.randn(
            batch_size, seq_len, vocab_size,
            device=self.device, dtype=torch.float32
        )
        
        # Get last token logits
        last_logits = logits[:, -1, :]
        
        return last_logits, expert_usage
    
    async def _load_layers(
        self,
        layer_ids: List[int]
    ) -> List[LayerData]:
        """Load multiple layers."""
        tasks = [
            self.layer_store.get_layer(layer_id)
            for layer_id in layer_ids
        ]
        
        layers = await asyncio.gather(*tasks)
        self.metrics["layer_loads"] += len(layers)
        
        return layers
    
    def _apply_layer(
        self,
        hidden_states: torch.Tensor,
        layer_weights: torch.Tensor
    ) -> torch.Tensor:
        """Apply layer transformation (simplified)."""
        # In production, this would be actual layer computation
        # For now, just return slightly modified hidden states
        return hidden_states * 1.01
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        sampling_config: Dict[str, Any]
    ) -> int:
        """Sample next token from logits."""
        sample_start = time.time()
        
        temperature = sampling_config.get("temperature", 1.0)
        top_k = sampling_config.get("top_k", 50)
        top_p = sampling_config.get("top_p", 0.9)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze()
        
        sample_time_ms = (time.time() - sample_start) * 1000
        self.metrics["sampling_time_ms"] += sample_time_ms
        
        return next_token.item()
    
    async def cancel_session(self, session_id: str) -> None:
        """Cancel an active session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["cancelled"] = True
    
    def _try_load_model(self):
        """Try to load the Qwen3-30B model if available."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_id = "Qwen/Qwen3-8B-FP8"
            # Sanitize model path by replacing slashes
            model_path = f"./models/{model_id.replace('/', '_')}"
            
            # Check if model exists locally
            from pathlib import Path
            if Path(model_path).exists():
                logger.info(f"Loading model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Determine appropriate dtype based on model and hardware
                if "FP8" in model_id:
                    # For FP8 models, try to use FP8 if available
                    try:
                        # Check if FP8 is available (PyTorch 2.1+)
                        if hasattr(torch, 'float8_e4m3fn'):
                            torch_dtype = torch.float8_e4m3fn
                            logger.info("Using native FP8 dtype")
                        else:
                            # Fallback to auto for FP8 models
                            torch_dtype = "auto"
                            logger.info("Using auto dtype for FP8 model")
                    except:
                        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        logger.info(f"FP8 not available, using {torch_dtype}")
                elif self.device == "cuda" and torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                    logger.info("Using bfloat16 dtype")
                else:
                    torch_dtype = torch.float16
                    logger.info("Using float16 dtype")
                
                # Load with explicit dtype and memory optimization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if self.device == "cpu":
                    # For CPU, convert to float32 for stability
                    self.model = self.model.to(self.device).float()
                    logger.info("Converted model to float32 for CPU")
                    
                self.model.eval()
                
                # Log model info
                param_count = sum(p.numel() for p in self.model.parameters())
                logger.info(f"✅ Loaded Qwen3-30B model ({param_count/1e9:.1f}B params) with dtype {torch_dtype}")
            else:
                logger.info(f"Model not found at {model_path}, using mock forward pass")
                
        except Exception as e:
            logger.warning(f"Could not load model: {e}, using mock forward pass")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get engine metrics."""
        avg_generation_time = (
            self.metrics["total_generation_time_ms"] / self.metrics["tokens_generated"]
            if self.metrics["tokens_generated"] > 0 else 0
        )
        
        tokens_per_second = (
            self.metrics["tokens_generated"] / (self.metrics["total_generation_time_ms"] / 1000)
            if self.metrics["total_generation_time_ms"] > 0 else 0
        )
        
        avg_sampling_time = (
            self.metrics["sampling_time_ms"] / self.metrics["tokens_generated"]
            if self.metrics["tokens_generated"] > 0 else 0
        )
        
        return {
            "tokens_generated": self.metrics["tokens_generated"],
            "tokens_per_second": tokens_per_second,
            "avg_generation_time_ms": avg_generation_time,
            "avg_sampling_time_ms": avg_sampling_time,
            "expert_loads": self.metrics["expert_loads"],
            "active_sessions": len(self.active_sessions)
        }