"""
Integration Module for Block Runtime

Provides integration points with existing MoE inference system.
"""

from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from .runtime import BlockRuntime
from .types import RequestSpec, RuntimeConfig, CacheConfig, FetchStrategy
from .config import get_feature_flags

try:
    from config.model_profile import LAYERS, MOE
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False
    # Fallback values
    LAYERS = {"num_layers": 48}
    MOE = {"num_experts": 128, "num_activated_experts": 8}

logger = logging.getLogger(__name__)


class BlockRuntimeAdapter:
    """
    Adapter to integrate BlockRuntime with existing MoE inference.
    
    This class provides a compatibility layer between the new block runtime
    and the existing inference code in backend/model/moe_infer.py.
    """
    
    def __init__(self):
        self.runtime: Optional[BlockRuntime] = None
        self.initialized = False
        self.flags = get_feature_flags()
    
    async def initialize(self, chains: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the block runtime if enabled.
        
        Args:
            chains: Dictionary of blockchain chains (e.g., {'A': meta_chain, 'B': param_chain})
        """
        if not self.flags.block_runtime_enabled:
            logger.info("Block runtime is disabled")
            return
        
        logger.info("Initializing block runtime adapter...")
        
        # Create runtime config from feature flags
        config = RuntimeConfig(
            cache_config=CacheConfig(
                memory_cache_size_mb=self.flags.block_runtime_memory_cache_mb,
                disk_cache_size_mb=self.flags.block_runtime_disk_cache_mb,
                ttl_seconds=self.flags.block_runtime_cache_ttl_seconds
            ),
            fetch_strategy=(
                FetchStrategy.HEDGED if self.flags.block_runtime_hedged_fetch
                else FetchStrategy.STANDARD
            ),
            max_concurrent_fetches=self.flags.block_runtime_max_concurrent_fetches,
            fetch_timeout_ms=self.flags.block_runtime_fetch_timeout_ms,
            hedged_delay_ms=self.flags.block_runtime_hedged_delay_ms,
            prefetch_early_layers=self.flags.block_runtime_prefetch_layers if self.flags.block_runtime_prefetch else 0,
            enable_verification=self.flags.block_runtime_enable_verification,
            enable_metrics=self.flags.block_runtime_enable_metrics
        )
        
        # Get peers from environment or config
        import os
        peers = os.environ.get("BLOCK_RUNTIME_PEERS", "").split(",")
        peers = [p.strip() for p in peers if p.strip()]
        if not peers:
            peers = ["http://localhost:8001", "http://localhost:8002"]
        
        # Initialize runtime with chains
        self.runtime = BlockRuntime(
            config=config,
            manifest_path=Path("./data/expert_manifest.json"),
            peers=peers,
            chains=chains  # Pass chains for blockchain access
        )
        
        # Warmup if configured
        if os.environ.get("BLOCK_RUNTIME_WARMUP", "true").lower() == "true":
            await self.runtime.warmup()
        
        self.initialized = True
        logger.info("Block runtime adapter initialized successfully")
    
    def should_use_block_runtime(self, session_id: str) -> bool:
        """Check if block runtime should be used for this session."""
        return self.flags.should_use_block_runtime(session_id)
    
    async def run_inference(
        self,
        prompt: str,
        model_id: str,
        layer_experts: Optional[Dict[int, List[int]]] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        session_id: Optional[str] = None,
        stream_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run inference using block runtime.
        
        This method provides a compatibility interface for existing code.
        """
        if not self.initialized or self.runtime is None:
            raise RuntimeError("Block runtime not initialized")
        
        # Generate session ID if not provided
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        # Use layer_experts if provided, otherwise generate based on model_id
        if layer_experts is None:
            layer_experts = self._get_layer_experts(model_id)
        
        # Store model_id for later reference
        self.model_id = model_id
        
        # Tokenize prompt using real tokenizer
        try:
            from transformers import AutoTokenizer
            from pathlib import Path
            
            # Try local model path first
            model_path = Path(f"./models/{model_id.replace('/', '_')}")
            if model_path.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_id}: {e}")
            # Better fallback tokenization
            import torch
            words = prompt.split()[:100]  # Limit to 100 words
            input_ids = torch.tensor([[hash(w) % 50000 for w in words]])
        
        # Create request spec
        request = RequestSpec(
            model_id=model_id,
            input_ids=input_ids,
            layer_plan=layer_experts,
            sampling={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            },
            session_id=session_id,
            max_tokens=max_tokens,
            stream=stream_callback is not None
        )
        
        # Collect generated tokens
        generated_tokens = []
        expert_usage = {}
        
        async def token_collector(token):
            """Collect tokens and optionally stream them."""
            generated_tokens.append(token.token_id)
            
            # Merge expert usage
            for expert, usage in token.expert_usage.items():
                expert_usage[expert] = expert_usage.get(expert, 0) + usage
            
            # Call original stream callback if provided
            if stream_callback:
                await stream_callback(token.token_id)
        
        # Run inference
        await self.runtime.run_inference(request, token_collector)
        
        # Return results in expected format
        return {
            "tokens": generated_tokens,
            "text": self._decode_tokens(generated_tokens),
            "expert_usage": expert_usage,
            "session_id": session_id,
            "metrics": self.runtime.get_metrics() if self.flags.block_runtime_enable_metrics else {}
        }
    
    def _decode_tokens(self, token_ids: List[int]) -> str:
        """Decode tokens to text using real tokenizer."""
        # Try to use the stored tokenizer if available
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            try:
                return self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Tokenizer decode failed: {e}")
        
        # Try to load tokenizer for the model
        if hasattr(self, 'model_id'):
            try:
                from transformers import AutoTokenizer
                from pathlib import Path
                
                model_path = Path(f"./models/{self.model_id.replace('/', '_')}")
                if model_path.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                
                self.tokenizer = tokenizer  # Cache for future use
                return tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for decoding: {e}")
        
        # Last resort fallback (should never reach in production)
        logger.error("No tokenizer available for decoding, using fallback")
        return " ".join(str(t) for t in token_ids)
    
    def _get_layer_experts(self, model_id: str) -> Dict[int, List[int]]:
        """
        Get layer to expert mapping for the model.
        
        Returns:
            Dict mapping layer index to list of expert indices
        """
        # Use model profile configuration
        if PROFILE_AVAILABLE:
            num_layers = LAYERS["num_layers"]
            num_activated = MOE["num_activated_experts"]
            num_experts = MOE["num_experts"]
        elif "Qwen3" in model_id or "30B" in model_id:
            # Qwen3-30B specific
            num_layers = 48
            num_activated = 8
            num_experts = 128
        else:
            # Default fallback
            num_layers = 24
            num_activated = 2
            num_experts = 16
        
        return {
            layer_idx: list(range(min(num_activated, num_experts)))
            for layer_idx in range(num_layers)
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics."""
        if not self.initialized or self.runtime is None:
            return {}
        
        return self.runtime.get_metrics()
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.initialized or self.runtime is None:
            return ""
        
        return self.runtime.get_prometheus_metrics()
    
    async def shutdown(self) -> None:
        """Shutdown the runtime."""
        if self.runtime:
            await self.runtime.shutdown()
            self.initialized = False


# Global adapter instance
_adapter: Optional[BlockRuntimeAdapter] = None


async def get_block_runtime_adapter(chains: Optional[Dict[str, Any]] = None) -> BlockRuntimeAdapter:
    """Get or create the global block runtime adapter.
    
    Args:
        chains: Dictionary of blockchain chains to pass to the adapter
    """
    global _adapter
    
    if _adapter is None:
        _adapter = BlockRuntimeAdapter()
        await _adapter.initialize(chains)
    
    return _adapter


def integrate_with_moe_manager(moe_manager: Any, chains: Optional[Dict[str, Any]] = None) -> None:
    """
    Integrate block runtime with existing MoEModelManager.
    
    Args:
        moe_manager: The MoE model manager instance
        chains: Dictionary of blockchain chains (e.g., {'A': meta_chain, 'B': param_chain})
    
    This function patches the MoEModelManager to use block runtime
    when enabled via feature flags.
    """
    original_generate = moe_manager.generate
    
    async def patched_generate(self, *args, **kwargs):
        """Patched generate method that uses block runtime when enabled."""
        session_id = kwargs.get("session_id")
        
        # Check if we should use block runtime
        flags = get_feature_flags()
        if flags.should_use_block_runtime(session_id):
            logger.info(f"Using block runtime for session {session_id}")
            
            # Get adapter with chains if available
            chains = getattr(self, 'chains', None)
            adapter = await get_block_runtime_adapter(chains)
            
            # Extract parameters
            prompt = args[0] if args else kwargs.get("prompt", "")
            max_tokens = kwargs.get("max_tokens", 100)
            temperature = kwargs.get("temperature", 1.0)
            top_k = kwargs.get("top_k", 50)
            top_p = kwargs.get("top_p", 0.9)
            
            # Get model_id from manager and layer experts
            model_id = getattr(self, 'model_id', 'Qwen/Qwen3-8B-FP8')
            # Store model_id in adapter for reference
            adapter.model_id = model_id
            layer_experts = adapter._get_layer_experts(model_id)
            
            # Run through block runtime
            return await adapter.run_inference(
                prompt=prompt,
                model_id=model_id,
                layer_experts=layer_experts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                session_id=session_id,
                stream_callback=kwargs.get("stream_callback")
            )
        else:
            # Use original implementation
            return await original_generate(self, *args, **kwargs)
    
    # Replace method
    moe_manager.generate = patched_generate.__get__(moe_manager, type(moe_manager))
    logger.info("Integrated block runtime with MoEModelManager")