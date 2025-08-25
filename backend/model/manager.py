"""
Unified Model Manager
====================
Single implementation for all model management and generation.
Consolidates blockchain_first_loader, real_model_loader, arch, and infer.
Supports delta composition for learning updates.
"""

import logging
import torch
import io
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class UnifiedModelManager:
    """Single model manager for all inference paths."""
    
    def __init__(
        self,
        root_dir: Path,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_blockchain: bool = True
    ):
        """
        Initialize unified model manager.
        
        Args:
            root_dir: Root directory for data/models
            model_name: Model identifier
            device: Device to run on
            use_blockchain: Whether to load from blockchain
        """
        self.root_dir = Path(root_dir)
        self.model_name = model_name
        self.device = device
        self.use_blockchain = use_blockchain
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
        if use_blockchain:
            self._init_blockchain()
        else:
            self._init_local()
    
    def _init_blockchain(self) -> None:
        """Initialize blockchain-based model loading."""
        try:
            from backend.core.chain import Chain
            from backend.core.param_index import ParameterIndex
            from backend.core.delta_index import DeltaIndex
            
            logger.info(f"📦 Initializing blockchain loader...")
            logger.info(f"   Root dir: {self.root_dir}")
            
            # Initialize chains with proper parameters
            self.meta_chain = Chain(self.root_dir, "A", difficulty=1, skip_pol=True)
            self.param_chain = Chain(self.root_dir, "B", difficulty=1, skip_pol=True)
            self.param_index = ParameterIndex(self.root_dir / "param_index.json")
            self.delta_index = DeltaIndex(self.root_dir / "delta_index.json")
            
            # Log what we found
            layers = self.param_index.get_all_layers()
            logger.info(f"   Found {len(layers)} layers in param_index")
            if layers:
                logger.info(f"   First 3: {layers[:3]}")
            
            logger.info("✅ Blockchain model manager initialized with delta support")
        except Exception as e:
            logger.error(f"❌ Failed to init blockchain: {e}")
            import traceback
            logger.error(f"   {traceback.format_exc()}")
            # Fallback to local
            self._init_local()
    
    def _init_local(self) -> None:
        """Initialize local model loading."""
        self.use_blockchain = False
        logger.info("Using local model loading")
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded - FAST PATH."""
        if self._initialized:
            return
            
        try:
            # Load cached tokenizer (or create once)
            self._load_or_cache_tokenizer()
            
            # Check if we have blockchain weights
            if self.use_blockchain and self._has_blockchain_weights():
                # Skip HF completely, load from blockchain
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._load_from_blockchain()
            elif not self.use_blockchain:
                self._load_from_local()
            else:
                logger.warning("No blockchain weights found, falling back to HF")
                self._load_from_local()
            
            self._initialized = True
            logger.info("✅ Model ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_lora_delta(self, base_tensor: torch.Tensor, delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply LoRA delta to base tensor.
        
        Args:
            base_tensor: Base weight tensor
            delta: LoRA delta with 'lora_A' and 'lora_B' keys
            
        Returns:
            Updated tensor
        """
        if 'lora_A' in delta and 'lora_B' in delta:
            # LoRA update: W' = W + BA where A and B are low-rank matrices
            lora_A = delta['lora_A']
            lora_B = delta['lora_B']
            scaling = delta.get('scaling', 1.0)
            
            # Apply LoRA: base + (B @ A) * scaling
            update = (lora_B @ lora_A) * scaling
            return base_tensor + update.to(base_tensor.dtype)
        else:
            # Direct delta addition
            return base_tensor + delta.get('delta', torch.zeros_like(base_tensor))
    
    def _has_blockchain_weights(self) -> bool:
        """Check if we have valid blockchain weights."""
        if not hasattr(self, 'param_index'):
            return False
        layers = self.param_index.get_all_layers()
        # Need at least embedding, one layer, and lm_head
        return len(layers) >= 3
    
    def _load_or_cache_tokenizer(self) -> None:
        """Load tokenizer from cache or download once."""
        cache_path = self.root_dir / "tokenizer_cache" / self.model_name.replace("/", "_")
        
        if cache_path.exists():
            logger.info("Loading cached tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(str(cache_path))
        else:
            logger.info(f"Downloading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Cache it
            cache_path.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(cache_path))
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _create_empty_model_structure(self) -> torch.nn.Module:
        """Create model structure without loading weights from HF."""
        # For Qwen3-8B architecture
        from transformers import AutoConfig
        from transformers.models.qwen2 import Qwen2ForCausalLM
        
        # Load config from cache or minimal download
        config = AutoConfig.from_pretrained(self.model_name)
        
        # Create empty model with config (no weight download)
        with torch.device("meta"):
            model = Qwen2ForCausalLM(config)
        
        # Move to target device
        model = model.to_empty(device=self.device)
        return model
    
    def _load_block_direct(self, block_index: int) -> Optional[Dict]:
        """Load a block directly by index - thread-safe."""
        try:
            block = self.param_chain.storage.get_block_by_index(block_index)
            if block and block.payload:
                # Try safetensors first, fall back to pickle
                if block.header.payload_type == "safetensors":
                    from safetensors.torch import load
                    return load(block.payload)
                else:
                    import io
                    return torch.load(io.BytesIO(block.payload), map_location=self.device)
        except Exception as e:
            logger.warning(f"Failed to load block {block_index}: {e}")
            return None
    
    def _try_load_fused_snapshot(self) -> bool:
        """Try to load from fused snapshot for instant boot."""
        if os.getenv("ENABLE_FUSED_SNAPSHOT", "true").lower() != "true":
            return False
        
        # Compute current snapshot key from param_index
        current_key = self._get_snapshot_key()
        if not current_key:
            return False
        
        snapshot_path = self.root_dir / "models" / "fused" / f"{current_key}.safetensors"
        
        if snapshot_path.exists():
            try:
                # Validate snapshot metadata before loading
                metadata_path = snapshot_path.with_suffix('.meta.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verify param_index hash matches
                    if metadata.get("param_index_hash") != current_key:
                        logger.warning(f"Snapshot metadata mismatch - regenerating")
                        return False
                    
                    # 🔒 Security: Verify snapshot checksum
                    if "snapshot_checksum" in metadata:
                        import hashlib
                        with open(snapshot_path, 'rb') as f:
                            actual_hash = hashlib.sha256(f.read()).hexdigest()
                        
                        if actual_hash != metadata["snapshot_checksum"]:
                            logger.error(f"❌ SECURITY: Snapshot checksum mismatch!")
                            logger.error(f"   Expected: {metadata['snapshot_checksum'][:16]}...")
                            logger.error(f"   Got: {actual_hash[:16]}...")
                            # Delete corrupted/tampered snapshot
                            snapshot_path.unlink()
                            metadata_path.unlink()
                            return False
                        logger.info(f"   ✅ Checksum verified: {actual_hash[:16]}...")
                    
                    # Check if production safe
                    if not metadata.get("production_safe", False):
                        logger.warning("   ⚠️ Snapshot not marked production safe")
                    
                    # Check timestamp for staleness (optional)
                    if "timestamp" in metadata:
                        age = time.time() - metadata["timestamp"]
                        max_age = int(os.getenv("SNAPSHOT_MAX_AGE_HOURS", "24")) * 3600
                        if age > max_age:
                            logger.info(f"Snapshot too old ({age/3600:.1f}h) - regenerating")
                            return False
                
                logger.info(f"Loading validated fused snapshot: {current_key}")
                from safetensors.torch import load_file
                state_dict = load_file(str(snapshot_path), device=str(self.device))
                
                # Create model structure and load weights
                self.model = self._create_empty_model_structure()
                self.model.load_state_dict(state_dict, strict=False)
                
                logger.info(f"✅ Loaded from fused snapshot (hash: {current_key[:8]}...)")
                return True
            except Exception as e:
                logger.warning(f"Failed to load snapshot: {e}")
        
        return False
    
    def _save_fused_snapshot(self) -> None:
        """Save current model as fused snapshot with security metadata."""
        try:
            snapshot_key = self._get_snapshot_key()
            if not snapshot_key:
                return
            
            snapshot_dir = self.root_dir / "models" / "fused"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / f"{snapshot_key}.safetensors"
            metadata_path = snapshot_path.with_suffix('.meta.json')
            
            logger.info(f"Saving fused snapshot: {snapshot_key}")
            
            # Save model weights
            from safetensors.torch import save_file
            state_dict = self.model.state_dict()
            save_file(state_dict, str(snapshot_path))
            
            # Calculate snapshot checksum for security
            import hashlib
            with open(snapshot_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Save metadata with security info
            metadata = {
                "param_index_hash": snapshot_key,
                "snapshot_checksum": file_hash,  # For integrity verification
                "timestamp": time.time(),
                "model_name": self.model_name,
                "num_layers": len(self.param_index.get_all_layers()),
                "created_by": "UnifiedModelManager",
                "version": "1.0",
                "production_safe": True
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Saved secure snapshot (checksum: {file_hash[:16]}...)")
            
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")
    
    def _get_snapshot_key(self) -> Optional[str]:
        """Get snapshot key based on param_index state."""
        try:
            import hashlib
            import json
            
            # Hash the param_index to get a unique key
            index_data = json.dumps(self.param_index.all(), sort_keys=True)
            return hashlib.sha256(index_data.encode()).hexdigest()[:16]
        except:
            return None
    
    def _compose_layer_with_deltas(self, layer_name: str, base_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compose a layer by applying all approved deltas.
        
        Args:
            layer_name: Name of the layer
            base_tensor: Base layer tensor
            
        Returns:
            Composed tensor with deltas applied
        """
        if not hasattr(self, 'delta_index'):
            return base_tensor
        
        # Get current base hash for this layer
        base_hash = self.delta_index.get_current_base(layer_name)
        if not base_hash:
            return base_tensor
        
        # Get best delta for this base
        delta_record = self.delta_index.get_best_delta(layer_name, base_hash)
        if not delta_record:
            return base_tensor
        
        try:
            # Load delta from blockchain
            delta_block = None
            for block in self.param_chain.storage.iter_blocks():
                if block.compute_hash() == delta_record.delta_hash:
                    delta_block = block
                    break
            
            if not delta_block:
                logger.warning(f"Delta block {delta_record.delta_hash[:8]}... not found")
                return base_tensor
            
            # Deserialize delta
            delta_data = torch.load(io.BytesIO(delta_block.payload), map_location=self.device)
            
            # Apply based on compression type
            if delta_record.compression == 'sparse' or delta_record.sparsity_ratio < 0.1:
                # LoRA or sparse delta
                return self._apply_lora_delta(base_tensor, delta_data)
            else:
                # Dense delta
                if 'delta' in delta_data:
                    return base_tensor + delta_data['delta'].to(base_tensor.dtype)
                else:
                    return base_tensor + delta_data.to(base_tensor.dtype)
                    
        except Exception as e:
            logger.error(f"Failed to apply delta for {layer_name}: {e}")
            return base_tensor
    
    def _load_from_blockchain(self) -> None:
        """Load model weights from blockchain - PRODUCTION OPTIMIZED."""
        import io
        import os
        from safetensors import safe_open
        from safetensors.torch import load_file
        
        logger.info("🚀 Starting optimized blockchain model load...")
        
        # Skip HuggingFace entirely if we have blockchain weights
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("   Set TRANSFORMERS_OFFLINE=1 (no HF downloads)")
        
        # Check for fused snapshot first (fastest path)
        logger.info("   📦 Checking for fused snapshot...")
        if self._try_load_fused_snapshot():
            logger.info("✅ Loaded from fused snapshot (instant boot)")
            return
        
        logger.info("   📊 Loading from blockchain blocks (parallel fetch)...")
        
        # Initialize empty model structure (no HF download)
        self.model = self._create_empty_model_structure()
        
        # Load weights from blockchain via param_index
        try:
            state_dict = {}
            
            # Get all layers from param_index (no chain scanning)
            all_layers = self.param_index.get_all_layers()
            logger.info(f"Found {len(all_layers)} layers in param_index")
            
            # Parallel block fetch with configurable workers
            from concurrent.futures import ThreadPoolExecutor
            
            # Configurable max workers based on system resources
            max_workers = int(os.getenv("BLOCK_FETCH_MAX_WORKERS", "4"))
            
            # Auto-adjust based on CPU count if set to 0
            if max_workers == 0:
                import multiprocessing
                max_workers = min(multiprocessing.cpu_count(), 8)
            
            # Clamp to reasonable range
            max_workers = max(1, min(max_workers, 16))
            
            logger.info(f"Using {max_workers} parallel workers for block fetching")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for layer_name in all_layers:
                    block_index = self.param_index.get(layer_name)
                    if block_index is not None:
                        future = executor.submit(self._load_block_direct, block_index)
                        futures[future] = layer_name
                
                # Collect results
                for future in futures:
                    layer_name = futures[future]
                    try:
                        tensor_dict = future.result(timeout=10)
                        if tensor_dict:
                            # Apply deltas if available
                            for key, tensor in tensor_dict.items():
                                state_dict[key] = self._compose_layer_with_deltas(key, tensor)
                            logger.debug(f"Loaded {layer_name} from block")
                    except Exception as e:
                        logger.warning(f"Failed to load {layer_name}: {e}")
            
            # Load state dict into model (strict=False for flexibility)
            if state_dict:
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ Loaded {len(state_dict)} tensors from blockchain")
                
                # Save fused snapshot for next boot
                if os.getenv("ENABLE_FUSED_SNAPSHOT", "true").lower() == "true":
                    self._save_fused_snapshot()
            else:
                logger.error("No weights loaded from blockchain")
                raise RuntimeError("Blockchain weights not available")
            
            # Check if we have minimum required layers
            if loaded_count < 3:  # Need at least embedding, one layer, and lm_head
                logger.error(f"Insufficient layers loaded from blockchain: {loaded_count}/{len(expected_layers)}")
                logger.error(f"Missing layers: {missing_layers}")
                
                # Fallback to pretrained weights
                logger.warning("Falling back to pretrained weights due to insufficient blockchain data")
                self._load_from_local()
            else:
                logger.info(f"✅ Successfully loaded {loaded_count}/{len(expected_layers)} layers from blockchain")
                if missing_layers:
                    logger.warning(f"Missing layers (using random weights): {missing_layers}")
                    
        except Exception as e:
            logger.error(f"Failed to load from blockchain: {e}")
            logger.warning("Falling back to pretrained weights")
            self._load_from_local()
    
    def _load_from_local(self) -> None:
        """Load model from local files."""
        logger.info(f"Loading model from local: {self.model_name}")
        
        model_config = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        if self.device == "cuda":
            model_config["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_config
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt.
        
        This is the SINGLE generation method used by all endpoints.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream tokens
            
        Returns:
            Generated text
        """
        self._ensure_model_loaded()
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32768
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> str:
        """Async wrapper for generation."""
        # Run in thread pool to avoid blocking
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.generate,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            stream
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        self._ensure_model_loaded()
        
        if not self.model:
            return {"error": "Model not loaded"}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        
        info = {
            "model_name": self.model_name,
            "parameters": param_count,
            "parameters_billions": param_count / 1e9,
            "device": self.device,
            "loaded_from": "blockchain" if self.use_blockchain else "local",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
        }
        
        if self.device == "cuda":
            info["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1024**3
        
        return info
    
    def get_available_experts(self) -> List[str]:
        """
        Get list of available experts from blockchain - OPTIMIZED.
        
        Returns:
            List of expert identifiers available in the blockchain.
        """
        available_experts = []
        
        if self.use_blockchain and self.param_chain:
            try:
                # OPTIMIZATION: Use parameter index instead of loading all blocks
                if hasattr(self, 'param_index') and self.param_index:
                    # Get all layer names from index
                    all_layers = self.param_index.get_all_layers()
                    available_experts = [layer for layer in all_layers if layer and layer != 'meta']
                    logger.info(f"Found {len(available_experts)} layers in parameter index (fast path)")
                    return available_experts
                
                # Fallback: Get blocks (but limit iteration)
                logger.warning("Parameter index not available, falling back to block iteration")
                blocks = []
                
                # Extract expert blocks
                for block in blocks:
                    if block.header.block_type == 'expert':
                        # Extract expert name from block data or metadata
                        expert_info = block.header.expert_info
                        if expert_info and 'expert_name' in expert_info:
                            expert_id = expert_info['expert_name']
                            if expert_id not in available_experts:
                                available_experts.append(expert_id)
                        elif hasattr(block.header, 'expert_name'):
                            expert_id = block.header.expert_name
                            if expert_id not in available_experts:
                                available_experts.append(expert_id)
                
                logger.info(f"Found {len(available_experts)} experts in blockchain")
                
            except Exception as e:
                logger.error(f"Error getting available experts: {e}")
        
        # If no blockchain experts, check for loaded model layers
        if not available_experts and self.model:
            try:
                # Check if model has expert structure
                if hasattr(self.model, 'config'):
                    config = self.model.config
                    if hasattr(config, 'num_experts'):
                        num_experts = config.num_experts
                        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 36
                        
                        # Generate expert list based on model structure
                        for layer_idx in range(num_layers):
                            for expert_idx in range(num_experts):
                                available_experts.append(f"layer{layer_idx}.expert{expert_idx}")
                    else:
                        # Non-MoE model - treat layers as experts
                        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 36
                        for layer_idx in range(num_layers):
                            available_experts.append(f"layer{layer_idx}")
                            
            except Exception as e:
                logger.warning(f"Could not determine expert structure: {e}")
        
        return available_experts


# Global instance management
_global_manager: Optional[UnifiedModelManager] = None


def get_model_manager(
    root_dir: Path = Path("./data"),
    force_new: bool = False
) -> UnifiedModelManager:
    """
    Get or create global model manager instance.
    
    Args:
        root_dir: Root directory for data
        force_new: Force creation of new instance
        
    Returns:
        UnifiedModelManager instance
    """
    global _global_manager
    
    if _global_manager is None or force_new:
        _global_manager = UnifiedModelManager(root_dir)
    
    return _global_manager


def create_model_manager(
    root_dir: Path = Path("./data"),
    **kwargs
) -> UnifiedModelManager:
    """
    Create new model manager instance.
    
    Args:
        root_dir: Root directory
        **kwargs: Additional arguments for UnifiedModelManager
        
    Returns:
        New UnifiedModelManager instance
    """
    return UnifiedModelManager(root_dir, **kwargs)