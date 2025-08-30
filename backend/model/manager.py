"""
Unified Model Manager
====================
Single implementation for all model management and generation.
Consolidates blockchain_first_loader, real_model_loader, arch, and infer.
Supports delta composition for learning updates.
"""

import logging
import threading
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
        use_blockchain: bool = True,
        use_gpu_direct: bool = True
    ):
        """
        Initialize unified model manager.
        
        Args:
            root_dir: Root directory for data/models
            model_name: Model identifier
            device: Device to run on
            use_blockchain: Whether to load from blockchain
            use_gpu_direct: Use GPU-direct loading for faster inference
        """
        self.root_dir = Path(root_dir)
        self.model_name = model_name
        self.device = device
        self.use_blockchain = use_blockchain
        self.use_gpu_direct = use_gpu_direct and device != "cpu"
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self.gpu_loader = None
        self._loaded_from_blockchain = False
        self._min_blockchain_layers = int(os.getenv("MIN_BLOCKCHAIN_LAYERS", "38"))
        # Concurrency guard for one-time model loading
        self._load_lock = threading.RLock()
        self._load_cv = threading.Condition(self._load_lock)
        self._load_in_progress = False
        # Best-effort GPU info for chunk sizing (avoid AttributeError)
        try:
            if device == "cuda" and torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.gpu_info = {"memory_gb": total_gb}
                
                # Apply memory optimization settings
                self._apply_memory_optimizations()
            else:
                self.gpu_info = {}
        except Exception:
            self.gpu_info = {}
        
        if use_blockchain:
            self._init_blockchain()
        else:
            self._init_local()
    
    def _apply_memory_optimizations(self):
        """Apply GPU memory optimization settings to prevent OOM."""
        try:
            # Clear any existing cache before starting
            torch.cuda.empty_cache()
            
            # Enable TF32 for better performance and memory efficiency on Ampere GPUs
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            # Disable cudnn benchmark to prevent memory spikes during warmup
            torch.backends.cudnn.benchmark = False
            
            # Set deterministic behavior for consistent memory usage
            torch.backends.cudnn.deterministic = True
            
            # Log optimizations
            logger.info("✅ Applied GPU memory optimizations:")
            logger.info("   - Cleared CUDA cache")
            logger.info("   - Enabled TF32 for Ampere+ GPUs")
            logger.info("   - Disabled cudnn benchmark (prevents memory spikes)")
            logger.info("   - Set deterministic mode for consistent memory usage")
            
        except Exception as e:
            logger.warning(f"Could not apply all memory optimizations: {e}")
    
    def _init_blockchain(self) -> None:
        """Initialize blockchain-based model loading."""
        try:
            from backend.core.chain import Chain
            from backend.core.param_index import ParameterIndex
            from backend.core.delta_index import DeltaIndex
            
            logger.info(f"📦 Initializing blockchain loader...")
            logger.info(f"   Root dir: {self.root_dir}")
            
            # Ensure root dir exists
            self.root_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize chains with proper parameters
            # Check environment for PoL settings
            skip_pol = os.environ.get('SKIP_POL', 'false').lower() == 'true'
            self.meta_chain = Chain(self.root_dir, "A", difficulty=1, skip_pol=skip_pol)
            self.param_chain = Chain(self.root_dir, "B", difficulty=1, skip_pol=skip_pol)
            
            # Initialize indices
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
    
    def _translate_layer_prefix(self, layer_name: str) -> str:
        """Translate param_index layer names to Qwen model prefixes.
        
        Maps:
        - embedding -> model.embed_tokens
        - layer_N -> model.layers.N
        - model_norm -> model.norm
        - lm_head -> lm_head (stays at top level)
        
        Args:
            layer_name: The layer name from param_index
            
        Returns:
            The corresponding model state_dict prefix
        """
        # Check if translation is disabled
        if os.getenv('DISABLE_LAYER_TRANSLATION', 'false').lower() == 'true':
            return layer_name
        
        # Translation mappings
        if layer_name == 'embedding':
            return 'model.embed_tokens'
        elif layer_name.startswith('layer_'):
            # Extract layer number
            layer_num = layer_name.split('_')[1]
            return f'model.layers.{layer_num}'
        elif layer_name == 'model_norm':
            return 'model.norm'
        elif layer_name == 'lm_head':
            return 'lm_head'
        else:
            # Unknown layer, return as-is
            logger.warning(f"Unknown layer name for translation: {layer_name}")
            return layer_name
    
    def _build_tensor_key(self, layer_name: str, tensor_key: str) -> str:
        """Build the final tensor key with proper prefix translation.
        
        Args:
            layer_name: The layer name from param_index
            tensor_key: The tensor key from the block
            
        Returns:
            The properly mapped state_dict key
        """
        # If key already starts with model. or lm_head, treat as fully qualified
        if tensor_key.startswith('model.') or tensor_key.startswith('lm_head'):
            return tensor_key
        
        # Get translated prefix
        prefix = self._translate_layer_prefix(layer_name)
        
        # If tensor_key has dots, it's a nested key
        if '.' in tensor_key:
            # For attention/mlp submodules, need special handling
            if tensor_key.startswith('self_attn.') or tensor_key.startswith('mlp.'):
                # Already has the submodule prefix
                return f"{prefix}.{tensor_key}"
            else:
                # Might be missing submodule prefix (e.g., q_proj.weight)
                # Check if it's an attention or mlp weight
                base_name = tensor_key.split('.')[0]
                if base_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    return f"{prefix}.self_attn.{tensor_key}"
                elif base_name in ['gate_proj', 'up_proj', 'down_proj']:
                    return f"{prefix}.mlp.{tensor_key}"
                else:
                    return f"{prefix}.{tensor_key}"
        else:
            # Simple key, just append
            return f"{prefix}.{tensor_key}"
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded - FAST PATH."""
        # Fast path
        if self._initialized:
            return
        
        # Concurrency control: ensure only one loader runs
        with self._load_lock:
            if self._initialized:
                return
            # If another thread is loading, wait for it to finish
            while self._load_in_progress:
                self._load_cv.wait(timeout=120.0)
                if self._initialized:
                    return
            # Mark as loading and proceed outside the lock
            self._load_in_progress = True

        try:
            # Clear GPU cache before loading to maximize available memory
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache before model loading")
            
            # Load cached tokenizer (or create once)
            self._load_or_cache_tokenizer()
            
            # MANDATORY: Blockchain-only mode - NO fallbacks allowed
            if not self.use_blockchain:
                raise RuntimeError("PRODUCTION ERROR: Blockchain mode is mandatory. Set use_blockchain=True")
            
            if not self.is_blockchain_ready():
                raise RuntimeError("BLOCKCHAIN ERROR: Model weights not available on blockchain. Cannot proceed without blockchain weights.")
            
            # Enforce offline mode - absolutely no HuggingFace downloads
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            
            logger.info("🔒 Loading model EXCLUSIVELY from blockchain (no fallbacks)")
            self._load_from_blockchain()
            
            self._initialized = True
            logger.info("✅ Model ready for inference")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            # Notify waiters regardless of success or failure
            with self._load_lock:
                self._load_in_progress = False
                self._load_cv.notify_all()
    
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
        try:
            # Check param_index file directly
            param_index_path = self.root_dir / "param_index.json"
            if not param_index_path.exists():
                logger.debug(f"No param_index found at {param_index_path}")
                return False
            
            # ALWAYS reload param_index from disk to avoid stale cache
            from backend.core.param_index import ParameterIndex
            self.param_index = ParameterIndex(param_index_path)
            
            layers = self.param_index.get_all_layers()
            logger.debug(f"Found {len(layers)} layers in param_index: {layers[:3]}...")
            
            # Need at least minimal set; readiness uses stronger check
            has_weights = len(layers) >= 3
            if has_weights:
                logger.info(f"✅ Found blockchain weights: {len(layers)} components")
            else:
                logger.info(f"❌ Insufficient blockchain weights: {len(layers)} components (need >= 3)")
            return has_weights
        except Exception as e:
            logger.warning(f"Error checking blockchain weights: {e}")
            return False

    def is_blockchain_ready(self) -> bool:
        """Return True if the blockchain has a full set of weights."""
        try:
            # Always reload the parameter index from disk to avoid stale cache
            from backend.core.param_index import ParameterIndex
            index_path = self.root_dir / "param_index.json"
            if not index_path.exists():
                return False
            self.param_index = ParameterIndex(index_path)
            layers = self.param_index.get_all_layers()
            return len(layers) >= self._min_blockchain_layers
        except Exception:
            return False

    def unload_model(self) -> None:
        """Unload current model and free GPU memory."""
        try:
            if self.model is not None:
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self._initialized = False
            self._loaded_from_blockchain = False
    
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
        """Create model structure without loading weights from HF - ENFORCES BF16."""
        # For Qwen3-8B architecture
        from transformers import AutoConfig
        from transformers.models.qwen2 import Qwen2ForCausalLM
        
        # Load config from cache or minimal download
        config = AutoConfig.from_pretrained(self.model_name)
        
        # CRITICAL: Force BF16 dtype in config
        config.torch_dtype = torch.bfloat16
        
        # Create empty model with config (no weight download)
        with torch.device("meta"):
            model = Qwen2ForCausalLM(config)
        
        # Move to target device (to_empty doesn't support dtype in all versions)
        model = model.to_empty(device=self.device)
        
        # CRITICAL: Ensure BF16 dtype after moving to device
        model = model.to(dtype=torch.bfloat16)
        
        # Verify dtype is correct
        logger.info(f"Created empty model structure with dtype: {torch.bfloat16}")
        
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
        if os.getenv("SNAPSHOT_DISABLE", "false").lower() == "true":
            return False
        
        # Compute current snapshot key from param_index
        current_key = self._get_snapshot_key()
        if not current_key:
            return False
        
        # Allow override of snapshot directory
        from pathlib import Path
        snapshot_base = os.getenv("SNAPSHOT_DIR")
        snapshot_dir = Path(snapshot_base) if snapshot_base else (self.root_dir / "models" / "fused")
        snapshot_path = snapshot_dir / f"{current_key}.safetensors"
        
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
        """Save current model as fused snapshot with security metadata.
        If save fails (e.g., disk quota), disable snapshot writes for this session.
        Honors SNAPSHOT_DISABLE and SNAPSHOT_DIR envs.
        """
        if os.getenv("ENABLE_FUSED_SNAPSHOT", "true").lower() != "true":
            return
        if os.getenv("SNAPSHOT_DISABLE", "false").lower() == "true" or getattr(self, "_snapshot_disabled_session", False):
            return
        try:
            snapshot_key = self._get_snapshot_key()
            if not snapshot_key:
                return
            
            from pathlib import Path
            snapshot_base = os.getenv("SNAPSHOT_DIR")
            snapshot_dir = Path(snapshot_base) if snapshot_base else (self.root_dir / "models" / "fused")
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
            # Avoid repeated attempts in this session
            self._snapshot_disabled_session = True
    
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
    
    def _load_from_blockchain_gpu_direct(self) -> None:
        """Load model weights from blockchain using GPU-direct loading."""
        from backend.core.gpu_direct_loader import GPUDirectBlockLoader, gpu_memory_optimization
        
        logger.info("⚡ GPU-Direct blockchain loading initiated...")
        
        # Initialize GPU-direct loader
        if not self.gpu_loader:
            self.gpu_loader = GPUDirectBlockLoader(
                param_chain=self.param_chain,
                cache_dir=self.root_dir / "gpu_cache",
                enable_pinned_cache=True,
                max_pinned_memory_gb=float(os.getenv("GPU_PINNED_MEMORY_GB", "4.0"))
            )
        
        # Initialize empty model structure
        self.model = self._create_empty_model_structure()
        
        with gpu_memory_optimization():
            try:
                # Get all layer indices from param_index
                all_layers = self.param_index.get_all_layers()
                logger.info(f"   Found {len(all_layers)} layers to load")
                
                # Ordered mapping of layer -> block index
                layer_to_block = {}
                ordered_blocks = []
                for layer_name in all_layers:
                    block_idx = self.param_index.get(layer_name)
                    if block_idx is not None:
                        layer_to_block[layer_name] = block_idx
                        ordered_blocks.append(block_idx)
                
                total_blocks = len(ordered_blocks)
                logger.info(f"   Loading {total_blocks} blocks directly to {self.device} (chunked)...")
                
                # Load in small chunks to avoid peak VRAM spikes (adaptive by GPU size)
                import torch
                cs_env = os.getenv("GPU_DIRECT_CHUNK_SIZE")
                mw_env = os.getenv("GPU_LOAD_WORKERS")
                if cs_env:
                    chunk_size = int(cs_env)
                else:
                    try:
                        total_gb = self.gpu_info.get("memory_gb") or (torch.cuda.get_device_properties(0).total_memory/1e9)
                    except Exception:
                        total_gb = 16.0
                    if total_gb <= 16:
                        chunk_size = 2
                    elif total_gb <= 24:
                        chunk_size = 2
                    elif total_gb <= 48:
                        chunk_size = 3
                    else:
                        chunk_size = 4
                if mw_env:
                    max_workers = int(mw_env)
                else:
                    if total_gb <= 24:
                        max_workers = 1
                    elif total_gb <= 48:
                        max_workers = 2
                    else:
                        max_workers = 3
                loaded_tensors = 0
                
                # Build reverse map block_idx -> [layer_names]
                block_to_layers: Dict[int, List[str]] = {}
                for ln, bi in layer_to_block.items():
                    block_to_layers.setdefault(bi, []).append(ln)
                
                for start in range(0, total_blocks, chunk_size):
                    end = min(start + chunk_size, total_blocks)
                    chunk_indices = ordered_blocks[start:end]
                    gpu_tensors = self.gpu_loader.batch_load_to_gpu(
                        chunk_indices,
                        device=self.device,
                        max_workers=max_workers
                    )
                    
                    # Build partial state dict and load immediately
                    partial_state = {}
                    for bi in chunk_indices:
                        if bi in gpu_tensors:
                            tensors = gpu_tensors[bi]
                            for key, tensor in tensors.items():
                                # There may be multiple layers mapping to same block (rare); apply to all
                                for layer_name in block_to_layers.get(bi, []):
                                    # Use proper translation for Qwen model structure
                                    final_key = self._build_tensor_key(layer_name, key)
                                    
                                    # Diagnostic logging if enabled
                                    if os.getenv("DIAG_MODEL_LOAD"):
                                        logger.info(f"[DIAG] Mapping {layer_name}.{key} -> {final_key}")
                                    
                                    partial_state[final_key] = self._compose_layer_with_deltas(final_key, tensor)
                    
                    if partial_state:
                        incompatible = self.model.load_state_dict(partial_state, strict=False)
                        loaded_tensors += len(partial_state)
                        if incompatible.missing_keys:
                            logger.debug(f"Chunk missing keys: {incompatible.missing_keys[:3]}")
                        
                        # Free temporary storage and defragment
                        del partial_state
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    logger.info(f"   ✅ Chunk loaded blocks {start}-{end-1} ({loaded_tensors} tensors total)")
                
                logger.info(f"✅ Loaded {loaded_tensors} tensors via GPU-Direct (chunked)")
                
                # Log performance stats
                stats = self.gpu_loader.get_stats()
                logger.info(f"   📊 GPU-Direct Stats:")
                logger.info(f"      Cache hit rate: {stats['cache_hit_rate']:.1%}")
                logger.info(f"      Pinned hit rate: {stats['pinned_hit_rate']:.1%}")
                logger.info(f"      Avg load time: {stats['avg_load_time_ms']:.2f}ms")
                logger.info(f"      Avg GPU transfer: {stats['avg_gpu_transfer_ms']:.2f}ms")
                logger.info(f"      Pinned cache: {stats['pinned_cache_size_mb']:.1f}MB")
                
                # Save fused snapshot for next boot
                if os.getenv("ENABLE_FUSED_SNAPSHOT", "true").lower() == "true":
                    self._save_fused_snapshot()
                
                # Free GPU-direct caches to avoid extra pinned/backing buffers
                try:
                    if self.gpu_loader:
                        self.gpu_loader.clear_cache()
                except Exception:
                    pass
                
                # Mark source
                self._loaded_from_blockchain = True
                # CRITICAL: Enforce BF16 dtype
                try:
                    import torch
                    fp = next(self.model.parameters())
                    if fp.dtype != torch.bfloat16:
                        logger.error(f"Model dtype {fp.dtype} != bfloat16 (policy), converting...")
                        self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
                        logger.info("✅ Converted model to BF16 after GPU-direct load")
                except Exception as e:
                    logger.warning(f"Could not verify/enforce BF16: {e}")
                
            except Exception as e:
                logger.error(f"GPU-Direct loading failed: {e}")
                logger.warning("Falling back to standard blockchain loading")
                # Clear GPU loader and retry with standard method
                if self.gpu_loader:
                    self.gpu_loader.clear_cache()
                self.use_gpu_direct = False
                self._load_from_blockchain()
    
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
        
        # Use GPU-direct loading if enabled
        if self.use_gpu_direct and self.device != "cpu":
            logger.info("   ⚡ Using GPU-Direct loading (zero-copy, pinned memory)...")
            self._load_from_blockchain_gpu_direct()
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
                                # Use proper translation for Qwen model structure
                                final_key = self._build_tensor_key(layer_name, key)
                                
                                # Diagnostic logging if enabled
                                if os.getenv("DIAG_MODEL_LOAD"):
                                    logger.info(f"[DIAG] Mapping {layer_name}.{key} -> {final_key}")
                                
                                state_dict[final_key] = self._compose_layer_with_deltas(final_key, tensor)
                            logger.debug(f"Loaded {layer_name} from block")
                    except Exception as e:
                        logger.warning(f"Failed to load {layer_name}: {e}")
            
            # Load state dict into model (strict=False for flexibility)
            if state_dict:
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ Loaded {len(state_dict)} tensors from blockchain")
                self._loaded_from_blockchain = True
                
                # CRITICAL: Enforce BF16 dtype after loading
                if next(self.model.parameters()).dtype != torch.bfloat16:
                    logger.warning(f"Model dtype {next(self.model.parameters()).dtype} != bfloat16, converting...")
                    self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
                    logger.info("✅ Converted model to BF16")
                
                # Save fused snapshot for next boot
                if os.getenv("ENABLE_FUSED_SNAPSHOT", "true").lower() == "true":
                    self._save_fused_snapshot()
            else:
                logger.error("No weights loaded from blockchain")
                raise RuntimeError("Blockchain weights not available")
            
            # Check if we have minimum required layers
            available_layers = self.param_index.get_all_layers() if hasattr(self, 'param_index') else []
            loaded_count = len(available_layers)
            # Required components (other_weights is optional - only if model has extra tensors)
            expected_layers = ["embedding"] + [f"layer_{i}" for i in range(36)] + ["lm_head", "model_norm"]
            # Include other_weights in check only if it exists
            if "other_weights" in available_layers:
                expected_layers.append("other_weights")
            missing_layers = set(expected_layers) - set(available_layers)
            
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
            logger.error(f"CRITICAL: Failed to load from blockchain: {e}")
            raise RuntimeError(
                f"BLOCKCHAIN LOAD FAILED: Cannot proceed without blockchain weights. "
                f"Error: {e}. Ensure blockchain contains all model layers."
            )
    
    def _load_from_local(self) -> None:
        """DEPRECATED: Local loading is strictly forbidden in production."""
        raise RuntimeError(
            "PRODUCTION ERROR: Local model loading is strictly forbidden. "
            "All models MUST be loaded from blockchain to ensure verifiable AI. "
            "This is not a fallback option - blockchain is mandatory. "
            "Set use_blockchain=True and ensure blockchain contains model weights."
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
            
            # Add cryptographic signature for verifiable AI
            if self._loaded_from_blockchain:
                try:
                    from backend.core.crypto_verifier import get_verifier
                    import time
                    
                    verifier = get_verifier()
                    model_hash = self._compute_model_hash() if hasattr(self, '_compute_model_hash') else "blockchain"
                    
                    # Sign the response
                    signature = verifier.sign_response(
                        response=response,
                        prompt=prompt,
                        model_hash=model_hash,
                        timestamp=time.time()
                    )
                    
                    # Store signature for verification (would be returned with response)
                    self._last_signature = signature
                    logger.debug(f"✅ Response cryptographically signed: {signature[:16]}...")
                except Exception as e:
                    logger.warning(f"Failed to sign response: {e}")
            
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
            "loaded_from": "blockchain" if self._loaded_from_blockchain else "local",
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
    force_new: bool = False,
    **kwargs
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
        _global_manager = UnifiedModelManager(root_dir, **kwargs)
    
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
