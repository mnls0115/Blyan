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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
        # Dynamic minimum blockchain components - set after config load
        self._min_blockchain_components = None
        # Concurrency guard for one-time model loading
        self._load_lock = threading.RLock()
        self._load_cv = threading.Condition(self._load_lock)
        self._load_in_progress = False
        
        # Model adapter for architecture-specific handling
        self.adapter = None
        self.model_config = None
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
            
            # Set model metadata for validation
            from config.model_profile import MODEL_ID, LAYERS as MODEL_LAYERS
            self.param_index.set_metadata(
                model_id=MODEL_ID,
                num_hidden_layers=MODEL_LAYERS["num_hidden_layers"],
                vocab_size=MODEL_LAYERS.get("vocab_size", 0),
                hidden_size=MODEL_LAYERS.get("hidden_size", 0)
            )
            
            # Attempt rebuild if corrupted
            if self.param_index.rebuild_if_corrupted(self.param_chain):
                logger.info("   Rebuilt param_index from blockchain after corruption")
            
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
                elif base_name in ['q_norm', 'k_norm']:
                    # Self-attention norms also need self_attn prefix
                    return f"{prefix}.self_attn.{tensor_key}"
                elif base_name in ['gate_proj', 'up_proj', 'down_proj']:
                    return f"{prefix}.mlp.{tensor_key}"
                else:
                    return f"{prefix}.{tensor_key}"
        else:
            # Simple key, just append
            return f"{prefix}.{tensor_key}"

    def _translate_key(self, layer_name: str, tensor_key: str) -> str:
        """Translate blockchain (layer_name, tensor_key) to a model state_dict key using the adapter.

        Falls back to the legacy _build_tensor_key if the adapter is unavailable or throws.
        """
        try:
            if self.adapter is not None and self.model_config is not None:
                return self.adapter.translate_key(layer_name, tensor_key, self.model_config)
        except Exception as e:
            logger.debug(f"Adapter translate_key failed ({e}); falling back to default mapping")
        return self._build_tensor_key(layer_name, tensor_key)
    
    def _validate_critical_components(self, incompatible) -> None:
        """Validate that critical model components are loaded."""
        missing_keys = incompatible.missing_keys
        
        # Check for critical missing components
        critical_missing = []
        if any('model.embed_tokens.weight' in k for k in missing_keys):
            critical_missing.append('model.embed_tokens.weight')
        if any('lm_head.weight' in k for k in missing_keys):
            critical_missing.append('lm_head.weight')
        if any('model.norm.weight' in k for k in missing_keys):
            critical_missing.append('model.norm.weight')
        
        if critical_missing:
            logger.error(f"CRITICAL: Missing essential model weights: {critical_missing}")
            logger.error("Model will produce garbage output without these components!")
            raise RuntimeError(f"Missing critical model components: {critical_missing}. "
                             f"Cannot serve model without embedding, lm_head, and model_norm weights.")
        
        # Check layer count if we have model_config
        if hasattr(self, 'model_config'):
            expected_layers = self.model_config.num_hidden_layers
            # Count loaded layers from param_index
            if hasattr(self, 'param_index'):
                layer_count = sum(1 for l in self.param_index.get_all_layers() if l.startswith('layer_'))
                if layer_count != expected_layers:
                    logger.error(f"Layer count mismatch: loaded {layer_count} but model expects {expected_layers}")
                    raise RuntimeError(f"Layer count mismatch: {layer_count} != {expected_layers}")
    
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
            
            # Dynamically determine minimum components if not set
            if self._min_blockchain_components is None:
                try:
                    # Use model config to calculate required components
                    from config.model_profile import LAYERS
                    # Need all transformer layers + embedding + lm_head + norm
                    self._min_blockchain_components = LAYERS["num_hidden_layers"] + 3
                    logger.info(f"Set minimum blockchain components to {self._min_blockchain_components} based on model config")
                except Exception as e:
                    logger.warning(f"Could not determine model components dynamically: {e}")
                    # Conservative default for 8B model (36 layers + 3 components)
                    self._min_blockchain_components = 39
            
            return len(layers) >= self._min_blockchain_components
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
            # Respect offline mode strictly (no network access when offline)
            if os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true"):
                raise RuntimeError(
                    "Tokenizer cache missing and TRANSFORMERS_OFFLINE=1. "
                    "Pre-cache the tokenizer under data/tokenizer_cache or temporarily unset offline mode."
                )
            logger.info(f"Downloading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Cache it
            cache_path.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(cache_path))
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _create_empty_model_structure(self) -> torch.nn.Module:
        """Create model structure without loading weights from HF - ENFORCES BF16."""
        # Import adapter system
        from backend.model.adapters import get_adapter
        
        # Load config from cache or minimal download
        # Set TRANSFORMERS_OFFLINE=1 to prevent weight downloads
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Get appropriate adapter for this model
        self.adapter = get_adapter(config, auto_detect=True)
        logger.info(f"Using adapter: {self.adapter.__class__.__name__} for {self.model_name}")
        
        # Validate config with adapter
        if not self.adapter.validate_config(config):
            logger.warning(f"Config validation failed for {self.adapter.__class__.__name__}, using default adapter")
            from backend.model.adapters import get_adapter
            self.adapter = get_adapter("default")
        
        # Store config for validation
        self.model_config = config
        
        # Build empty model using adapter
        model = self.adapter.build_empty_model(
            config=config,
            device=self.device,
            dtype=torch.bfloat16
        )
        
        logger.info(f"Created model structure: {model.__class__.__name__}")
        logger.info(f"Architecture: {getattr(config, 'architectures', ['Unknown'])[0]}")
        logger.info(f"Number of layers: {config.num_hidden_layers}")
        logger.info(f"Model family detected: {self.adapter.__class__.__name__.replace('Adapter', '')}")
        
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
        """Try to load from fused snapshot for instant boot with model validation."""
        if os.getenv("ENABLE_FUSED_SNAPSHOT", "true").lower() != "true":
            return False
        if os.getenv("SNAPSHOT_DISABLE", "false").lower() == "true":
            return False
        
        # First validate param_index against current model
        from config.model_profile import MODEL_ID, LAYERS
        if self.param_index.invalidate_if_model_changed(MODEL_ID, LAYERS["num_hidden_layers"]):
            logger.info("Model configuration changed - snapshots invalidated")
            self._cleanup_old_snapshots()
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
                    
                    # Comprehensive metadata validation
                    if not self._validate_snapshot_metadata(metadata, current_key):
                        logger.warning(f"Snapshot metadata validation failed - regenerating")
                        # Delete invalid snapshot
                        snapshot_path.unlink()
                        metadata_path.unlink()
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
            
            # Get comprehensive model metadata
            from config.model_profile import MODEL_ID, LAYERS
            param_metadata = self.param_index.get_metadata()
            
            # Save metadata with security info and model versioning
            metadata = {
                "param_index_hash": snapshot_key,
                "snapshot_checksum": file_hash,  # For integrity verification
                "timestamp": time.time(),
                "model_id": MODEL_ID,
                "model_name": self.model_name,
                "num_hidden_layers": LAYERS["num_hidden_layers"],
                "num_layers": len(self.param_index.get_all_layers()),
                "config_hash": param_metadata.get("config_hash", ""),
                "dtype": str(self.model.dtype) if self.model else "torch.bfloat16",
                "vocab_size": LAYERS.get("vocab_size", 0),
                "created_by": "UnifiedModelManager",
                "version": "2.0",
                "production_safe": True
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Saved secure snapshot (checksum: {file_hash[:16]}...)")
            
            # Create meta_v2 block for on-chain provenance
            try:
                from backend.core.meta_v2 import create_meta_v2_block
                from config.model_profile import get_model_config
                
                if hasattr(self, 'meta_chain'):
                    block_hash = create_meta_v2_block(
                        self.meta_chain,
                        get_model_config(),
                        snapshot_key
                    )
                    if block_hash:
                        logger.info(f"   Created meta_v2 block for snapshot: {block_hash[:8]}...")
            except Exception as e:
                logger.debug(f"Could not create meta_v2 block: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")
            # Avoid repeated attempts in this session
            self._snapshot_disabled_session = True
    
    def _get_snapshot_key(self) -> Optional[str]:
        """Get snapshot key based on param_index state and model metadata."""
        try:
            # Use the enhanced index hash that includes metadata
            return self.param_index.get_index_hash()
        except:
            return None
    
    def _validate_snapshot_metadata(self, metadata: Dict[str, Any], expected_key: str) -> bool:
        """Validate snapshot metadata against current model configuration.
        
        Args:
            metadata: Snapshot metadata to validate
            expected_key: Expected param_index hash
            
        Returns:
            True if valid, False otherwise
        """
        from config.model_profile import MODEL_ID, LAYERS
        
        # Structured logging for rejection events
        rejection_event = {
            "event": "snapshot_rejected",
            "snapshot_hash": metadata.get("param_index_hash", "unknown")[:8] + "...",
            "reason": None,
            "expected": {},
            "actual": {}
        }
        
        # Check param_index hash
        if metadata.get("param_index_hash") != expected_key:
            rejection_event["reason"] = "param_index_hash_mismatch"
            rejection_event["expected"]["hash"] = expected_key[:8] + "..."
            rejection_event["actual"]["hash"] = metadata.get("param_index_hash", "")[:8] + "..."
            logger.warning(f"🚫 SNAPSHOT REJECTED: {json.dumps(rejection_event)}")
            return False
        
        # Check model ID
        if metadata.get("model_id") != MODEL_ID:
            rejection_event["reason"] = "model_id_mismatch"
            rejection_event["expected"]["model_id"] = MODEL_ID
            rejection_event["actual"]["model_id"] = metadata.get("model_id")
            logger.warning(f"🚫 SNAPSHOT REJECTED: {json.dumps(rejection_event)}")
            return False
        
        # Check layer count
        expected_layers = LAYERS["num_hidden_layers"]
        if metadata.get("num_hidden_layers") != expected_layers:
            rejection_event["reason"] = "layer_count_mismatch"
            rejection_event["expected"]["layers"] = expected_layers
            rejection_event["actual"]["layers"] = metadata.get("num_hidden_layers")
            logger.warning(f"🚫 SNAPSHOT REJECTED: {json.dumps(rejection_event)}")
            return False
        
        # Check config hash if available
        param_metadata = self.param_index.get_metadata()
        if param_metadata.get("config_hash"):
            if metadata.get("config_hash") != param_metadata["config_hash"]:
                rejection_event["reason"] = "config_hash_mismatch"
                rejection_event["expected"]["config_hash"] = param_metadata["config_hash"][:8] + "..."
                rejection_event["actual"]["config_hash"] = metadata.get("config_hash", "")[:8] + "..."
                logger.warning(f"🚫 SNAPSHOT REJECTED: {json.dumps(rejection_event)}")
                return False
        
        # Check version compatibility
        if metadata.get("version", "1.0") < "2.0":
            rejection_event["reason"] = "version_too_old"
            rejection_event["expected"]["min_version"] = "2.0"
            rejection_event["actual"]["version"] = metadata.get("version", "1.0")
            logger.warning(f"🚫 SNAPSHOT REJECTED: {json.dumps(rejection_event)}")
            return False
        
        # Log successful validation
        logger.info(f"✅ Snapshot validated: hash={expected_key[:8]}..., model={MODEL_ID}")
        return True
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove snapshots that don't match current model configuration."""
        try:
            from pathlib import Path
            snapshot_base = os.getenv("SNAPSHOT_DIR")
            snapshot_dir = Path(snapshot_base) if snapshot_base else (self.root_dir / "models" / "fused")
            
            if not snapshot_dir.exists():
                return
            
            # Find all snapshot files
            for snapshot_path in snapshot_dir.glob("*.safetensors"):
                metadata_path = snapshot_path.with_suffix('.meta.json')
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if this snapshot is for a different model
                        from config.model_profile import MODEL_ID
                        if metadata.get("model_id") != MODEL_ID:
                            logger.info(f"Removing snapshot for different model: {snapshot_path.name}")
                            snapshot_path.unlink()
                            metadata_path.unlink()
                    except Exception as e:
                        logger.debug(f"Failed to check snapshot {snapshot_path.name}: {e}")
        except Exception as e:
            logger.debug(f"Snapshot cleanup failed: {e}")
    
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
                                    # Adapter-aware translation to state_dict key
                                    final_key = self._translate_key(layer_name, key)
                                    
                                    # Diagnostic logging if enabled
                                    if os.getenv("DIAG_MODEL_LOAD"):
                                        logger.info(f"[DIAG] Mapping {layer_name}.{key} -> {final_key}")
                                    
                                    partial_state[final_key] = self._compose_layer_with_deltas(final_key, tensor)
                    
                    if partial_state:
                        incompatible = self.model.load_state_dict(partial_state, strict=False)
                        loaded_tensors += len(partial_state)
                        
                        # Reduced diagnostic logging for chunks (summarize at end)
                        if os.getenv("DIAG_MODEL_LOAD") and os.getenv("DIAG_VERBOSE_CHUNKS"):
                            # Adapter-aware filtering to highlight only critical issues
                            try:
                                if self.adapter is not None and self.model_config is not None:
                                    crit_missing, _ = self.adapter.filter_missing_keys(list(incompatible.missing_keys), self.model_config)
                                    prob_unexp, _ = self.adapter.filter_unexpected_keys(list(incompatible.unexpected_keys), self.model_config)
                                else:
                                    crit_missing = list(incompatible.missing_keys)
                                    prob_unexp = list(incompatible.unexpected_keys)
                            except Exception:
                                crit_missing = list(incompatible.missing_keys)
                                prob_unexp = list(incompatible.unexpected_keys)

                            if crit_missing or prob_unexp:
                                logger.debug(
                                    f"[DIAG] Chunk {start}-{end-1}: {len(crit_missing)} critical missing, {len(prob_unexp)} problematic unexpected"
                                )
                        elif incompatible.missing_keys and os.getenv("DIAG_VERBOSE_CHUNKS"):
                            logger.debug(f"Chunk missing keys: {len(incompatible.missing_keys)} keys")
                        
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
                                # Adapter-aware translation to state_dict key
                                final_key = self._translate_key(layer_name, key)
                                
                                # Diagnostic logging if enabled
                                if os.getenv("DIAG_MODEL_LOAD"):
                                    logger.info(f"[DIAG] Mapping {layer_name}.{key} -> {final_key}")
                                
                                state_dict[final_key] = self._compose_layer_with_deltas(final_key, tensor)
                            logger.debug(f"Loaded {layer_name} from block")
                    except Exception as e:
                        logger.warning(f"Failed to load {layer_name}: {e}")
            
            # Load state dict into model (strict=False for flexibility)
            if state_dict:
                incompatible = self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ Loaded {len(state_dict)} tensors from blockchain")
                self._loaded_from_blockchain = True
                
                # Enhanced diagnostic logging for standard loader (adapter-aware)
                if os.getenv("DIAG_MODEL_LOAD"):
                    missing_keys = list(incompatible.missing_keys)
                    unexpected_keys = list(incompatible.unexpected_keys)

                    try:
                        if self.adapter is not None and self.model_config is not None:
                            critical_missing, ignorable_missing = self.adapter.filter_missing_keys(missing_keys, self.model_config)
                            problematic_unexpected, acceptable_unexpected = self.adapter.filter_unexpected_keys(unexpected_keys, self.model_config)
                        else:
                            critical_missing, ignorable_missing = missing_keys, []
                            problematic_unexpected, acceptable_unexpected = unexpected_keys, []
                    except Exception:
                        critical_missing, ignorable_missing = missing_keys, []
                        problematic_unexpected, acceptable_unexpected = unexpected_keys, []

                    logger.info(f"[DIAG] Final load summary:")
                    logger.info(f"[DIAG]   Loaded tensors: {len(state_dict)}")
                    logger.info(f"[DIAG]   Critical missing: {len(critical_missing)} (ignored: {len(ignorable_missing)})")
                    logger.info(f"[DIAG]   Problematic unexpected: {len(problematic_unexpected)} (acceptable: {len(acceptable_unexpected)})")

                    if critical_missing:
                        logger.info(f"[DIAG] Critical missing keys:")
                        for key in critical_missing[:5]:
                            logger.info(f"[DIAG]   - {key}")
                        if len(critical_missing) > 5:
                            logger.info(f"[DIAG]   ... and {len(critical_missing) - 5} more")

                    if problematic_unexpected:
                        logger.info(f"[DIAG] Problematic unexpected keys:")
                        for key in problematic_unexpected[:5]:
                            logger.info(f"[DIAG]   - {key}")
                        if len(problematic_unexpected) > 5:
                            logger.info(f"[DIAG]   ... and {len(problematic_unexpected) - 5} more")
                
                # STRICT VALIDATION if enabled
                if os.getenv("STRICT_MODEL_LOAD", "false").lower() == "true":
                    self._validate_critical_components(incompatible)
                
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
            # Get actual layer count from model config
            num_layers = getattr(self.model_config, 'num_hidden_layers', 32) if hasattr(self, 'model_config') else 32
            expected_layers = ["embedding"] + [f"layer_{i}" for i in range(num_layers)] + ["lm_head", "model_norm"]
            # Include other_weights in check only if it exists
            if "other_weights" in available_layers:
                expected_layers.append("other_weights")
            missing_layers = set(expected_layers) - set(available_layers)
            
            # Critical components that MUST be present
            critical_missing = []
            if "embedding" not in available_layers:
                critical_missing.append("embedding")
            if "lm_head" not in available_layers:
                critical_missing.append("lm_head")
            if "model_norm" not in available_layers:
                critical_missing.append("model_norm")
            
            # Count actual layers present
            layer_count = sum(1 for l in available_layers if l.startswith("layer_"))
            
            if critical_missing and os.getenv("STRICT_MODEL_LOAD", "false").lower() == "true":
                logger.error(f"CRITICAL: Missing essential model components: {critical_missing}")
                logger.error(f"Cannot proceed without: embedding, lm_head, and model_norm")
                raise RuntimeError(f"Missing critical model components: {critical_missing}")
            elif loaded_count < 3:  # Need at least embedding, one layer, and lm_head
                logger.error(f"Insufficient layers loaded from blockchain: {loaded_count}/{len(expected_layers)}")
                logger.error(f"Missing layers: {missing_layers}")
                
                # Fallback to pretrained weights
                logger.warning("Falling back to pretrained weights due to insufficient blockchain data")
                self._load_from_local()
            else:
                logger.info(f"✅ Successfully loaded {loaded_count}/{len(expected_layers)} layers from blockchain")
                logger.info(f"   - Model layers: {layer_count}/{num_layers}")
                logger.info(f"   - Critical components: {'✓' if not critical_missing else '✗ Missing: ' + str(critical_missing)}")
                if missing_layers:
                    logger.warning(f"Missing layers (blockchain incomplete): {missing_layers}")
                    
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
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
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
            # Apply chat template if available and input is not already templated
            formatted_prompt = prompt
            try:
                if hasattr(self.tokenizer, "apply_chat_template"):
                    # Detect if prompt already contains chat markers
                    if isinstance(prompt, str) and ("<|im_start|>" not in prompt and "<|im_end|>" not in prompt):
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
            except Exception as e:
                # Non-fatal: fall back to raw prompt on any template error
                logger.debug(f"Chat template not applied: {e}")

            # Respect tokenizer/model max length (avoid hardcoding)
            try:
                max_len = int(getattr(self.tokenizer, "model_max_length", 32768) or 32768)
            except Exception:
                max_len = 32768

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
                
                # Only include top_k if provided
                if top_k is not None:
                    generate_kwargs["top_k"] = top_k
                
                # Add repetition penalty controls
                if repetition_penalty != 1.0:
                    generate_kwargs["repetition_penalty"] = repetition_penalty
                if no_repeat_ngram_size > 0:
                    generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
                
                outputs = self.model.generate(**generate_kwargs)
            
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
        top_k: Optional[int] = None,
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
            top_k,
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
