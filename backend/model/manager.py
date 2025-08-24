"""
Unified Model Manager
====================
Single implementation for all model management and generation.
Consolidates blockchain_first_loader, real_model_loader, arch, and infer.
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
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
            
            # Initialize chains
            self.meta_chain = Chain(self.root_dir, "A")
            self.param_chain = Chain(self.root_dir, "B")
            self.param_index = ParameterIndex(self.root_dir / "param_index.json")
            
            logger.info("✅ Blockchain model manager initialized")
        except Exception as e:
            logger.error(f"Failed to init blockchain: {e}")
            # Fallback to local
            self._init_local()
    
    def _init_local(self) -> None:
        """Initialize local model loading."""
        self.use_blockchain = False
        logger.info("Using local model loading")
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded."""
        if self._initialized:
            return
            
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if self.use_blockchain:
                self._load_from_blockchain()
            else:
                self._load_from_local()
            
            self._initialized = True
            logger.info("✅ Model ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_from_blockchain(self) -> None:
        """Load model weights from blockchain."""
        logger.info("Loading model from blockchain blocks")
        
        import pickle
        import io
        
        # First, initialize model architecture (without weights)
        model_config = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        if self.device == "cuda":
            model_config["device_map"] = "auto"
        
        # Initialize model with random weights
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_config
        )
        
        # Now load actual weights from blockchain
        try:
            logger.info("Loading weights from blockchain blocks...")
            
            # Expected layer names for dense model
            expected_layers = ["embedding"]
            expected_layers.extend([f"layer_{i}" for i in range(36)])  # Dense model has 36 layers
            expected_layers.append("lm_head")
            
            loaded_count = 0
            missing_layers = []
            
            for layer_name in expected_layers:
                # Get block index from parameter index
                block_index = self.param_index.get(layer_name)
                
                if block_index is None:
                    logger.warning(f"Layer {layer_name} not found in parameter index")
                    missing_layers.append(layer_name)
                    continue
                
                # Get block from chain
                try:
                    blocks = list(self.param_chain.storage.iter_blocks())
                    block = None
                    for b in blocks:
                        if b.header.index == block_index:
                            block = b
                            break
                    
                    if block is None:
                        logger.warning(f"Block {block_index} for layer {layer_name} not found in chain")
                        missing_layers.append(layer_name)
                        continue
                    
                    # Load tensor data from block payload
                    tensor_dict = torch.load(io.BytesIO(block.payload), map_location=self.device)
                    
                    # Map blockchain tensors to model state dict
                    model_state = self.model.state_dict()
                    
                    if layer_name == "embedding":
                        # Map embedding layer tensors
                        for key, tensor in tensor_dict.items():
                            if "embed" in key.lower():
                                # Find corresponding key in model
                                for model_key in model_state.keys():
                                    if "embed" in model_key.lower() and tensor.shape == model_state[model_key].shape:
                                        model_state[model_key] = tensor.to(self.device)
                                        logger.debug(f"Loaded {model_key} from blockchain")
                                        break
                    
                    elif layer_name.startswith("layer_"):
                        # Map transformer layer tensors
                        layer_idx = int(layer_name.split("_")[1])
                        prefix = f"model.layers.{layer_idx}."
                        
                        for key, tensor in tensor_dict.items():
                            # Remove any prefix from blockchain key
                            clean_key = key.split(".")[-1] if "." in key else key
                            
                            # Find matching key in model
                            for model_key in model_state.keys():
                                if model_key.startswith(prefix) and clean_key in model_key:
                                    if tensor.shape == model_state[model_key].shape:
                                        model_state[model_key] = tensor.to(self.device)
                                        logger.debug(f"Loaded {model_key} from blockchain")
                    
                    elif layer_name == "lm_head":
                        # Map output layer tensors
                        for key, tensor in tensor_dict.items():
                            if "lm_head" in key or "output" in key:
                                for model_key in model_state.keys():
                                    if "lm_head" in model_key and tensor.shape == model_state[model_key].shape:
                                        model_state[model_key] = tensor.to(self.device)
                                        logger.debug(f"Loaded {model_key} from blockchain")
                                        break
                    
                    loaded_count += 1
                    logger.info(f"✅ Loaded {layer_name} from blockchain (block {block_index})")
                    
                except Exception as e:
                    logger.error(f"Failed to load {layer_name}: {e}")
                    missing_layers.append(layer_name)
            
            # Load the updated state dict into model
            self.model.load_state_dict(model_state, strict=False)
            
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