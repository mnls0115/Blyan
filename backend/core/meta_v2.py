"""
Meta V2 Block Format
====================
Enhanced metadata block format with comprehensive model versioning.
Ensures consistency across model upgrades and prevents cross-model reuse.
"""

import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class MetaV2Block:
    """Enhanced metadata block with model versioning and validation."""
    
    VERSION = "2.0"
    BLOCK_TYPE = "meta_v2"
    
    def __init__(self, model_id: str, num_hidden_layers: int, **kwargs):
        """
        Initialize meta v2 block.
        
        Args:
            model_id: Model identifier (e.g., "Qwen/Qwen3-8B")
            num_hidden_layers: Number of hidden layers
            **kwargs: Additional metadata fields
        """
        self.model_id = model_id
        self.num_hidden_layers = num_hidden_layers
        self.metadata = {
            "version": self.VERSION,
            "block_type": self.BLOCK_TYPE,
            "model_id": model_id,
            "num_hidden_layers": num_hidden_layers,
            "created_at": datetime.utcnow().isoformat(),
            **kwargs
        }
    
    def add_model_config(self, config: Dict[str, Any]) -> None:
        """Add model configuration details.
        
        Args:
            config: Model configuration dictionary
        """
        # Extract critical config fields
        self.metadata.update({
            "vocab_size": config.get("vocab_size", 0),
            "hidden_size": config.get("hidden_size", 0),
            "intermediate_size": config.get("intermediate_size", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "num_key_value_heads": config.get("num_key_value_heads", 0),
            "max_position_embeddings": config.get("max_position_embeddings", 0),
            "rope_theta": config.get("rope_theta", 10000.0),
            "dtype": config.get("torch_dtype", "bfloat16"),
            "architecture": config.get("architectures", ["unknown"])[0] if "architectures" in config else "unknown"
        })
        
        # Compute config hash for validation
        self.metadata["config_hash"] = self._compute_config_hash(config)
    
    def add_training_info(self, info: Dict[str, Any]) -> None:
        """Add training/fine-tuning information.
        
        Args:
            info: Training information dictionary
        """
        self.metadata["training"] = {
            "base_model": info.get("base_model", self.model_id),
            "training_steps": info.get("training_steps", 0),
            "training_dataset": info.get("training_dataset", "unknown"),
            "training_date": info.get("training_date", datetime.utcnow().isoformat()),
            "lora_rank": info.get("lora_rank"),
            "lora_alpha": info.get("lora_alpha"),
            "learning_rate": info.get("learning_rate"),
            "batch_size": info.get("batch_size")
        }
    
    def add_blockchain_info(self, chain_id: str, block_height: int, 
                           param_index_hash: str) -> None:
        """Add blockchain-specific information.
        
        Args:
            chain_id: Chain identifier (e.g., "A" for meta, "B" for params)
            block_height: Current blockchain height
            param_index_hash: Hash of the parameter index
        """
        self.metadata["blockchain"] = {
            "chain_id": chain_id,
            "block_height": block_height,
            "param_index_hash": param_index_hash,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of model configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Short hash of the configuration
        """
        # Select deterministic fields for hashing
        hash_fields = {
            "model_id": self.model_id,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": config.get("vocab_size"),
            "hidden_size": config.get("hidden_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "architecture": config.get("architectures", ["unknown"])[0] if "architectures" in config else "unknown"
        }
        
        # Create stable JSON representation
        config_str = json.dumps(hash_fields, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_bytes(self) -> bytes:
        """Serialize metadata to bytes for blockchain storage.
        
        Returns:
            Serialized metadata as bytes
        """
        return json.dumps(self.metadata, sort_keys=True, indent=2).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MetaV2Block':
        """Deserialize metadata from bytes.
        
        Args:
            data: Serialized metadata bytes
            
        Returns:
            MetaV2Block instance
        """
        metadata = json.loads(data.decode('utf-8'))
        
        # Create instance with basic fields
        instance = cls(
            model_id=metadata["model_id"],
            num_hidden_layers=metadata["num_hidden_layers"]
        )
        
        # Restore full metadata
        instance.metadata = metadata
        return instance
    
    def validate(self, model_id: str, num_hidden_layers: int, 
                 config_hash: Optional[str] = None) -> bool:
        """Validate metadata against expected values.
        
        Args:
            model_id: Expected model ID
            num_hidden_layers: Expected layer count
            config_hash: Expected config hash (optional)
            
        Returns:
            True if valid, False otherwise
        """
        # Check model ID
        if self.metadata.get("model_id") != model_id:
            return False
        
        # Check layer count
        if self.metadata.get("num_hidden_layers") != num_hidden_layers:
            return False
        
        # Check config hash if provided
        if config_hash and self.metadata.get("config_hash") != config_hash:
            return False
        
        # Check version
        if self.metadata.get("version", "1.0") < "2.0":
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metadata for logging.
        
        Returns:
            Summary dictionary
        """
        return {
            "model_id": self.model_id,
            "num_hidden_layers": self.num_hidden_layers,
            "version": self.metadata.get("version"),
            "config_hash": self.metadata.get("config_hash", "")[:8] + "...",
            "created_at": self.metadata.get("created_at"),
            "vocab_size": self.metadata.get("vocab_size"),
            "dtype": self.metadata.get("dtype")
        }


def create_meta_v2_block(chain, model_config: Dict[str, Any], 
                        param_index_hash: str) -> Optional[str]:
    """Create and add a meta v2 block to the blockchain.
    
    Args:
        chain: Blockchain chain instance
        model_config: Model configuration dictionary
        param_index_hash: Current parameter index hash
        
    Returns:
        Block hash if successful, None otherwise
    """
    try:
        # Extract key fields
        model_id = model_config.get("model_id", "unknown")
        num_hidden_layers = model_config.get("layers", {}).get("num_hidden_layers", 0)
        
        # Create meta v2 block
        meta_block = MetaV2Block(model_id, num_hidden_layers)
        
        # Add full config
        meta_block.add_model_config(model_config)
        
        # Add blockchain info
        block_height = len(chain._hash_index) if hasattr(chain, '_hash_index') else 0
        meta_block.add_blockchain_info(
            chain_id=chain.chain_id,
            block_height=block_height,
            param_index_hash=param_index_hash
        )
        
        # Add to blockchain
        block_data = meta_block.to_bytes()
        block_hash = chain.add_block(
            block_data,
            block_type="meta_v2"
        )
        
        import logging
        logging.info(f"Created meta_v2 block: {block_hash[:8]}...")
        logging.info(f"  Model: {model_id}")
        logging.info(f"  Layers: {num_hidden_layers}")
        logging.info(f"  Config hash: {meta_block.metadata.get('config_hash', '')[:8]}...")
        
        return block_hash
        
    except Exception as e:
        import logging
        logging.error(f"Failed to create meta_v2 block: {e}")
        return None


def validate_meta_v2_block(chain, model_id: str, num_hidden_layers: int) -> bool:
    """Validate that the blockchain has a valid meta_v2 block for the model.
    
    Args:
        chain: Blockchain chain instance
        model_id: Expected model ID
        num_hidden_layers: Expected layer count
        
    Returns:
        True if valid meta_v2 block exists, False otherwise
    """
    try:
        # Look for meta_v2 blocks by scanning all blocks
        meta_blocks = []
        block_count = len(chain._hash_index) if hasattr(chain, '_hash_index') else 0
        
        for i in range(block_count):
            block = chain.storage.get_block_by_index(i)
            if block and hasattr(block.header, 'block_type') and block.header.block_type == 'meta_v2':
                meta_blocks.append(block)
        
        if not meta_blocks:
            return False
        
        # Check the most recent meta_v2 block
        latest_block = meta_blocks[-1]
        if not latest_block or not latest_block.payload:
            return False
        
        # Parse and validate
        meta = MetaV2Block.from_bytes(latest_block.payload)
        return meta.validate(model_id, num_hidden_layers)
        
    except Exception as e:
        import logging
        logging.debug(f"Meta v2 validation failed: {e}")
        return False