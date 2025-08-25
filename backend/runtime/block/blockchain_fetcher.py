"""
Blockchain Layer Fetcher

Real implementation for fetching dense model layers from blockchain storage.
"""

import pickle
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


class BlockchainLayerFetcher:
    """Fetches layer weights from blockchain storage."""
    
    def __init__(self, chain_b, storage_path: Path = None):
        """
        Initialize fetcher with blockchain chain.
        
        Args:
            chain_b: The parameter chain containing layer blocks
            storage_path: Path to blockchain storage directory
        """
        self.chain = chain_b
        self.storage_path = storage_path or Path("./data/chain_B")
        
    def fetch_layer(self, layer_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Fetch layer from blockchain.
        
        Args:
            layer_id: Layer index
            
        Returns:
            Dict containing layer weights or None if not found
        """
        layer_name = f"layer_{layer_id}" if layer_id >= 0 else ("embedding" if layer_id == -1 else "lm_head")
        
        try:
            # Search for layer block in chain
            if hasattr(self.chain, 'get_blocks_by_type'):
                layer_blocks = self.chain.get_blocks_by_type('dense_layer')
                
                for block in layer_blocks:
                    if hasattr(block.header, 'layer_name') and block.header.layer_name == layer_name:
                        # Found the layer block, deserialize payload
                        expert_data = pickle.loads(block.payload)
                        
                        # Convert to tensors if needed
                        result = {}
                        for key, value in expert_data.items():
                            if isinstance(value, torch.Tensor):
                                result[key] = value
                            else:
                                # Convert numpy or list to tensor
                                result[key] = torch.tensor(value, dtype=torch.bfloat16)  # BF16 ONLY
                        
                        logger.info(f"Loaded layer {layer_name} from blockchain")
                        return result
            
            # Alternative: Direct file access for performance
            block_files = list(self.storage_path.glob("*.block"))
            for block_file in block_files:
                try:
                    with open(block_file, 'rb') as f:
                        block_data = pickle.load(f)
                        
                    if (hasattr(block_data, 'header') and 
                        hasattr(block_data.header, 'layer_name') and 
                        block_data.header.layer_name == layer_name):
                        
                        expert_data = pickle.loads(block_data.payload)
                        
                        result = {}
                        for key, value in expert_data.items():
                            if isinstance(value, torch.Tensor):
                                result[key] = value
                            else:
                                result[key] = torch.tensor(value, dtype=torch.bfloat16)  # BF16 ONLY
                        
                        logger.info(f"Loaded layer {layer_name} from block file")
                        return result
                        
                except Exception as e:
                    continue
            
            logger.warning(f"Layer {layer_name} not found in blockchain")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch layer {layer_name}: {e}")
            return None
    
    def compute_layer_hash(self, layer_data: Dict[str, torch.Tensor]) -> str:
        """
        Compute hash of expert weights for verification.
        
        Args:
            expert_data: Dictionary of expert tensors
            
        Returns:
            SHA256 hash hex string
        """
        hasher = hashlib.sha256()
        
        # Sort keys for consistent hashing
        for key in sorted(expert_data.keys()):
            tensor = expert_data[key]
            # Convert to bytes and update hash
            tensor_bytes = tensor.cpu().numpy().tobytes()
            hasher.update(key.encode())
            hasher.update(tensor_bytes)
        
        return hasher.hexdigest()
    
    def verify_expert(self, expert_data: Dict[str, torch.Tensor], expected_hash: str) -> bool:
        """
        Verify expert data against expected hash.
        
        Args:
            expert_data: Expert weights to verify
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if verification passes
        """
        computed_hash = self.compute_expert_hash(expert_data)
        return computed_hash == expected_hash