"""
Blockchain Proof Generation for Verifiable AI Inference
========================================================
Ensures every inference response can be traced back to blockchain weights.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlockchainProof:
    """Proof that inference used blockchain-stored weights."""
    
    # Core proof components
    model_hash: str              # Hash of all model weights used
    block_references: List[str]  # List of blockchain block hashes
    layer_proofs: Dict[int, str] # Layer ID -> block hash mapping
    timestamp: float              # When inference occurred
    
    # Additional verification data
    chain_id: str                 # Which blockchain (e.g., "B" for params)
    chain_height: int             # Blockchain height at inference time
    inference_hash: str           # Hash of (prompt + response + weights)
    node_signature: Optional[str] = None  # Node's cryptographic signature
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockchainProof':
        """Create from dictionary."""
        return cls(**data)
    
    def verify_integrity(self) -> bool:
        """Verify proof integrity."""
        # Check all required fields are present
        if not all([self.model_hash, self.block_references, 
                   self.layer_proofs, self.chain_id]):
            return False
        
        # Verify inference hash matches
        computed_hash = self.compute_inference_hash()
        return computed_hash == self.inference_hash
    
    def compute_inference_hash(self) -> str:
        """Compute deterministic hash of inference components."""
        components = {
            'model_hash': self.model_hash,
            'blocks': sorted(self.block_references),
            'layers': dict(sorted(self.layer_proofs.items())),
            'chain': self.chain_id,
            'height': self.chain_height,
            'timestamp': self.timestamp
        }
        
        # Create deterministic JSON string
        json_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class BlockchainProofGenerator:
    """Generates cryptographic proofs for blockchain-based inference."""
    
    def __init__(self, chain, node_id: Optional[str] = None):
        """
        Initialize proof generator.
        
        Args:
            chain: Blockchain instance containing model weights
            node_id: Optional node identifier for signatures
        """
        self.chain = chain
        self.node_id = node_id
        self._layer_cache = {}  # Cache layer -> block hash mapping
        
    def generate_proof(
        self,
        prompt: str,
        response: str,
        layers_used: List[int],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> BlockchainProof:
        """
        Generate proof that inference used blockchain weights.
        
        Args:
            prompt: Input prompt
            response: Generated response
            layers_used: List of layer IDs used in inference
            additional_metadata: Optional extra metadata
            
        Returns:
            BlockchainProof instance
        """
        try:
            # Collect block references for all layers
            block_references = []
            layer_proofs = {}
            
            for layer_id in layers_used:
                # Get block hash for this layer
                block = self._get_layer_block(layer_id)
                if block:
                    block_hash = block.hash
                    block_references.append(block_hash)
                    layer_proofs[layer_id] = block_hash
                else:
                    logger.warning(f"No blockchain block found for layer {layer_id}")
            
            # Compute model hash (hash of all weight hashes)
            model_hash = self._compute_model_hash(block_references)
            
            # Create proof
            proof = BlockchainProof(
                model_hash=model_hash,
                block_references=block_references,
                layer_proofs=layer_proofs,
                timestamp=time.time(),
                chain_id=self.chain.chain_id if hasattr(self.chain, 'chain_id') else 'B',
                chain_height=len(self.chain.blocks) if hasattr(self.chain, 'blocks') else 0,
                inference_hash="",  # Will be computed below
                node_signature=None  # Would add cryptographic signature here
            )
            
            # Compute and set inference hash
            proof.inference_hash = self._compute_inference_hash(
                prompt, response, proof
            )
            
            # Add node signature if available
            if self.node_id:
                proof.node_signature = self._sign_proof(proof)
            
            logger.info(f"Generated blockchain proof with {len(block_references)} block references")
            return proof
            
        except Exception as e:
            logger.error(f"Failed to generate blockchain proof: {e}")
            # Return minimal proof on error
            return BlockchainProof(
                model_hash="error",
                block_references=[],
                layer_proofs={},
                timestamp=time.time(),
                chain_id="unknown",
                chain_height=0,
                inference_hash="error"
            )
    
    def _get_layer_block(self, layer_id: int):
        """Get blockchain block for a specific layer."""
        # Check cache first
        if layer_id in self._layer_cache:
            return self._layer_cache[layer_id]
        
        # Search blockchain for layer block
        try:
            # Look for blocks with layer metadata
            for block in reversed(self.chain.blocks):
                if hasattr(block, 'metadata') and block.metadata:
                    if block.metadata.get('layer') == layer_id:
                        self._layer_cache[layer_id] = block
                        return block
            
            # Alternative: search by block type
            blocks = self.chain.get_blocks_by_type('layer')
            for block in blocks:
                if hasattr(block, 'metadata') and block.metadata:
                    if block.metadata.get('layer_id') == layer_id:
                        self._layer_cache[layer_id] = block
                        return block
                        
        except Exception as e:
            logger.error(f"Error getting block for layer {layer_id}: {e}")
        
        return None
    
    def _compute_model_hash(self, block_hashes: List[str]) -> str:
        """Compute hash of all model weight blocks."""
        if not block_hashes:
            return "no_blocks"
        
        # Sort for deterministic hash
        sorted_hashes = sorted(block_hashes)
        combined = "".join(sorted_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _compute_inference_hash(
        self,
        prompt: str,
        response: str,
        proof: BlockchainProof
    ) -> str:
        """Compute hash of entire inference operation."""
        inference_data = {
            'prompt': prompt,
            'response': response,
            'model_hash': proof.model_hash,
            'timestamp': proof.timestamp,
            'chain_id': proof.chain_id
        }
        
        json_str = json.dumps(inference_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _sign_proof(self, proof: BlockchainProof) -> str:
        """
        Create cryptographic signature of proof.
        
        In production, this would use proper cryptographic signing
        with the node's private key.
        """
        # For now, create a simple HMAC-like signature
        signature_data = {
            'node_id': self.node_id,
            'proof_hash': proof.inference_hash,
            'timestamp': proof.timestamp
        }
        
        json_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(f"{self.node_id}:{json_str}".encode()).hexdigest()
    
    def verify_proof(self, proof: BlockchainProof) -> bool:
        """
        Verify a blockchain proof.
        
        Args:
            proof: BlockchainProof to verify
            
        Returns:
            True if proof is valid
        """
        try:
            # Check proof integrity
            if not proof.verify_integrity():
                logger.error("Proof integrity check failed")
                return False
            
            # Verify all referenced blocks exist on chain
            for block_hash in proof.block_references:
                if not self._verify_block_exists(block_hash):
                    logger.error(f"Block {block_hash} not found on chain")
                    return False
            
            # Verify layer mappings
            for layer_id, block_hash in proof.layer_proofs.items():
                if block_hash not in proof.block_references:
                    logger.error(f"Layer {layer_id} block not in references")
                    return False
            
            logger.info("Blockchain proof verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
    
    def _verify_block_exists(self, block_hash: str) -> bool:
        """Check if a block hash exists on the blockchain."""
        try:
            for block in self.chain.blocks:
                if block.hash == block_hash:
                    return True
        except Exception:
            pass
        return False


def create_proof_for_inference(
    chain,
    prompt: str,
    response: str,
    model_manager=None
) -> Dict[str, Any]:
    """
    Convenience function to create proof for an inference.
    
    Args:
        chain: Blockchain instance
        prompt: Input prompt
        response: Generated response
        model_manager: Optional model manager to get layer info
        
    Returns:
        Dictionary with proof data
    """
    # Determine which layers were used
    if model_manager and hasattr(model_manager, 'get_loaded_layers'):
        layers_used = model_manager.get_loaded_layers()
    else:
        # Default: assume all 32 layers for Qwen3-8B
        layers_used = list(range(32))
    
    # Generate proof
    generator = BlockchainProofGenerator(chain)
    proof = generator.generate_proof(prompt, response, layers_used)
    
    return proof.to_dict()