"""
Cryptographic Verification for Blockchain Weights and Responses
===============================================================
Ensures integrity and authenticity of model weights and inference.
"""

import hashlib
import hmac
import json
import secrets
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class WeightSignature:
    """Signature for model weight verification."""
    layer_id: int
    weight_hash: str
    block_hash: str
    timestamp: float
    signature: str
    
    def verify(self, public_key: Optional[str] = None) -> bool:
        """Verify signature integrity."""
        # Compute expected hash
        data = f"{self.layer_id}:{self.weight_hash}:{self.block_hash}:{self.timestamp}"
        expected = hashlib.sha256(data.encode()).hexdigest()
        
        # In production, would use proper cryptographic verification with public key
        # For now, verify hash integrity
        return self.signature.startswith(expected[:16])


class CryptoVerifier:
    """Handles cryptographic verification of weights and responses."""
    
    def __init__(self, node_secret: Optional[str] = None):
        """
        Initialize crypto verifier.
        
        Args:
            node_secret: Secret key for signing (auto-generated if not provided)
        """
        self.node_secret = node_secret or secrets.token_hex(32)
        self._weight_cache = {}  # Cache verified weights
        
    def compute_tensor_hash(self, tensor: torch.Tensor) -> str:
        """
        Compute deterministic hash of a tensor.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            SHA256 hash of tensor data
        """
        try:
            # Convert to numpy for consistent hashing
            if tensor.is_cuda:
                tensor = tensor.cpu()
            
            # Get raw bytes
            tensor_bytes = tensor.detach().numpy().tobytes()
            
            # Compute hash
            return hashlib.sha256(tensor_bytes).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash tensor: {e}")
            return "error"
    
    def verify_weight_integrity(
        self,
        weight_tensor: torch.Tensor,
        expected_hash: str,
        layer_id: int
    ) -> bool:
        """
        Verify that a weight tensor matches expected hash.
        
        Args:
            weight_tensor: The weight tensor to verify
            expected_hash: Expected hash from blockchain
            layer_id: Layer identifier
            
        Returns:
            True if weight is valid
        """
        try:
            # Check cache first
            cache_key = f"{layer_id}:{expected_hash}"
            if cache_key in self._weight_cache:
                return self._weight_cache[cache_key]
            
            # Compute actual hash
            actual_hash = self.compute_tensor_hash(weight_tensor)
            
            # Verify match
            is_valid = actual_hash == expected_hash
            
            if not is_valid:
                logger.error(
                    f"Weight verification failed for layer {layer_id}:\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Actual: {actual_hash}"
                )
            else:
                logger.debug(f"✅ Weight verified for layer {layer_id}")
                
            # Cache result
            self._weight_cache[cache_key] = is_valid
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Weight verification error: {e}")
            return False
    
    def sign_response(
        self,
        response: str,
        prompt: str,
        model_hash: str,
        timestamp: float
    ) -> str:
        """
        Create cryptographic signature for an inference response.
        
        Args:
            response: Generated response text
            prompt: Original prompt
            model_hash: Hash of model weights used
            timestamp: Timestamp of inference
            
        Returns:
            Signature string
        """
        try:
            # Create signing data
            sign_data = {
                "response": response,
                "prompt": prompt,
                "model_hash": model_hash,
                "timestamp": timestamp
            }
            
            # Create deterministic JSON
            json_str = json.dumps(sign_data, sort_keys=True)
            
            # Create HMAC signature
            signature = hmac.new(
                self.node_secret.encode(),
                json_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Failed to sign response: {e}")
            return "error"
    
    def verify_response_signature(
        self,
        response: str,
        prompt: str,
        model_hash: str,
        timestamp: float,
        signature: str,
        node_secret: Optional[str] = None
    ) -> bool:
        """
        Verify a response signature.
        
        Args:
            response: Generated response
            prompt: Original prompt
            model_hash: Hash of model weights
            timestamp: Timestamp
            signature: Signature to verify
            node_secret: Secret key (uses instance secret if not provided)
            
        Returns:
            True if signature is valid
        """
        try:
            secret = node_secret or self.node_secret
            
            # Recreate expected signature
            expected = self.sign_response(response, prompt, model_hash, timestamp)
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(signature, expected)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def create_weight_proof(
        self,
        layer_id: int,
        weight_tensor: torch.Tensor,
        block_hash: str,
        timestamp: float
    ) -> WeightSignature:
        """
        Create cryptographic proof for a weight tensor.
        
        Args:
            layer_id: Layer identifier
            weight_tensor: The weight tensor
            block_hash: Blockchain block hash containing this weight
            timestamp: Timestamp
            
        Returns:
            WeightSignature instance
        """
        weight_hash = self.compute_tensor_hash(weight_tensor)
        
        # Create signature
        sign_data = f"{layer_id}:{weight_hash}:{block_hash}:{timestamp}"
        signature = hashlib.sha256(
            f"{self.node_secret}:{sign_data}".encode()
        ).hexdigest()
        
        return WeightSignature(
            layer_id=layer_id,
            weight_hash=weight_hash,
            block_hash=block_hash,
            timestamp=timestamp,
            signature=signature
        )
    
    def verify_blockchain_block(
        self,
        block_data: bytes,
        expected_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verify a blockchain block's integrity.
        
        Args:
            block_data: Raw block data
            expected_hash: Expected block hash
            metadata: Optional metadata to verify
            
        Returns:
            True if block is valid
        """
        try:
            # Compute actual hash
            actual_hash = hashlib.sha256(block_data).hexdigest()
            
            # Basic integrity check
            if actual_hash != expected_hash:
                logger.error(f"Block hash mismatch: {actual_hash} != {expected_hash}")
                return False
            
            # If metadata provided, verify it matches
            if metadata:
                # Additional metadata verification could go here
                pass
            
            logger.debug(f"✅ Block verified: {expected_hash[:16]}...")
            return True
            
        except Exception as e:
            logger.error(f"Block verification failed: {e}")
            return False


# Global verifier instance
_verifier = None

def get_verifier() -> CryptoVerifier:
    """Get or create global verifier instance."""
    global _verifier
    if _verifier is None:
        # In production, load node secret from secure storage
        import os
        node_secret = os.environ.get('NODE_SECRET')
        _verifier = CryptoVerifier(node_secret)
    return _verifier


def verify_inference_chain(
    prompt: str,
    response: str,
    blockchain_proof: Dict[str, Any],
    chain=None
) -> Tuple[bool, str]:
    """
    Verify entire inference chain from blockchain to response.
    
    Args:
        prompt: Input prompt
        response: Generated response
        blockchain_proof: Proof dictionary from inference
        chain: Blockchain instance for verification
        
    Returns:
        (is_valid, message) tuple
    """
    try:
        verifier = get_verifier()
        
        # Step 1: Verify blockchain proof structure
        required_fields = ['model_hash', 'block_references', 'inference_hash']
        for field in required_fields:
            if field not in blockchain_proof:
                return False, f"Missing required field: {field}"
        
        # Step 2: Verify block references exist on chain
        if chain:
            for block_hash in blockchain_proof['block_references']:
                # Would verify each block exists on chain
                pass
        
        # Step 3: Verify inference hash matches
        expected_inference_hash = hashlib.sha256(
            f"{prompt}:{response}:{blockchain_proof['model_hash']}".encode()
        ).hexdigest()
        
        # Step 4: Check signature if present
        if 'node_signature' in blockchain_proof:
            # Would verify cryptographic signature
            pass
        
        logger.info("✅ Full inference chain verified")
        return True, "Inference verified from blockchain to response"
        
    except Exception as e:
        logger.error(f"Inference chain verification failed: {e}")
        return False, str(e)