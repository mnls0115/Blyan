"""Chain ↔ Inference Bridge middleware for receipt generation and model resolution."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import structlog

from backend.crypto.signing import KeyPair
from backend.utils.json_canonical import dumps_canonical

logger = structlog.get_logger()


class IModelResolver(ABC):
    """Interface for model version resolution."""
    
    @abstractmethod
    async def resolve(self, request: Dict[str, Any]) -> Tuple[str, str, str]:
        """Resolve model_version, model_hash, tokenizer_hash from request.
        
        Returns:
            Tuple of (model_version, model_hash, tokenizer_hash)
        """
        pass
    
    @abstractmethod
    async def get_default_version(self) -> Tuple[str, str, str]:
        """Get default model version info."""
        pass


class ChainModelResolver(IModelResolver):
    """Resolver that queries blockchain for model metadata."""
    
    def __init__(self, chain_client=None, cache_ttl: int = 300):
        self.chain_client = chain_client
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.cache_timestamps = {}
    
    async def resolve(self, request: Dict[str, Any]) -> Tuple[str, str, str]:
        """Resolve from chain or cache."""
        model_id = request.get("model", "qwen_1_5_moe_a2_7b")
        version = request.get("model_version", "latest")
        
        cache_key = f"{model_id}:{version}"
        
        # Check cache
        if cache_key in self.cache:
            if time.time() - self.cache_timestamps[cache_key] < self.cache_ttl:
                return self.cache[cache_key]
        
        # In production, query chain via self.chain_client
        # For now, return deterministic values based on model_id
        if version == "latest":
            model_version = "v1.0.0"
        else:
            model_version = version
            
        # Generate deterministic hashes
        model_hash = hashlib.sha256(f"{model_id}:{model_version}".encode()).hexdigest()[:16]
        tokenizer_hash = hashlib.sha256(f"tokenizer:{model_id}".encode()).hexdigest()[:16]
        
        result = (model_version, model_hash, tokenizer_hash)
        
        # Cache result
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
        
        return result
    
    async def get_default_version(self) -> Tuple[str, str, str]:
        """Get default model version."""
        return await self.resolve({"model": "qwen_1_5_moe_a2_7b", "model_version": "latest"})


@dataclass
class InferenceReceipt:
    """Signed receipt for inference execution."""
    request_id: str
    request_hash: str
    model_version: str
    model_hash: str
    tokenizer_hash: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: float
    merkle_root: Optional[str] = None
    signature: Optional[str] = None
    chain_height: Optional[int] = None  # Block height at time of inference
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_canonical_json(self) -> str:
        """Get canonical JSON for signing."""
        # Exclude signature from canonical form
        data = {k: v for k, v in self.to_dict().items() 
                if k not in ['signature', 'merkle_root']}
        return dumps_canonical(data)
    
    def compute_hash(self) -> str:
        """Compute receipt hash."""
        return hashlib.sha256(self.to_canonical_json().encode()).hexdigest()


class ChainBridgeMiddleware:
    """Middleware bridging inference requests with blockchain."""
    
    def __init__(
        self,
        model_resolver: IModelResolver,
        keypair: Optional[KeyPair] = None,
        receipt_log_path: Optional[Path] = None
    ):
        self.model_resolver = model_resolver
        self.keypair = keypair or KeyPair.from_file(Path("data/keys/inference_signing.key"))
        self.receipt_log_path = receipt_log_path or Path("data/receipts/inference_receipts.jsonl")
        
        # Ensure receipt log dir exists
        self.receipt_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Receipt tracking
        self.receipts: Dict[str, InferenceReceipt] = {}
        self.receipt_hashes: Dict[str, str] = {}
        
        # Metrics
        self.total_receipts = 0
        self.total_resolution_errors = 0
    
    async def pre_inference(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-process request with model resolution.
        
        Enriches request with resolved model metadata.
        """
        try:
            # Resolve model version and hashes
            model_version, model_hash, tokenizer_hash = await self.model_resolver.resolve(request)
            
            # Enrich request
            request["_resolved"] = {
                "model_version": model_version,
                "model_hash": model_hash,
                "tokenizer_hash": tokenizer_hash,
                "resolved_at": time.time()
            }
            
            logger.info(
                "Model resolved",
                request_id=request.get("request_id"),
                model_version=model_version,
                model_hash=model_hash
            )
            
        except Exception as e:
            logger.error(f"Model resolution failed: {e}", request_id=request.get("request_id"))
            self.total_resolution_errors += 1
            
            # Fall back to defaults
            model_version, model_hash, tokenizer_hash = await self.model_resolver.get_default_version()
            request["_resolved"] = {
                "model_version": model_version,
                "model_hash": model_hash,
                "tokenizer_hash": tokenizer_hash,
                "resolved_at": time.time(),
                "fallback": True
            }
        
        return request
    
    async def post_inference(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        latency_ms: float
    ) -> Dict[str, Any]:
        """Post-process response with signed receipt.
        
        Attaches cryptographic receipt to response.
        """
        request_id = request.get("request_id") or response.get("id")
        
        # Get resolved metadata
        resolved = request.get("_resolved", {})
        
        # Compute request hash (deterministic)
        request_data = {
            "prompt": request.get("prompt", "")[:100],  # First 100 chars
            "max_tokens": request.get("max_tokens", 0),
            "temperature": request.get("temperature", 1.0)
        }
        request_hash = hashlib.sha256(
            dumps_canonical(request_data).encode()
        ).hexdigest()[:16]
        
        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Create receipt
        receipt = InferenceReceipt(
            request_id=request_id,
            request_hash=request_hash,
            model_version=resolved.get("model_version", "unknown"),
            model_hash=resolved.get("model_hash", "unknown"),
            tokenizer_hash=resolved.get("tokenizer_hash", "unknown"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            timestamp=time.time()
        )
        
        # Compute merkle root (simplified - in production would aggregate with other receipts)
        receipt.merkle_root = self._compute_merkle_root(receipt)
        
        # Sign receipt
        receipt.signature = self._sign_receipt(receipt)
        
        # Store receipt
        self.receipts[request_id] = receipt
        self.receipt_hashes[request_id] = receipt.compute_hash()
        
        # Log to file
        await self._log_receipt(receipt)
        
        # Attach to response
        response["receipt"] = receipt.to_dict()
        response["receipt_hash"] = receipt.compute_hash()
        
        self.total_receipts += 1
        
        logger.info(
            "Receipt generated",
            request_id=request_id,
            receipt_hash=receipt.compute_hash(),
            tokens=receipt.total_tokens
        )
        
        return response
    
    def _compute_merkle_root(self, receipt: InferenceReceipt) -> str:
        """Compute merkle root for receipt."""
        # In production, aggregate with recent receipts for merkle tree
        # For now, simple hash chain
        components = [
            receipt.request_hash,
            receipt.model_hash,
            receipt.tokenizer_hash,
            str(receipt.timestamp)
        ]
        
        root = components[0]
        for component in components[1:]:
            combined = root + component
            root = hashlib.sha256(combined.encode()).hexdigest()
        
        return root[:16]
    
    def _sign_receipt(self, receipt: InferenceReceipt) -> str:
        """Sign receipt with Ed25519."""
        message = receipt.to_canonical_json().encode()
        signature = self.keypair.sign(message)
        return signature.hex()
    
    async def _log_receipt(self, receipt: InferenceReceipt):
        """Append receipt to log file."""
        try:
            with open(self.receipt_log_path, 'a') as f:
                f.write(json.dumps(receipt.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log receipt: {e}")
    
    def verify_receipt(self, receipt: InferenceReceipt) -> bool:
        """Verify receipt signature."""
        try:
            message = receipt.to_canonical_json().encode()
            signature_bytes = bytes.fromhex(receipt.signature)
            return self.keypair.verify(message, signature_bytes)
        except Exception as e:
            logger.error(f"Receipt verification failed: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """Get bridge metrics."""
        return {
            "total_receipts": self.total_receipts,
            "total_resolution_errors": self.total_resolution_errors,
            "cached_receipts": len(self.receipts)
        }