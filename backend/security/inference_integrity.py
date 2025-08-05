"""
Inference Integrity Verification System for Blyan

This module implements real-time verification mechanisms to detect:
1. Wrong expert execution (different weights)
2. Routing manipulation (wrong expert selection)
3. Output tampering during transmission
4. Runtime environment spoofing

Key principles:
- Real-time detection during inference streaming
- Cryptographic proofs with minimal overhead
- Immediate failure indication, no post-hoc verification
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np

from backend.core.param_index import ParameterIndex


@dataclass
class SecurityBeacon:
    """Security beacon transmitted during inference streaming."""
    beacon_type: str  # "header", "activation", "weight_proof", "rolling", "footer"
    timestamp: float
    request_id: str
    beacon_data: Dict[str, Any]
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceAuditContext:
    """Context for auditing a single inference request."""
    request_id: str
    audit_nonce: str
    required_experts: List[str]
    merkle_root: str
    routing_seed: int
    image_digest: str
    expected_routing: Optional[Set[str]] = None
    activation_checkpoints: List[int] = None  # Layer indices for activation checks
    rolling_hash: str = ""
    beacon_history: List[SecurityBeacon] = None
    
    def __post_init__(self):
        if self.beacon_history is None:
            self.beacon_history = []
        if self.activation_checkpoints is None:
            self.activation_checkpoints = [0, 2]  # Check layers 0 and 2


class WeightMerkleTree:
    """Merkle tree for weight verification with page-level granularity."""
    
    def __init__(self, expert_weights: Dict[str, torch.Tensor], page_size: int = 4096):
        self.page_size = page_size
        self.expert_pages: Dict[str, List[bytes]] = {}
        self.merkle_trees: Dict[str, List[str]] = {}
        self.expert_roots: Dict[str, str] = {}
        
        self._build_trees(expert_weights)
    
    def _build_trees(self, expert_weights: Dict[str, torch.Tensor]):
        """Build merkle trees for each expert's weights."""
        for expert_name, tensor in expert_weights.items():
            # Convert tensor to bytes and split into pages
            tensor_bytes = tensor.detach().cpu().numpy().tobytes()
            pages = []
            
            for i in range(0, len(tensor_bytes), self.page_size):
                page = tensor_bytes[i:i + self.page_size]
                pages.append(page)
            
            self.expert_pages[expert_name] = pages
            
            # Build merkle tree for this expert
            page_hashes = [hashlib.sha256(page).hexdigest() for page in pages]
            tree = self._build_merkle_tree(page_hashes)
            
            self.merkle_trees[expert_name] = tree
            self.expert_roots[expert_name] = tree[0] if tree else ""
    
    def _build_merkle_tree(self, leaf_hashes: List[str]) -> List[str]:
        """Build merkle tree from leaf hashes."""
        if not leaf_hashes:
            return []
        
        tree = leaf_hashes[:]
        level_size = len(leaf_hashes)
        
        while level_size > 1:
            next_level = []
            for i in range(0, level_size, 2):
                left = tree[i]
                right = tree[i + 1] if i + 1 < level_size else left
                parent = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
                next_level.append(parent)
            
            tree = next_level + tree  # Prepend new level
            level_size = len(next_level)
        
        return tree
    
    def get_merkle_proof(self, expert_name: str, page_indices: List[int]) -> Dict[str, Any]:
        """Generate merkle proof for specific pages of an expert."""
        if expert_name not in self.expert_pages:
            return {"error": f"Expert {expert_name} not found"}
        
        pages = self.expert_pages[expert_name]
        proofs = {}
        
        for page_idx in page_indices:
            if 0 <= page_idx < len(pages):
                page_hash = hashlib.sha256(pages[page_idx]).hexdigest()
                # Simplified proof - in production, include sibling path
                proofs[str(page_idx)] = {
                    "page_hash": page_hash,
                    "page_data": pages[page_idx].hex()[:32],  # First 16 bytes as hex
                }
        
        return {
            "expert_name": expert_name,
            "merkle_root": self.expert_roots[expert_name],
            "page_proofs": proofs
        }
    
    def verify_proof(self, expert_name: str, page_idx: int, page_data: bytes, proof: Dict) -> bool:
        """Verify a merkle proof for a specific page."""
        expected_hash = hashlib.sha256(page_data).hexdigest()
        provided_hash = proof.get("page_hash", "")
        return expected_hash == provided_hash


class ActivationBeaconGenerator:
    """Generates activation hash beacons during inference."""
    
    def __init__(self, audit_context: InferenceAuditContext):
        self.audit_context = audit_context
        self.projection_vectors: Dict[int, torch.Tensor] = {}
        self._generate_projections()
    
    def _generate_projections(self):
        """Generate random projection vectors for each checkpoint layer."""
        # Use audit_nonce as seed for reproducible randomness
        torch.manual_seed(int(self.audit_context.audit_nonce[:8], 16))
        
        for layer_idx in self.audit_context.activation_checkpoints:
            # Generate random projection vector (dimension should match layer output)
            # For demo, assume 768-dimensional hidden states
            self.projection_vectors[layer_idx] = torch.randn(768)
    
    def generate_beacon(self, layer_idx: int, activation_tensor: torch.Tensor) -> Optional[SecurityBeacon]:
        """Generate activation beacon for a specific layer."""
        if layer_idx not in self.projection_vectors:
            return None
        
        # Compute projected activation
        proj_vector = self.projection_vectors[layer_idx]
        
        # Handle different tensor shapes
        if activation_tensor.dim() == 3:  # [batch, seq_len, hidden]
            activation_flat = activation_tensor.mean(dim=[0, 1])  # Average over batch and sequence
        elif activation_tensor.dim() == 2:  # [seq_len, hidden]
            activation_flat = activation_tensor.mean(dim=0)  # Average over sequence
        else:
            activation_flat = activation_tensor.flatten()[:768]  # Take first 768 elements
        
        # Ensure dimensions match
        if activation_flat.size(0) != proj_vector.size(0):
            min_dim = min(activation_flat.size(0), proj_vector.size(0))
            activation_flat = activation_flat[:min_dim]
            proj_vector = proj_vector[:min_dim]
        
        # Compute dot product and quantize
        dot_product = torch.dot(activation_flat, proj_vector).item()
        quantized_value = round(dot_product, 3)  # 3 decimal places
        
        # Generate hash beacon
        beacon_data = f"{quantized_value}|{self.audit_context.audit_nonce}|{layer_idx}"
        beacon_hash = hashlib.sha256(beacon_data.encode()).hexdigest()[:16]
        
        return SecurityBeacon(
            beacon_type="activation",
            timestamp=time.time(),
            request_id=self.audit_context.request_id,
            beacon_data={
                "layer_idx": layer_idx,
                "activation_beacon": beacon_hash,
                "quantized_projection": quantized_value
            }
        )


class RoutingCanaryDetector:
    """Detects routing manipulation using canary tokens."""
    
    def __init__(self):
        self.canary_patterns = {
            # Trigger patterns that should route to specific experts
            "quantum_physics": {"layer0.expert0", "layer1.expert1"},
            "python_code": {"layer0.expert6", "layer1.expert7"},
            "french_translation": {"layer0.expert3", "layer1.expert4"}
        }
    
    def inject_canary(self, prompt: str, routing_seed: int) -> Tuple[str, Set[str]]:
        """Inject canary token into prompt and return expected routing."""
        # Use routing_seed to deterministically select canary
        canary_keys = list(self.canary_patterns.keys())
        canary_idx = routing_seed % len(canary_keys)
        canary_key = canary_keys[canary_idx]
        
        # Subtle injection that doesn't change meaning
        injected_prompt = prompt + f" (note: {canary_key})"
        expected_experts = self.canary_patterns[canary_key]
        
        return injected_prompt, expected_experts
    
    def verify_routing(self, expected_experts: Set[str], actual_experts: Set[str]) -> bool:
        """Verify that routing matches expected pattern."""
        return expected_experts.issubset(actual_experts)


class RollingOutputCommitment:
    """Maintains rolling hash commitment of output tokens."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.rolling_hash = hashlib.sha256(request_id.encode()).hexdigest()
        self.token_count = 0
    
    def update(self, token: str) -> str:
        """Update rolling hash with new token."""
        self.token_count += 1
        combined_data = f"{self.rolling_hash}|{token}|{self.token_count}"
        self.rolling_hash = hashlib.sha256(combined_data.encode()).hexdigest()[:16]
        return self.rolling_hash
    
    def generate_beacon(self) -> SecurityBeacon:
        """Generate rolling commitment beacon."""
        return SecurityBeacon(
            beacon_type="rolling",
            timestamp=time.time(),
            request_id=self.request_id,
            beacon_data={
                "rolling_hash": self.rolling_hash,
                "token_count": self.token_count
            }
        )


class InferenceIntegrityCoordinator:
    """Coordinates all integrity verification mechanisms."""
    
    def __init__(self, param_index: ParameterIndex):
        self.param_index = param_index
        self.active_audits: Dict[str, InferenceAuditContext] = {}
        self.canary_detector = RoutingCanaryDetector()
        
        # Load weight merkle trees (mock for demo)
        self.weight_trees: Dict[str, WeightMerkleTree] = {}
    
    def initialize_audit(
        self, 
        request_id: str, 
        prompt: str, 
        required_experts: List[str],
        image_digest: str = "sha256:mock_digest"
    ) -> InferenceAuditContext:
        """Initialize audit context for a new inference request."""
        
        # Generate audit parameters
        audit_nonce = secrets.token_hex(16)
        routing_seed = secrets.randbelow(10000)
        
        # Create mock merkle root (in production, derive from parameter chain)
        merkle_data = "|".join(sorted(required_experts))
        merkle_root = hashlib.sha256(merkle_data.encode()).hexdigest()
        
        # Inject routing canary
        canary_prompt, expected_routing = self.canary_detector.inject_canary(prompt, routing_seed)
        
        audit_context = InferenceAuditContext(
            request_id=request_id,
            audit_nonce=audit_nonce,
            required_experts=required_experts,
            merkle_root=merkle_root,
            routing_seed=routing_seed,
            image_digest=image_digest,
            expected_routing=expected_routing
        )
        
        self.active_audits[request_id] = audit_context
        return audit_context
    
    def generate_header_beacon(self, audit_context: InferenceAuditContext) -> SecurityBeacon:
        """Generate header beacon with all audit parameters."""
        return SecurityBeacon(
            beacon_type="header",
            timestamp=time.time(),
            request_id=audit_context.request_id,
            beacon_data={
                "merkle_root": audit_context.merkle_root,
                "image_digest": audit_context.image_digest,
                "routing_seed": audit_context.routing_seed,
                "audit_nonce": audit_context.audit_nonce,
                "required_experts": audit_context.required_experts,
                "activation_checkpoints": audit_context.activation_checkpoints
            }
        )
    
    def verify_weight_proof(self, expert_name: str, proof_data: Dict) -> Tuple[bool, str]:
        """Verify weight merkle proof from node."""
        if expert_name not in self.weight_trees:
            return False, f"No merkle tree for expert {expert_name}"
        
        tree = self.weight_trees[expert_name]
        
        # Verify each page proof
        for page_idx_str, page_proof in proof_data.get("page_proofs", {}).items():
            page_idx = int(page_idx_str)
            page_data = bytes.fromhex(page_proof["page_data"])
            
            if not tree.verify_proof(expert_name, page_idx, page_data, page_proof):
                return False, f"Weight proof failed for page {page_idx}"
        
        return True, "Weight proof verified"
    
    def analyze_beacon_stream(self, request_id: str, beacons: List[SecurityBeacon]) -> Dict[str, Any]:
        """Analyze complete beacon stream for anomalies."""
        if request_id not in self.active_audits:
            return {"error": "Unknown request ID"}
        
        audit_context = self.active_audits[request_id]
        results = {
            "request_id": request_id,
            "total_beacons": len(beacons),
            "beacon_types": {},
            "integrity_score": 1.0,
            "anomalies": [],
            "verified_components": []
        }
        
        # Count beacon types
        for beacon in beacons:
            beacon_type = beacon.beacon_type
            results["beacon_types"][beacon_type] = results["beacon_types"].get(beacon_type, 0) + 1
        
        # Check for required beacons
        required_types = {"header", "footer"}
        missing_types = required_types - set(results["beacon_types"].keys())
        if missing_types:
            results["anomalies"].append(f"Missing beacon types: {missing_types}")
            results["integrity_score"] *= 0.5
        
        # Verify routing canary (if footer beacon has expert list)
        footer_beacons = [b for b in beacons if b.beacon_type == "footer"]
        if footer_beacons and audit_context.expected_routing:
            footer_data = footer_beacons[0].beacon_data
            actual_experts = set(footer_data.get("used_experts", []))
            
            if self.canary_detector.verify_routing(audit_context.expected_routing, actual_experts):
                results["verified_components"].append("routing_canary")
            else:
                results["anomalies"].append("Routing canary mismatch")
                results["integrity_score"] *= 0.3
        
        # Check activation beacons
        activation_beacons = [b for b in beacons if b.beacon_type == "activation"]
        if len(activation_beacons) >= len(audit_context.activation_checkpoints):
            results["verified_components"].append("activation_beacons")
        else:
            results["anomalies"].append("Insufficient activation beacons")
            results["integrity_score"] *= 0.7
        
        # Assign trust level
        if results["integrity_score"] >= 0.9:
            results["trust_level"] = "HIGH"
        elif results["integrity_score"] >= 0.7:
            results["trust_level"] = "MEDIUM"
        else:
            results["trust_level"] = "LOW"
        
        return results
    
    def get_audit_summary(self, request_id: str) -> Dict[str, Any]:
        """Get summary of audit results for a completed request."""
        if request_id not in self.active_audits:
            return {"error": "Audit context not found"}
        
        audit_context = self.active_audits[request_id]
        
        return {
            "request_id": request_id,
            "audit_nonce": audit_context.audit_nonce,
            "required_experts": audit_context.required_experts,
            "expected_routing": list(audit_context.expected_routing) if audit_context.expected_routing else None,
            "beacon_count": len(audit_context.beacon_history),
            "merkle_root": audit_context.merkle_root,
            "image_digest": audit_context.image_digest
        }
    
    def cleanup_audit(self, request_id: str):
        """Clean up audit context after request completion."""
        if request_id in self.active_audits:
            del self.active_audits[request_id]


# Utility functions for integration

def create_integrity_coordinator(param_index: ParameterIndex) -> InferenceIntegrityCoordinator:
    """Factory function to create integrity coordinator."""
    return InferenceIntegrityCoordinator(param_index)

def format_beacon_for_stream(beacon: SecurityBeacon) -> str:
    """Format beacon for streaming transmission."""
    return f"BEACON:{json.dumps(beacon.to_dict())}\n"

def parse_beacon_from_stream(beacon_line: str) -> Optional[SecurityBeacon]:
    """Parse beacon from streaming line."""
    if not beacon_line.startswith("BEACON:"):
        return None
    
    try:
        beacon_data = json.loads(beacon_line[7:])  # Remove "BEACON:" prefix
        return SecurityBeacon(**beacon_data)
    except Exception:
        return None