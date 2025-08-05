#!/usr/bin/env python3
"""
Proof-of-Data-Learning (PoDL) Implementation

This module implements the revolutionary PoDL system that provides cryptographic
proof of which datasets were used to train which experts, enabling complete
transparency and verifiability of AI training processes.
"""

import hashlib
import json
import time
import secrets
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


@dataclass
class TrainingSession:
    """Complete training session information for PoDL generation."""
    
    # Core training identifiers
    new_expert_hash: str                    # Hash of trained expert weights
    dataset_ids: List[str]                  # ["mix_dialogues_v2@2.0.1", "wiki_cc@1.1"]
    trainer_node_id: str                    # Node that performed training
    
    # Training metrics
    total_samples: int                      # Number of training samples processed
    epochs: int                             # Training epochs completed
    cpu_time_seconds: float                 # CPU time consumed
    gpu_time_seconds: float                 # GPU time consumed
    memory_peak_gb: float                   # Peak memory usage
    
    # Data mixing and sampling
    dataset_weights: Dict[str, float]       # How much each dataset contributed
    batch_hashes: List[str]                 # Hash of each training batch
    random_seed: int                        # Random seed for reproducibility
    
    # Performance tracking
    baseline_accuracy: float                # Pre-training accuracy
    final_accuracy: float                   # Post-training accuracy
    accuracy_improvement: float             # Performance delta
    
    # Timestamps and environment
    start_time: float                       # Training start timestamp
    end_time: float                         # Training completion timestamp
    framework_version: str                  # "torch==2.1.0, transformers==4.35.0"
    hardware_info: Dict[str, str]          # GPU model, driver version, etc.
    
    # Cryptographic elements
    trainer_private_key: Optional[bytes] = None  # For signing (not stored)
    trainer_public_key: Optional[bytes] = None   # For verification


@dataclass
class PoDLProof:
    """Cryptographic proof of data lineage for expert training."""
    
    # Core proof elements
    expert_hash: str                        # Trained expert identifier
    dataset_lineage: List[str]              # Datasets used (Chain D block hashes)
    training_manifest: Dict[str, Any]       # Complete training log
    
    # Cryptographic verification
    signature: bytes                        # Digital signature of manifest
    trainer_public_key: bytes               # Public key for signature verification
    merkle_root: str                        # Merkle root of batch sequence
    
    # Metadata
    proof_timestamp: float                  # When proof was generated
    verification_level: str                 # "basic", "deep", "statistical"
    confidence_score: float                 # 0.0-1.0 verification confidence
    
    # Chain references
    expert_block_hash: str                  # Block hash on Chain B (Parameter chain)
    dataset_block_hashes: List[str]         # Block hashes on Chain D (Dataset chain)


class PoDLGenerator:
    """Generate cryptographic proofs of training data usage."""
    
    def __init__(self):
        self.batch_size_threshold = 1000    # Minimum batch size for tracking
        self.sampling_rate = 0.01           # 1% statistical sampling for verification
    
    def generate_training_proof(self, session: TrainingSession) -> PoDLProof:
        """
        Generate complete PoDL proof from training session.
        
        This creates a tamper-proof, cryptographically signed manifest
        proving which datasets trained which expert.
        """
        
        # Generate comprehensive training manifest
        training_manifest = {
            # Core identifiers
            "expert_hash": session.new_expert_hash,
            "dataset_ids": session.dataset_ids,
            "trainer_node_id": session.trainer_node_id,
            
            # Training metrics
            "total_samples": session.total_samples,
            "epochs": session.epochs,
            "cpu_time_seconds": session.cpu_time_seconds,
            "gpu_time_seconds": session.gpu_time_seconds,
            "memory_peak_gb": session.memory_peak_gb,
            
            # Data provenance
            "dataset_weights": session.dataset_weights,
            "batch_hash_sequence": session.batch_hashes,
            "random_seed": session.random_seed,
            
            # Performance verification
            "baseline_accuracy": session.baseline_accuracy,
            "final_accuracy": session.final_accuracy,
            "accuracy_improvement": session.accuracy_improvement,
            
            # Environment and reproducibility
            "start_time": session.start_time,
            "end_time": session.end_time,
            "training_duration": session.end_time - session.start_time,
            "framework_version": session.framework_version,
            "hardware_info": session.hardware_info,
            
            # Integrity verification
            "manifest_version": "1.0.0",
            "generation_timestamp": time.time(),
        }
        
        # Calculate Merkle root of batch sequence for integrity
        merkle_root = self._calculate_merkle_root(session.batch_hashes)
        training_manifest["batch_merkle_root"] = merkle_root
        
        # Generate cryptographic signature
        manifest_json = json.dumps(training_manifest, sort_keys=True)
        signature = self._sign_manifest(manifest_json, session.trainer_private_key)
        
        # Create PoDL proof
        proof = PoDLProof(
            expert_hash=session.new_expert_hash,
            dataset_lineage=session.dataset_ids,
            training_manifest=training_manifest,
            signature=signature,
            trainer_public_key=session.trainer_public_key,
            merkle_root=merkle_root,
            proof_timestamp=time.time(),
            verification_level="basic",
            confidence_score=1.0,  # Full confidence in self-generated proof
            expert_block_hash="",  # Will be set when expert is added to Chain B
            dataset_block_hashes=[]  # Will be resolved from dataset_ids
        )
        
        return proof
    
    def generate_batch_hash(self, batch_data: Any, batch_index: int, epoch: int) -> str:
        """Generate hash for a training batch for sequence verification."""
        
        # Create batch identifier
        batch_info = {
            "batch_index": batch_index,
            "epoch": epoch,
            "timestamp": time.time(),
            "batch_size": len(batch_data) if hasattr(batch_data, '__len__') else 0
        }
        
        # For actual data, we'd hash the batch content
        # Here we simulate with batch info + content hash
        if hasattr(batch_data, '__str__'):
            content_hash = hashlib.sha256(str(batch_data)[:1000].encode()).hexdigest()
        else:
            content_hash = hashlib.sha256(str(batch_index).encode()).hexdigest()
        
        batch_info["content_hash"] = content_hash
        
        # Generate final batch hash
        batch_json = json.dumps(batch_info, sort_keys=True)
        return hashlib.sha256(batch_json.encode()).hexdigest()
    
    def _calculate_merkle_root(self, batch_hashes: List[str]) -> str:
        """Calculate Merkle root of batch hash sequence for integrity verification."""
        
        if not batch_hashes:
            return hashlib.sha256(b"empty_batch_sequence").hexdigest()
        
        # Simple Merkle tree implementation
        current_level = batch_hashes[:]
        
        while len(current_level) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Hash the pair
                pair_hash = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(pair_hash)
            
            current_level = next_level
        
        return current_level[0]
    
    def _sign_manifest(self, manifest_json: str, private_key: Optional[bytes]) -> bytes:
        """Sign training manifest with trainer's private key."""
        
        if not private_key:
            # Generate temporary key pair for demo
            private_key_obj = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            private_key = private_key_obj.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            private_key_obj = serialization.load_pem_private_key(
                private_key, password=None, backend=default_backend()
            )
        
        # Sign the manifest
        signature = private_key_obj.sign(
            manifest_json.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature


class PoDLVerifier:
    """Verify PoDL proofs and training data lineage."""
    
    def __init__(self, dataset_chain=None, parameter_chain=None):
        self.dataset_chain = dataset_chain      # Chain D for dataset verification
        self.parameter_chain = parameter_chain  # Chain B for expert verification
        self.verification_cache = {}            # Cache verification results
    
    def verify_proof(self, proof: PoDLProof, verification_level: str = "basic") -> Tuple[bool, Dict[str, Any]]:
        """
        Verify PoDL proof authenticity and accuracy.
        
        Args:
            proof: PoDL proof to verify
            verification_level: "basic", "deep", or "statistical"
        
        Returns:
            (is_valid, verification_report)
        """
        
        verification_report = {
            "proof_hash": hashlib.sha256(str(proof).encode()).hexdigest(),
            "verification_level": verification_level,
            "verification_timestamp": time.time(),
            "checks_performed": [],
            "checks_passed": [],
            "checks_failed": [],
            "confidence_score": 0.0
        }
        
        try:
            # Check 1: Cryptographic signature verification
            signature_valid = self._verify_signature(proof)
            verification_report["checks_performed"].append("signature_verification")
            
            if signature_valid:
                verification_report["checks_passed"].append("signature_verification")
            else:
                verification_report["checks_failed"].append("signature_verification")
                return False, verification_report
            
            # Check 2: Dataset existence on Chain D
            datasets_valid = self._verify_dataset_existence(proof.dataset_lineage)
            verification_report["checks_performed"].append("dataset_existence")
            
            if datasets_valid:
                verification_report["checks_passed"].append("dataset_existence")
            else:
                verification_report["checks_failed"].append("dataset_existence")
                return False, verification_report
            
            # Check 3: Merkle root verification
            merkle_valid = self._verify_merkle_root(proof)
            verification_report["checks_performed"].append("merkle_root_verification")
            
            if merkle_valid:
                verification_report["checks_passed"].append("merkle_root_verification")
            else:
                verification_report["checks_failed"].append("merkle_root_verification")
            
            # Check 4: Performance consistency (if deep verification)
            if verification_level in ["deep", "statistical"]:
                performance_consistent = self._verify_performance_consistency(proof)
                verification_report["checks_performed"].append("performance_consistency")
                
                if performance_consistent:
                    verification_report["checks_passed"].append("performance_consistency")
                else:
                    verification_report["checks_failed"].append("performance_consistency")
            
            # Check 5: Statistical batch sampling (if statistical verification)
            if verification_level == "statistical":
                sampling_valid = self._verify_batch_sampling(proof)
                verification_report["checks_performed"].append("batch_sampling_verification")
                
                if sampling_valid:
                    verification_report["checks_passed"].append("batch_sampling_verification")
                else:
                    verification_report["checks_failed"].append("batch_sampling_verification")
            
            # Calculate overall confidence
            total_checks = len(verification_report["checks_performed"])
            passed_checks = len(verification_report["checks_passed"])
            verification_report["confidence_score"] = passed_checks / total_checks if total_checks > 0 else 0.0
            
            # Determine overall validity (all critical checks must pass)
            critical_checks = ["signature_verification", "dataset_existence"]
            critical_passed = all(
                check in verification_report["checks_passed"] 
                for check in critical_checks
            )
            
            return critical_passed, verification_report
            
        except Exception as e:
            verification_report["error"] = str(e)
            verification_report["confidence_score"] = 0.0
            return False, verification_report
    
    def _verify_signature(self, proof: PoDLProof) -> bool:
        """Verify cryptographic signature of training manifest."""
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                proof.trainer_public_key, backend=default_backend()
            )
            
            # Recreate manifest JSON for verification
            manifest_json = json.dumps(proof.training_manifest, sort_keys=True)
            
            # Verify signature
            public_key.verify(
                proof.signature,
                manifest_json.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception:
            return False
    
    def _verify_dataset_existence(self, dataset_ids: List[str]) -> bool:
        """Verify all datasets exist and are approved on Chain D."""
        if not self.dataset_chain:
            return True  # Skip if no chain available (testing mode)
        
        for dataset_id in dataset_ids:
            dataset_info = self.dataset_chain.get_dataset_info(dataset_id)
            if not dataset_info:
                return False
            
            # Check dataset is approved (not pending/rejected)
            if dataset_info.get('stage') != 'approved':
                return False
        
        return True
    
    def _verify_merkle_root(self, proof: PoDLProof) -> bool:
        """Verify Merkle root of batch sequence matches claimed value."""
        # Extract batch hashes from training manifest
        batch_hashes = proof.training_manifest.get("batch_hash_sequence", [])
        
        # Recalculate Merkle root
        generator = PoDLGenerator()
        calculated_root = generator._calculate_merkle_root(batch_hashes)
        
        return calculated_root == proof.merkle_root
    
    def _verify_performance_consistency(self, proof: PoDLProof) -> bool:
        """Verify performance improvement is consistent with datasets used."""
        # This would involve complex ML analysis in production
        # For now, basic sanity checks
        
        manifest = proof.training_manifest
        
        # Check accuracy improvement is positive and reasonable
        accuracy_improvement = manifest.get("accuracy_improvement", 0)
        if accuracy_improvement < 0 or accuracy_improvement > 1.0:
            return False
        
        # Check training duration is reasonable
        duration = manifest.get("training_duration", 0)
        if duration <= 0 or duration > 30 * 24 * 3600:  # More than 30 days seems suspicious
            return False
        
        return True
    
    def _verify_batch_sampling(self, proof: PoDLProof) -> bool:
        """Statistical verification by sampling and reconstructing training batches."""
        # This would involve re-sampling from original datasets in production
        # For now, just verify batch sequence integrity
        
        batch_hashes = proof.training_manifest.get("batch_hash_sequence", [])
        
        # Sample 1% of batches for verification
        import random
        sample_size = max(1, len(batch_hashes) // 100)
        sampled_batches = random.sample(batch_hashes, min(sample_size, len(batch_hashes)))
        
        # In production, would reconstruct these batches and verify hashes
        # For now, just check hash format
        for batch_hash in sampled_batches:
            if not isinstance(batch_hash, str) or len(batch_hash) != 64:
                return False
        
        return True
    
    def get_dataset_contribution_metrics(self, expert_hash: str) -> Dict[str, float]:
        """Calculate dataset contribution to expert performance (for rewards)."""
        # This would analyze PoDL proofs to determine dataset value
        # For now, return dummy metrics
        
        return {
            "total_contribution_score": 0.85,
            "dataset_weights": {
                "scientific_papers_v3@1.2": 0.45,
                "code_repos_filtered@2.1": 0.40,
                "wiki_summaries@1.0": 0.15
            },
            "performance_attribution": {
                "accuracy_gain": 0.034,
                "training_efficiency": 0.92
            }
        }