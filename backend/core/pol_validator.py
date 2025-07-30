"""Proof-of-Learning (PoL) Validation System.

This module implements a comprehensive PoL system that evaluates expert blocks
based on model improvement rather than hash difficulty.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import ecdsa

from .block import Block
from ..model.moe_infer import MoEModelManager


class ValidationDatasetManager:
    """Manages public validation datasets for PoL evaluation."""
    
    def __init__(self, data_dir: Path = Path("./validation_data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self._datasets = {}
        self._baseline_models = {}
        
    def get_validation_dataset(self, task_name: str = "default") -> Tuple[torch.Tensor, torch.Tensor]:
        """Get validation dataset for a specific task."""
        if task_name in self._datasets:
            return self._datasets[task_name]
        
        # Create synthetic validation dataset for now
        # In production, this would load real validation data
        if task_name == "default":
            # Synthetic language modeling task
            vocab_size = 1000
            seq_length = 32
            batch_size = 16
            
            # Random input sequences
            inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
            # Next token prediction targets
            targets = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            self._datasets[task_name] = (inputs, targets)
            return inputs, targets
        
        raise ValueError(f"Unknown validation task: {task_name}")
    
    def get_baseline_score(self, task_name: str = "default") -> float:
        """Get baseline model performance for comparison."""
        if task_name in self._baseline_models:
            return self._baseline_models[task_name]
        
        # Mock baseline score - in production, this would be from a real baseline model
        baseline_score = random.uniform(0.6, 0.8)  # Random accuracy between 60-80%
        self._baseline_models[task_name] = baseline_score
        
        print(f"📊 Baseline score for {task_name}: {baseline_score:.3f}")
        return baseline_score


class PoLEvaluator:
    """Evaluates expert blocks using Proof-of-Learning metrics."""
    
    def __init__(
        self, 
        validation_manager: ValidationDatasetManager,
        threshold: float = 0.01,  # 1% improvement required
        task_name: str = "default"
    ):
        self.validation_manager = validation_manager
        self.threshold = threshold  # Minimum improvement required
        self.task_name = task_name
        
    def evaluate_pol_score(
        self, 
        expert_block: Block,
        model_manager: MoEModelManager,
        previous_score: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate an expert block using Proof-of-Learning.
        
        Args:
            expert_block: The expert block to evaluate
            model_manager: MoE model manager for loading/testing experts
            previous_score: Previous best score for this expert (optional)
            
        Returns:
            Tuple of (delta_score, metrics_dict)
            - delta_score: Improvement over baseline/previous (positive = better)
            - metrics_dict: Detailed evaluation metrics
        """
        try:
            # Load validation dataset
            inputs, targets = self.validation_manager.get_validation_dataset(self.task_name)
            baseline_score = self.validation_manager.get_baseline_score(self.task_name)
            
            # Extract expert from block
            expert_tensors = self._extract_expert_from_block(expert_block)
            expert_name = expert_block.header.expert_name
            layer_id = expert_block.header.layer_id
            
            if not expert_name or not layer_id:
                raise ValueError("Expert block missing expert_name or layer_id")
            
            # Temporarily load expert into model manager
            with model_manager.temporary_expert_override(expert_name, expert_tensors):
                # Evaluate model performance with new expert
                candidate_score = self._evaluate_model_performance(
                    model_manager, inputs, targets, expert_name, layer_id
                )
            
            # Calculate improvement
            comparison_score = previous_score if previous_score is not None else baseline_score
            delta_score = candidate_score - comparison_score
            
            # Detailed metrics
            metrics = {
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "previous_score": previous_score,
                "delta_score": delta_score,
                "improvement_percentage": (delta_score / comparison_score * 100) if comparison_score > 0 else 0,
                "meets_threshold": delta_score >= (self.threshold * comparison_score),
                "expert_name": expert_name,
                "layer_id": layer_id,
                "evaluation_time": time.time(),
                "task_name": self.task_name
            }
            
            print(f"🧠 PoL Evaluation for {expert_name}:")
            print(f"   Candidate: {candidate_score:.4f}")
            print(f"   Baseline:  {comparison_score:.4f}")
            print(f"   Delta:     {delta_score:+.4f} ({metrics['improvement_percentage']:+.2f}%)")
            print(f"   Threshold: {self.threshold*100:.1f}% ({'✅ PASS' if metrics['meets_threshold'] else '❌ FAIL'})")
            
            return delta_score, metrics
            
        except Exception as e:
            print(f"❌ PoL evaluation failed: {e}")
            # Return negative score on evaluation failure
            return -1.0, {
                "error": str(e),
                "expert_name": getattr(expert_block.header, 'expert_name', 'unknown'),
                "evaluation_time": time.time()
            }
    
    def _extract_expert_from_block(self, expert_block: Block) -> Dict[str, torch.Tensor]:
        """Extract expert tensors from a block payload."""
        try:
            # Assuming the payload is a serialized PyTorch state dict
            payload_bytes = expert_block.payload
            buffer = io.BytesIO(payload_bytes)
            expert_tensors = torch.load(buffer, map_location='cpu')
            
            # Handle different serialization formats
            if isinstance(expert_tensors, dict):
                return expert_tensors
            else:
                raise ValueError(f"Unexpected expert tensor format: {type(expert_tensors)}")
                
        except Exception as e:
            raise ValueError(f"Failed to extract expert tensors: {e}")
    
    def _evaluate_model_performance(
        self, 
        model_manager: MoEModelManager,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        expert_name: str,
        layer_id: str
    ) -> float:
        """Evaluate model performance with the candidate expert."""
        try:
            # Create a simple evaluation task
            model_manager.model.eval()
            
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                batch_size = min(8, inputs.size(0))  # Small batches for evaluation
                
                for i in range(0, inputs.size(0), batch_size):
                    batch_inputs = inputs[i:i+batch_size]
                    batch_targets = targets[i:i+batch_size]
                    
                    # Forward pass through MoE model
                    try:
                        # This is a simplified evaluation - in practice you'd need
                        # to adapt this to your specific model architecture
                        outputs = self._mock_forward_pass(batch_inputs, batch_targets)
                        
                        # Calculate loss (cross-entropy for classification)
                        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                                             batch_targets.view(-1), ignore_index=-1)
                        
                        # Calculate accuracy
                        predictions = torch.argmax(outputs, dim=-1)
                        correct = (predictions == batch_targets).float().sum()
                        
                        total_loss += loss.item() * batch_inputs.size(0)
                        total_correct += correct.item()
                        total_samples += batch_inputs.numel()
                        
                    except Exception as e:
                        print(f"Warning: Batch evaluation failed: {e}")
                        continue
            
            if total_samples == 0:
                return 0.0
            
            # Return accuracy score (higher is better)
            accuracy = total_correct / total_samples
            return accuracy
            
        except Exception as e:
            print(f"Warning: Model evaluation failed: {e}")
            return 0.0  # Return 0 on evaluation failure
    
    def _mock_forward_pass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mock forward pass for evaluation (replace with real model inference)."""
        # This is a placeholder - in practice you'd use the actual MoE model
        batch_size, seq_length = inputs.shape
        vocab_size = 1000  # Match the validation dataset
        
        # Return random logits for now
        return torch.randn(batch_size, seq_length, vocab_size)


class ChainValidator:
    """Comprehensive blockchain validator with PoL support."""
    
    def __init__(
        self,
        pol_evaluator: PoLEvaluator,
        model_manager: MoEModelManager,
        enable_pol: bool = True,
        enable_pow: bool = False,
        pol_threshold: float = 0.01,
        max_evaluation_time: float = 30.0  # 30 seconds max per evaluation
    ):
        self.pol_evaluator = pol_evaluator
        self.model_manager = model_manager
        self.enable_pol = enable_pol
        self.enable_pow = enable_pow
        self.pol_threshold = pol_threshold
        self.max_evaluation_time = max_evaluation_time
        
        print(f"🔍 ChainValidator initialized:")
        print(f"   PoL enabled: {enable_pol}")
        print(f"   PoW enabled: {enable_pow}")
        print(f"   PoL threshold: {pol_threshold*100:.1f}%")
    
    def validate_block(
        self, 
        block: Block, 
        previous_score: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive block validation with PoL support.
        
        Args:
            block: Block to validate
            previous_score: Previous performance score for this expert
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_start = time.time()
        validation_details = {
            "block_hash": block.compute_hash(),
            "expert_name": getattr(block.header, 'expert_name', None),
            "block_type": getattr(block.header, 'block_type', 'unknown'),
            "validation_start": validation_start
        }
        
        try:
            # 1. Basic block integrity checks
            if not self._validate_block_integrity(block):
                validation_details["failure_reason"] = "Block integrity check failed"
                return False, validation_details
            
            # 2. ECDSA signature verification
            if not self._validate_signature(block):
                validation_details["failure_reason"] = "Signature verification failed"
                return False, validation_details
            
            # 3. PoW validation (if enabled)
            if self.enable_pow:
                if not self._validate_pow(block):
                    validation_details["failure_reason"] = "PoW validation failed"
                    return False, validation_details
                validation_details["pow_validated"] = True
            
            # 4. PoL validation (if enabled and applicable)
            if self.enable_pol and self._should_apply_pol(block):
                pol_valid, pol_details = self._validate_pol(block, previous_score)
                validation_details.update(pol_details)
                
                if not pol_valid:
                    validation_details["failure_reason"] = "PoL validation failed"
                    return False, validation_details
            
            # All validations passed
            validation_details["validation_time"] = time.time() - validation_start
            validation_details["success"] = True
            
            print(f"✅ Block validation passed: {block.compute_hash()[:16]}...")
            return True, validation_details
            
        except Exception as e:
            validation_details["failure_reason"] = f"Validation error: {str(e)}"
            validation_details["validation_time"] = time.time() - validation_start
            print(f"❌ Block validation error: {e}")
            return False, validation_details
    
    def _validate_block_integrity(self, block: Block) -> bool:
        """Validate basic block integrity."""
        try:
            # Check payload hash
            expected_hash = hashlib.sha256(block.payload).hexdigest()
            if block.header.payload_hash != expected_hash:
                print("❌ Payload hash mismatch")
                return False
            
            # Check payload size
            if block.header.payload_size != len(block.payload):
                print("❌ Payload size mismatch")
                return False
            
            # Check required fields for expert blocks
            if hasattr(block.header, 'block_type') and block.header.block_type in ('expert', 'router'):
                if not getattr(block.header, 'expert_name', None):
                    print("❌ Expert block missing expert_name")
                    return False
                if not getattr(block.header, 'layer_id', None):
                    print("❌ Expert block missing layer_id")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Block integrity check error: {e}")
            return False
    
    def _validate_signature(self, block: Block) -> bool:
        """Validate ECDSA signature."""
        try:
            if not block.miner_pub or not block.payload_sig:
                print("❌ Missing signature or public key")
                return False
            
            # Verify signature
            verifying_key = ecdsa.VerifyingKey.from_string(
                bytes.fromhex(block.miner_pub), 
                curve=ecdsa.SECP256k1
            )
            
            signature = bytes.fromhex(block.payload_sig)
            message = block.payload
            
            try:
                verifying_key.verify(signature, message, hashfunc=hashlib.sha256)
                return True
            except ecdsa.BadSignatureError:
                print("❌ Invalid signature")
                return False
                
        except Exception as e:
            print(f"❌ Signature validation error: {e}")
            return False
    
    def _validate_pow(self, block: Block) -> bool:
        """Validate Proof-of-Work (if enabled)."""
        try:
            # Import here to avoid circular dependency
            from .pow import verify_pow
            
            # Use difficulty from environment or default
            difficulty = int(os.environ.get('CHAIN_DIFFICULTY', '4'))
            
            data = block.header.to_json().encode() + block.payload
            return verify_pow(data, block.header.nonce, difficulty)
            
        except Exception as e:
            print(f"❌ PoW validation error: {e}")
            return False
    
    def _should_apply_pol(self, block: Block) -> bool:
        """Determine if PoL validation should be applied to this block."""
        # Apply PoL only to expert and router blocks
        block_type = getattr(block.header, 'block_type', None)
        return block_type in ('expert', 'router')
    
    def _validate_pol(
        self, 
        block: Block, 
        previous_score: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate using Proof-of-Learning."""
        try:
            start_time = time.time()
            
            # Run PoL evaluation with timeout
            delta_score, pol_metrics = self.pol_evaluator.evaluate_pol_score(
                block, self.model_manager, previous_score
            )
            
            evaluation_time = time.time() - start_time
            
            # Check if evaluation took too long
            if evaluation_time > self.max_evaluation_time:
                print(f"⚠️  PoL evaluation timeout ({evaluation_time:.1f}s > {self.max_evaluation_time}s)")
                return False, {
                    "pol_timeout": True,
                    "evaluation_time": evaluation_time,
                    "pol_metrics": pol_metrics
                }
            
            # Check if improvement meets threshold
            meets_threshold = pol_metrics.get("meets_threshold", False)
            
            pol_details = {
                "pol_validated": True,
                "pol_score": delta_score,
                "pol_metrics": pol_metrics,
                "evaluation_time": evaluation_time,
                "meets_threshold": meets_threshold
            }
            
            return meets_threshold, pol_details
            
        except Exception as e:
            print(f"❌ PoL validation error: {e}")
            return False, {
                "pol_error": str(e),
                "pol_validated": False
            }


# Utility functions for integration with existing codebase

def create_pol_validator(
    model_manager: MoEModelManager,
    enable_pol: bool = True,
    pol_threshold: float = 0.01,
    validation_data_dir: Path = Path("./validation_data")
) -> ChainValidator:
    """Factory function to create a configured ChainValidator with PoL support."""
    
    # Create validation dataset manager
    validation_manager = ValidationDatasetManager(validation_data_dir)
    
    # Create PoL evaluator
    pol_evaluator = PoLEvaluator(
        validation_manager=validation_manager,
        threshold=pol_threshold,
        task_name="default"
    )
    
    # Create chain validator
    validator = ChainValidator(
        pol_evaluator=pol_evaluator,
        model_manager=model_manager,
        enable_pol=enable_pol,
        enable_pow=False,  # Disable PoW when using PoL
        pol_threshold=pol_threshold
    )
    
    return validator