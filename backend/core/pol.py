"""Enhanced Proof-of-Learning (PoL) evaluation system.

Comprehensive delta validation system that prevents gradient poisoning
and ensures genuine learning improvements.
"""

from __future__ import annotations

import random
import time
import hashlib
import json
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class PoLScore:
    """Comprehensive PoL evaluation result."""
    is_valid: bool
    improvement_score: float
    confidence_score: float
    fraud_probability: float
    validation_loss_before: float
    validation_loss_after: float
    metadata: Dict[str, Any]

@dataclass
class ValidationDataset:
    """Small validation dataset for PoL evaluation."""
    inputs: List[torch.Tensor]
    targets: List[torch.Tensor]
    dataset_hash: str
    
    @classmethod
    def create_from_samples(cls, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> 'ValidationDataset':
        inputs = [s[0] for s in samples]
        targets = [s[1] for s in samples]
        
        # Create deterministic hash
        all_data = torch.cat([torch.cat(inputs).flatten(), torch.cat(targets).flatten()])
        dataset_hash = hashlib.sha256(all_data.cpu().numpy().tobytes()).hexdigest()
        
        return cls(inputs=inputs, targets=targets, dataset_hash=dataset_hash)

class EnhancedPoLValidator:
    """Advanced PoL validator with fraud detection."""
    
    def __init__(self, 
                 validation_dataset: Optional[ValidationDataset] = None,
                 min_improvement_threshold: float = 0.005,
                 fraud_detection_threshold: float = 0.7):
        
        self.validation_dataset = validation_dataset or self._create_mock_dataset()
        self.min_improvement_threshold = min_improvement_threshold
        self.fraud_detection_threshold = fraud_detection_threshold
        
        # Historical validation for fraud detection
        self.validation_history: List[Dict[str, Any]] = []
        self.expert_baselines: Dict[str, float] = {}
        
    def _create_mock_dataset(self) -> ValidationDataset:
        """Create mock validation dataset for testing."""
        samples = []
        for _ in range(100):  # Small validation set
            input_tensor = torch.randn(10)
            target_tensor = torch.randn(1)
            samples.append((input_tensor, target_tensor))
        
        return ValidationDataset.create_from_samples(samples)
    
    def evaluate_delta(self, 
                      expert_name: str,
                      original_weights: Dict[str, torch.Tensor],
                      delta_weights: Dict[str, torch.Tensor],
                      node_id: str) -> PoLScore:
        """
        Comprehensive delta evaluation with fraud detection.
        
        Args:
            expert_name: Name of expert being updated
            original_weights: Original expert weights
            delta_weights: Delta weights to apply
            node_id: ID of node submitting delta
            
        Returns:
            PoLScore with validation results
        """
        start_time = time.time()
        
        try:
            # 1. Apply delta to get candidate weights
            candidate_weights = self._apply_delta(original_weights, delta_weights)
            
            # 2. Evaluate performance before and after
            loss_before = self._evaluate_weights(original_weights)
            loss_after = self._evaluate_weights(candidate_weights)
            
            # 3. Calculate improvement
            improvement = loss_before - loss_after
            improvement_score = improvement / max(loss_before, 1e-8)
            
            # 4. Fraud detection analysis
            fraud_probability = self._detect_fraud(
                expert_name, node_id, improvement_score, 
                loss_before, loss_after, delta_weights
            )
            
            # 5. Confidence scoring
            confidence_score = self._calculate_confidence(
                expert_name, improvement_score, fraud_probability
            )
            
            # 6. Final validation decision
            is_valid = (
                improvement_score >= self.min_improvement_threshold and
                fraud_probability < self.fraud_detection_threshold and
                confidence_score > 0.5
            )
            
            # 7. Record validation event
            validation_event = {
                'timestamp': time.time(),
                'expert_name': expert_name,
                'node_id': node_id,
                'loss_before': loss_before,
                'loss_after': loss_after,
                'improvement_score': improvement_score,
                'fraud_probability': fraud_probability,
                'is_valid': is_valid,
                'evaluation_time_ms': (time.time() - start_time) * 1000
            }
            self.validation_history.append(validation_event)
            
            # Keep only last 1000 validations
            if len(self.validation_history) > 1000:
                self.validation_history.pop(0)
            
            # Update expert baseline
            if is_valid:
                self.expert_baselines[expert_name] = loss_after
            
            return PoLScore(
                is_valid=is_valid,
                improvement_score=improvement_score,
                confidence_score=confidence_score,
                fraud_probability=fraud_probability,
                validation_loss_before=loss_before,
                validation_loss_after=loss_after,
                metadata={
                    'expert_name': expert_name,
                    'node_id': node_id,
                    'evaluation_time_ms': validation_event['evaluation_time_ms'],
                    'dataset_hash': self.validation_dataset.dataset_hash
                }
            )
            
        except Exception as e:
            print(f"❌ PoL evaluation failed for {expert_name}: {e}")
            return PoLScore(
                is_valid=False,
                improvement_score=0.0,
                confidence_score=0.0,
                fraud_probability=1.0,
                validation_loss_before=float('inf'),
                validation_loss_after=float('inf'),
                metadata={'error': str(e)}
            )
    
    def _apply_delta(self, original_weights: Dict[str, torch.Tensor], 
                    delta_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply delta to original weights."""
        candidate_weights = {}
        
        for key, original_tensor in original_weights.items():
            if key in delta_weights:
                candidate_weights[key] = original_tensor + delta_weights[key]
            else:
                candidate_weights[key] = original_tensor.clone()
        
        return candidate_weights
    
    def _evaluate_weights(self, weights: Dict[str, torch.Tensor]) -> float:
        """Evaluate weights on validation dataset."""
        # Simplified evaluation - in practice this would run actual inference
        total_loss = 0.0
        
        for input_tensor, target_tensor in zip(
            self.validation_dataset.inputs, 
            self.validation_dataset.targets
        ):
            # Mock forward pass with weights
            # In real implementation, this would be actual model inference
            predicted = self._mock_forward_pass(input_tensor, weights)
            loss = torch.nn.functional.mse_loss(predicted, target_tensor)
            total_loss += loss.item()
        
        return total_loss / len(self.validation_dataset.inputs)
    
    def _mock_forward_pass(self, input_tensor: torch.Tensor, 
                          weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Mock forward pass for evaluation."""
        # Simplified mock - in practice this would be actual expert inference
        output = input_tensor.mean().unsqueeze(0)
        
        # Add some variation based on weights
        if weights:
            weight_influence = sum(w.mean().item() for w in weights.values()) / len(weights)
            output += weight_influence * 0.1
        
        return output
    
    def _detect_fraud(self, expert_name: str, node_id: str, improvement_score: float,
                     loss_before: float, loss_after: float, 
                     delta_weights: Dict[str, torch.Tensor]) -> float:
        """Detect potential fraud in delta submission."""
        fraud_indicators = []
        
        # 1. Check for impossibly large improvements
        if improvement_score > 0.5:  # >50% improvement is suspicious
            fraud_indicators.append(0.8)
        
        # 2. Check for suspicious delta magnitudes
        if delta_weights:
            delta_norms = [torch.norm(delta).item() for delta in delta_weights.values()]
            max_delta_norm = max(delta_norms)
            
            if max_delta_norm > 10.0:  # Very large delta
                fraud_indicators.append(0.7)
            elif max_delta_norm < 1e-6:  # Suspiciously small delta
                fraud_indicators.append(0.6)
        
        # 3. Check node submission history
        node_history = [v for v in self.validation_history if v['node_id'] == node_id]
        if len(node_history) >= 3:
            recent_improvements = [v['improvement_score'] for v in node_history[-3:]]
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement > 0.1:  # Consistently high improvements
                fraud_indicators.append(0.6)
            elif all(imp < 0 for imp in recent_improvements):  # Always making things worse
                fraud_indicators.append(0.9)
        
        # 4. Check expert-specific patterns
        expert_history = [v for v in self.validation_history if v['expert_name'] == expert_name]
        if len(expert_history) >= 5:
            historical_improvements = [v['improvement_score'] for v in expert_history[-5:]]
            
            if improvement_score > np.mean(historical_improvements) + 3 * np.std(historical_improvements):
                fraud_indicators.append(0.7)  # Statistical outlier
        
        # 5. Check for gradient reversal patterns
        if loss_after > loss_before * 1.5:  # Making things much worse
            fraud_indicators.append(0.9)
        
        # Calculate overall fraud probability
        if not fraud_indicators:
            return 0.0
        
        return min(1.0, max(fraud_indicators))
    
    def _calculate_confidence(self, expert_name: str, improvement_score: float, 
                            fraud_probability: float) -> float:
        """Calculate confidence in the validation result."""
        base_confidence = 0.5
        
        # Higher confidence for moderate improvements
        if 0.01 <= improvement_score <= 0.2:
            base_confidence += 0.3
        
        # Lower confidence for high fraud probability
        base_confidence -= fraud_probability * 0.4
        
        # Higher confidence if we have historical data for this expert
        if expert_name in self.expert_baselines:
            base_confidence += 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def get_expert_performance_history(self, expert_name: str) -> List[Dict[str, Any]]:
        """Get performance history for an expert."""
        return [v for v in self.validation_history if v['expert_name'] == expert_name]
    
    def get_node_trust_score(self, node_id: str) -> float:
        """Calculate trust score for a node based on submission history."""
        node_history = [v for v in self.validation_history if v['node_id'] == node_id]
        
        if not node_history:
            return 0.5  # Neutral trust for new nodes
        
        # Calculate success rate
        valid_submissions = sum(1 for v in node_history if v['is_valid'])
        success_rate = valid_submissions / len(node_history)
        
        # Calculate average fraud probability
        avg_fraud_prob = np.mean([v['fraud_probability'] for v in node_history])
        
        # Calculate trust score
        trust_score = success_rate * 0.7 + (1 - avg_fraud_prob) * 0.3
        
        return max(0.0, min(1.0, trust_score))
    
    def export_validation_report(self) -> Dict[str, Any]:
        """Export comprehensive validation report."""
        if not self.validation_history:
            return {'status': 'no_data'}
        
        total_validations = len(self.validation_history)
        valid_validations = sum(1 for v in self.validation_history if v['is_valid'])
        
        return {
            'total_validations': total_validations,
            'valid_validations': valid_validations,
            'success_rate': valid_validations / total_validations,
            'avg_fraud_probability': np.mean([v['fraud_probability'] for v in self.validation_history]),
            'avg_improvement_score': np.mean([v['improvement_score'] for v in self.validation_history]),
            'unique_experts': len(set(v['expert_name'] for v in self.validation_history)),
            'unique_nodes': len(set(v['node_id'] for v in self.validation_history)),
            'dataset_hash': self.validation_dataset.dataset_hash,
            'expert_baselines': self.expert_baselines.copy()
        }


# Legacy functions for backward compatibility
def mock_validation_loss() -> float:
    """Placeholder: returns a random loss between 0.5 and 2.0."""
    return random.uniform(0.5, 2.0)


def evaluate_candidate(
    candidate_loss_fn: Callable[[], float] | None = None,
    previous_loss: float | None = None,
    threshold: float = 0.005,
) -> bool:
    """Return True if candidate improves by `threshold` fraction."""
    if candidate_loss_fn is None:
        candidate_loss_fn = mock_validation_loss

    cand_loss = candidate_loss_fn()
    if previous_loss is None:
        return True  # first model wins by default

    improvement = previous_loss - cand_loss
    required = threshold * previous_loss
    return improvement >= required 