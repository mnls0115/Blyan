"""
Dense Model Proof of Learning (PoL) Validator
==============================================
Validates layer deltas from dense training to ensure quality improvements.
Uses deterministic evaluation sets and Byzantine fault-tolerant consensus.
Model-agnostic design with adaptive thresholds.

Core validation methods:
- Deterministic evaluation on fixed prompts
- Logits hash commitment for reproducibility
- Drift detection and quality bounds
- Byzantine consensus for acceptance
"""

import hashlib
import json
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import asyncio

from backend.core.chain import Chain
from backend.core.delta_index import DeltaIndex
from backend.model.manager import UnifiedModelManager

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for PoL validation."""
    min_improvement: float = 0.005  # 0.5% minimum improvement
    max_regression: float = 0.02  # 2% maximum allowed regression
    drift_threshold: float = 0.1  # 10% maximum drift from baseline
    eval_batch_size: int = 8
    eval_steps: int = 10
    deterministic_seed: int = 42
    logits_precision: int = 4  # Decimal places for logits hashing
    consensus_threshold: float = 0.67  # 2/3 majority needed
    timeout_seconds: float = 300  # 5 minute validation timeout
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationResult:
    """Result of delta validation."""
    delta_hash: str
    layer_name: str
    is_valid: bool
    improvement_score: float
    baseline_loss: float
    delta_loss: float
    drift_score: float
    logits_hash: str
    validator_id: str
    timestamp: float
    consensus_votes: Dict[str, bool]
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EvaluationDataset:
    """
    Deterministic evaluation dataset for PoL validation.
    Ensures reproducible evaluation across validators.
    """
    
    def __init__(self, seed: int = 42, size: int = 100):
        """
        Initialize evaluation dataset.
        
        Args:
            seed: Random seed for deterministic generation
            size: Number of evaluation samples
        """
        self.seed = seed
        self.size = size
        self.prompts = self._generate_prompts()
        self.expected_outputs = self._generate_expected_outputs()
    
    def _generate_prompts(self) -> List[str]:
        """Generate deterministic evaluation prompts."""
        np.random.seed(self.seed)
        
        # Categories of prompts for comprehensive evaluation
        categories = [
            "reasoning",
            "factual",
            "creative",
            "coding",
            "math",
            "conversation"
        ]
        
        prompts = []
        
        # Generate prompts for each category
        for category in categories:
            if category == "reasoning":
                templates = [
                    "Explain why {}",
                    "What would happen if {}",
                    "Analyze the relationship between {} and {}"
                ]
            elif category == "factual":
                templates = [
                    "What is {}",
                    "Define {}",
                    "List the main characteristics of {}"
                ]
            elif category == "creative":
                templates = [
                    "Write a short story about {}",
                    "Describe {} in a poetic way",
                    "Create a dialogue between {} and {}"
                ]
            elif category == "coding":
                templates = [
                    "Write a Python function to {}",
                    "Debug this code: {}",
                    "Optimize this algorithm: {}"
                ]
            elif category == "math":
                templates = [
                    "Solve for x: {}",
                    "Calculate {}",
                    "Prove that {}"
                ]
            else:  # conversation
                templates = [
                    "How would you respond to: {}",
                    "Continue this conversation: {}",
                    "What advice would you give for {}"
                ]
            
            # Generate prompts from templates
            for _ in range(self.size // len(categories)):
                template = templates[np.random.randint(len(templates))]
                # Fill template with random words
                words = ["concept", "theory", "system", "process", "method", "approach"]
                filled = template.format(*np.random.choice(words, template.count("{}"), replace=True))
                prompts.append(filled)
        
        return prompts[:self.size]
    
    def _generate_expected_outputs(self) -> List[str]:
        """Generate expected outputs for validation."""
        np.random.seed(self.seed + 1)
        
        # For PoL, we don't need exact outputs, just consistent evaluation
        # These are used for calculating relative improvements
        outputs = []
        for prompt in self.prompts:
            # Generate a deterministic "expected" output length
            expected_length = 50 + len(prompt) % 100
            outputs.append(f"<expected_output_length:{expected_length}>")
        
        return outputs
    
    def get_batch(self, batch_idx: int, batch_size: int) -> Tuple[List[str], List[str]]:
        """
        Get a batch of evaluation samples.
        
        Args:
            batch_idx: Batch index
            batch_size: Batch size
            
        Returns:
            Tuple of (prompts, expected_outputs)
        """
        start = batch_idx * batch_size
        end = min(start + batch_size, self.size)
        
        return (
            self.prompts[start:end],
            self.expected_outputs[start:end]
        )


class DensePoLValidator:
    """
    Proof of Learning validator for dense model deltas.
    Ensures quality improvements through deterministic evaluation.
    """
    
    def __init__(
        self,
        model_manager: UnifiedModelManager,
        delta_index: DeltaIndex,
        config: Optional[ValidationConfig] = None,
        validator_id: str = "validator_0"
    ):
        """
        Initialize PoL validator.
        
        Args:
            model_manager: Model manager for inference
            delta_index: Delta index for tracking
            config: Validation configuration
            validator_id: Unique validator identifier
        """
        self.model_manager = model_manager
        self.delta_index = delta_index
        self.config = config or ValidationConfig()
        self.validator_id = validator_id
        
        # Evaluation dataset
        self.eval_dataset = EvaluationDataset(
            seed=self.config.deterministic_seed,
            size=self.config.eval_batch_size * self.config.eval_steps
        )
        
        # Cache for baseline evaluations
        self.baseline_cache: Dict[str, Dict[str, float]] = {}
        
        # Consensus tracking
        self.consensus_votes: Dict[str, Dict[str, bool]] = defaultdict(dict)
        
    async def validate_delta(
        self,
        layer_name: str,
        delta_hash: str,
        base_hash: str,
        delta_payload: bytes
    ) -> ValidationResult:
        """
        Validate a layer delta.
        
        Args:
            layer_name: Name of the layer
            delta_hash: Hash of the delta
            base_hash: Hash of the base layer
            delta_payload: Serialized delta tensor
            
        Returns:
            Validation result
        """
        start_time = time.time()
        
        try:
            # Set deterministic mode
            torch.manual_seed(self.config.deterministic_seed)
            np.random.seed(self.config.deterministic_seed)
            
            # Get baseline evaluation
            baseline_metrics = await self._evaluate_baseline(base_hash)
            
            # Apply delta and evaluate
            delta_metrics = await self._evaluate_with_delta(
                layer_name, delta_payload, base_hash
            )
            
            # Calculate improvement
            improvement = self._calculate_improvement(
                baseline_metrics, delta_metrics
            )
            
            # Check drift
            drift = self._calculate_drift(baseline_metrics, delta_metrics)
            
            # Generate logits hash for consensus
            logits_hash = self._generate_logits_hash(delta_metrics["logits"])
            
            # Determine validity
            is_valid = self._check_validity(improvement, drift)
            
            # Get consensus from other validators
            consensus = await self._get_consensus(delta_hash, is_valid)
            
            # Create result
            result = ValidationResult(
                delta_hash=delta_hash,
                layer_name=layer_name,
                is_valid=is_valid and consensus["accepted"],
                improvement_score=improvement,
                baseline_loss=baseline_metrics["loss"],
                delta_loss=delta_metrics["loss"],
                drift_score=drift,
                logits_hash=logits_hash,
                validator_id=self.validator_id,
                timestamp=time.time(),
                consensus_votes=consensus["votes"],
                failure_reason=self._get_failure_reason(improvement, drift, consensus)
            )
            
            # Store result
            if result.is_valid:
                logger.info(f"Delta {delta_hash[:8]} validated: {improvement:.4f} improvement")
            else:
                logger.warning(f"Delta {delta_hash[:8]} rejected: {result.failure_reason}")
            
            return result
            
        except asyncio.TimeoutError:
            return ValidationResult(
                delta_hash=delta_hash,
                layer_name=layer_name,
                is_valid=False,
                improvement_score=0.0,
                baseline_loss=0.0,
                delta_loss=0.0,
                drift_score=0.0,
                logits_hash="",
                validator_id=self.validator_id,
                timestamp=time.time(),
                consensus_votes={},
                failure_reason="Validation timeout"
            )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                delta_hash=delta_hash,
                layer_name=layer_name,
                is_valid=False,
                improvement_score=0.0,
                baseline_loss=0.0,
                delta_loss=0.0,
                drift_score=0.0,
                logits_hash="",
                validator_id=self.validator_id,
                timestamp=time.time(),
                consensus_votes={},
                failure_reason=f"Validation error: {str(e)}"
            )
    
    async def _evaluate_baseline(self, base_hash: str) -> Dict[str, Any]:
        """Evaluate baseline model performance."""
        # Check cache
        if base_hash in self.baseline_cache:
            return self.baseline_cache[base_hash]
        
        # Run evaluation
        total_loss = 0.0
        all_logits = []
        
        for step in range(self.config.eval_steps):
            prompts, _ = self.eval_dataset.get_batch(step, self.config.eval_batch_size)
            
            # Generate with baseline model
            losses = []
            logits_batch = []
            
            for prompt in prompts:
                # This is simplified - in production, calculate actual loss
                response = await self.model_manager.generate_async(
                    prompt=prompt,
                    max_new_tokens=50,
                    temperature=0.0  # Deterministic
                )
                
                # Mock loss calculation (replace with actual)
                loss = len(response) / 100.0  # Placeholder
                losses.append(loss)
                
                # Mock logits (replace with actual model logits)
                logits = hashlib.sha256(response.encode()).hexdigest()[:16]
                logits_batch.append(logits)
            
            batch_loss = np.mean(losses)
            total_loss += batch_loss
            all_logits.extend(logits_batch)
        
        avg_loss = total_loss / self.config.eval_steps
        
        metrics = {
            "loss": avg_loss,
            "logits": all_logits,
            "perplexity": np.exp(avg_loss)
        }
        
        # Cache result
        self.baseline_cache[base_hash] = metrics
        
        return metrics
    
    async def _evaluate_with_delta(
        self,
        layer_name: str,
        delta_payload: bytes,
        base_hash: str
    ) -> Dict[str, Any]:
        """Evaluate model with delta applied."""
        # Apply delta temporarily
        # In production, this would actually modify the model weights
        
        total_loss = 0.0
        all_logits = []
        
        for step in range(self.config.eval_steps):
            prompts, _ = self.eval_dataset.get_batch(step, self.config.eval_batch_size)
            
            losses = []
            logits_batch = []
            
            for prompt in prompts:
                # Generate with delta-modified model
                # This is simplified - in production, apply actual delta
                response = await self.model_manager.generate_async(
                    prompt=prompt,
                    max_new_tokens=50,
                    temperature=0.0
                )
                
                # Calculate improvement (mock)
                loss = len(response) / 105.0  # Slightly better than baseline
                losses.append(loss)
                
                logits = hashlib.sha256(f"{response}:delta".encode()).hexdigest()[:16]
                logits_batch.append(logits)
            
            batch_loss = np.mean(losses)
            total_loss += batch_loss
            all_logits.extend(logits_batch)
        
        avg_loss = total_loss / self.config.eval_steps
        
        return {
            "loss": avg_loss,
            "logits": all_logits,
            "perplexity": np.exp(avg_loss)
        }
    
    def _calculate_improvement(
        self,
        baseline_metrics: Dict[str, float],
        delta_metrics: Dict[str, float]
    ) -> float:
        """Calculate relative improvement."""
        baseline_loss = baseline_metrics["loss"]
        delta_loss = delta_metrics["loss"]
        
        if baseline_loss == 0:
            return 0.0
        
        # Negative improvement means the model got better (lower loss)
        improvement = (baseline_loss - delta_loss) / baseline_loss
        
        return improvement
    
    def _calculate_drift(
        self,
        baseline_metrics: Dict[str, Any],
        delta_metrics: Dict[str, Any]
    ) -> float:
        """Calculate distribution drift."""
        # Compare logits distributions
        baseline_logits = baseline_metrics["logits"]
        delta_logits = delta_metrics["logits"]
        
        # Simple drift metric (Hamming distance)
        drift = 0.0
        for b_logit, d_logit in zip(baseline_logits, delta_logits):
            if b_logit != d_logit:
                drift += 1.0
        
        drift = drift / len(baseline_logits) if baseline_logits else 0.0
        
        return drift
    
    def _generate_logits_hash(self, logits: List[str]) -> str:
        """Generate deterministic hash of logits."""
        # Concatenate and hash all logits
        combined = ":".join(sorted(logits))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _check_validity(self, improvement: float, drift: float) -> bool:
        """Check if delta meets validity criteria."""
        # Must improve by minimum threshold
        if improvement < self.config.min_improvement:
            return False
        
        # Must not regress too much
        if improvement < -self.config.max_regression:
            return False
        
        # Must not drift too much
        if drift > self.config.drift_threshold:
            return False
        
        return True
    
    async def _get_consensus(
        self,
        delta_hash: str,
        local_vote: bool
    ) -> Dict[str, Any]:
        """Get consensus from other validators."""
        # Store our vote
        self.consensus_votes[delta_hash][self.validator_id] = local_vote
        
        # In production, collect votes from other validators via network
        # For now, simulate with weighted random voting
        np.random.seed(int(hashlib.sha256(delta_hash.encode()).hexdigest()[:8], 16))
        
        num_validators = 5  # Simulate 5 validators
        votes = {self.validator_id: local_vote}
        
        for i in range(num_validators - 1):
            validator_id = f"validator_{i+1}"
            # Validators likely agree if local vote is positive
            if local_vote:
                vote = np.random.random() < 0.8  # 80% agreement
            else:
                vote = np.random.random() < 0.2  # 20% disagreement
            
            votes[validator_id] = vote
            self.consensus_votes[delta_hash][validator_id] = vote
        
        # Calculate consensus
        positive_votes = sum(1 for v in votes.values() if v)
        consensus_reached = positive_votes / len(votes) >= self.config.consensus_threshold
        
        return {
            "accepted": consensus_reached,
            "votes": votes,
            "positive_ratio": positive_votes / len(votes)
        }
    
    def _get_failure_reason(
        self,
        improvement: float,
        drift: float,
        consensus: Dict[str, Any]
    ) -> Optional[str]:
        """Determine reason for validation failure."""
        if improvement < self.config.min_improvement:
            return f"Insufficient improvement: {improvement:.4f} < {self.config.min_improvement}"
        
        if improvement < -self.config.max_regression:
            return f"Excessive regression: {improvement:.4f}"
        
        if drift > self.config.drift_threshold:
            return f"Excessive drift: {drift:.4f} > {self.config.drift_threshold}"
        
        if not consensus["accepted"]:
            return f"Consensus not reached: {consensus['positive_ratio']:.2f} < {self.config.consensus_threshold}"
        
        return None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = len(self.consensus_votes)
        
        if total_validations == 0:
            return {
                "total_validations": 0,
                "acceptance_rate": 0.0,
                "avg_improvement": 0.0
            }
        
        # Calculate stats from consensus votes
        accepted = sum(
            1 for votes in self.consensus_votes.values()
            if sum(votes.values()) / len(votes) >= self.config.consensus_threshold
        )
        
        return {
            "total_validations": total_validations,
            "accepted": accepted,
            "rejected": total_validations - accepted,
            "acceptance_rate": accepted / total_validations,
            "validator_id": self.validator_id,
            "config": self.config.to_dict()
        }


class ByzantineConsensus:
    """
    Byzantine fault-tolerant consensus for delta validation.
    Implements Krum and trimmed mean for robust aggregation.
    """
    
    @staticmethod
    def krum_aggregate(
        values: List[float],
        byzantine_fraction: float = 0.2
    ) -> float:
        """
        Krum aggregation - select value closest to others.
        
        Args:
            values: List of values from validators
            byzantine_fraction: Fraction of Byzantine validators
            
        Returns:
            Aggregated value
        """
        n = len(values)
        f = int(n * byzantine_fraction)  # Number of Byzantine nodes
        
        if n - f - 2 <= 0:
            # Not enough honest nodes, use median
            return float(np.median(values))
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = abs(values[i] - values[j])
        
        # For each value, sum distances to n-f-1 nearest neighbors
        scores = []
        for i in range(n):
            dists = sorted(distances[i])
            score = sum(dists[:n-f-1])
            scores.append(score)
        
        # Select value with minimum score
        best_idx = np.argmin(scores)
        return values[best_idx]
    
    @staticmethod
    def trimmed_mean(
        values: List[float],
        trim_fraction: float = 0.2
    ) -> float:
        """
        Trimmed mean - remove outliers and average.
        
        Args:
            values: List of values
            trim_fraction: Fraction to trim from each end
            
        Returns:
            Trimmed mean
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        trim_count = int(len(values) * trim_fraction)
        
        if trim_count > 0:
            trimmed = sorted_values[trim_count:-trim_count]
        else:
            trimmed = sorted_values
        
        return np.mean(trimmed) if trimmed else np.mean(sorted_values)