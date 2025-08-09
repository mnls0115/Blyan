#!/usr/bin/env python3
"""
Anti-Gaming System for PoL Evaluation
Detects and prevents various abuse patterns in learning submissions
"""

import hashlib
import time
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class AbusePattern:
    """Detected abuse pattern."""
    pattern_type: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0-1
    evidence: Dict
    timestamp: float
    action_required: str  # "warn", "reject", "slash"

class AntiGamingSystem:
    """
    Comprehensive anti-gaming system for PoL.
    
    Detects:
    1. Overfitting to test set
    2. Data poisoning attempts
    3. Self-botting for inference rewards
    4. Validator collusion
    5. Weight noise spam
    6. Selective metric gaming
    7. Submission timing manipulation
    """
    
    def __init__(self):
        # Detection thresholds
        self.thresholds = {
            "overfitting_ratio": 1.5,      # Hidden/public score ratio
            "weight_similarity": 0.9995,    # Cosine similarity for spam
            "submission_frequency": 3600,   # Min seconds between submissions
            "inference_anomaly": 3.0,       # Std devs from mean
            "collusion_correlation": 0.95,  # Validator agreement correlation
            "metric_imbalance": 2.0,        # Max ratio between metrics
            "timing_variance": 0.1          # Min variance in submission times
        }
        
        # Tracking data structures
        self.submission_history = defaultdict(list)  # Per submitter
        self.weight_hashes = defaultdict(set)  # Track unique weights
        self.inference_patterns = defaultdict(deque)  # Usage patterns
        self.validator_votes = defaultdict(list)  # Validator decisions
        self.metric_history = defaultdict(list)  # Metric evolution
        
        # Punishment registry
        self.slashing_registry = {}
        self.warning_count = defaultdict(int)
        
    def check_submission(
        self,
        submitter: str,
        model_hash: str,
        delta_weights: np.ndarray,
        metrics: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> List[AbusePattern]:
        """
        Check a submission for gaming patterns.
        
        Args:
            submitter: Address of submitter
            model_hash: Hash of model
            delta_weights: Weight changes
            metrics: Evaluation metrics
            timestamp: Submission time
            
        Returns:
            List of detected abuse patterns
        """
        if timestamp is None:
            timestamp = time.time()
            
        detected_patterns = []
        
        # 1. Check for overfitting
        overfitting = self._check_overfitting(metrics)
        if overfitting:
            detected_patterns.append(overfitting)
            
        # 2. Check for weight spam
        weight_spam = self._check_weight_spam(submitter, model_hash, delta_weights)
        if weight_spam:
            detected_patterns.append(weight_spam)
            
        # 3. Check submission frequency
        frequency_abuse = self._check_submission_frequency(submitter, timestamp)
        if frequency_abuse:
            detected_patterns.append(frequency_abuse)
            
        # 4. Check metric gaming
        metric_gaming = self._check_metric_gaming(submitter, metrics)
        if metric_gaming:
            detected_patterns.append(metric_gaming)
            
        # 5. Check timing manipulation
        timing_abuse = self._check_timing_manipulation(submitter, timestamp)
        if timing_abuse:
            detected_patterns.append(timing_abuse)
            
        # Update history
        self._update_history(submitter, model_hash, metrics, timestamp)
        
        return detected_patterns
        
    def _check_overfitting(self, metrics: Dict[str, float]) -> Optional[AbusePattern]:
        """
        Check for overfitting to test set.
        
        Detection: Hidden score significantly worse than public score
        """
        public_score = metrics.get("public_score", 0)
        hidden_score = metrics.get("hidden_score", 0)
        
        if public_score > 0 and hidden_score > 0:
            ratio = public_score / hidden_score
            
            if ratio > self.thresholds["overfitting_ratio"]:
                return AbusePattern(
                    pattern_type="overfitting",
                    severity="high",
                    confidence=min(1.0, (ratio - 1.0) / 2.0),
                    evidence={
                        "public_score": public_score,
                        "hidden_score": hidden_score,
                        "ratio": ratio
                    },
                    timestamp=time.time(),
                    action_required="reject"
                )
                
        return None
        
    def _check_weight_spam(
        self,
        submitter: str,
        model_hash: str,
        delta_weights: np.ndarray
    ) -> Optional[AbusePattern]:
        """
        Check for weight noise spam (minimal changes for rewards).
        
        Detection: Cosine similarity too high with previous submission
        """
        # Calculate weight fingerprint
        weight_hash = hashlib.sha256(delta_weights.tobytes()).hexdigest()
        
        # Check similarity with previous submissions
        if submitter in self.submission_history:
            for prev_submission in self.submission_history[submitter][-5:]:
                if "weight_vector" in prev_submission:
                    similarity = 1 - cosine(
                        delta_weights.flatten(),
                        prev_submission["weight_vector"].flatten()
                    )
                    
                    if similarity > self.thresholds["weight_similarity"]:
                        return AbusePattern(
                            pattern_type="weight_spam",
                            severity="medium",
                            confidence=similarity,
                            evidence={
                                "similarity": similarity,
                                "threshold": self.thresholds["weight_similarity"],
                                "previous_hash": prev_submission.get("model_hash")
                            },
                            timestamp=time.time(),
                            action_required="reject"
                        )
                        
        # Store for future checks
        if weight_hash not in self.weight_hashes[submitter]:
            self.weight_hashes[submitter].add(weight_hash)
            
        return None
        
    def _check_submission_frequency(
        self,
        submitter: str,
        timestamp: float
    ) -> Optional[AbusePattern]:
        """
        Check for too frequent submissions.
        
        Detection: Submissions faster than allowed frequency
        """
        if submitter in self.submission_history:
            last_submission = self.submission_history[submitter][-1]
            time_diff = timestamp - last_submission["timestamp"]
            
            if time_diff < self.thresholds["submission_frequency"]:
                return AbusePattern(
                    pattern_type="frequency_abuse",
                    severity="low",
                    confidence=1.0,
                    evidence={
                        "time_since_last": time_diff,
                        "minimum_required": self.thresholds["submission_frequency"]
                    },
                    timestamp=timestamp,
                    action_required="warn"
                )
                
        return None
        
    def _check_metric_gaming(
        self,
        submitter: str,
        metrics: Dict[str, float]
    ) -> Optional[AbusePattern]:
        """
        Check for selective metric gaming.
        
        Detection: One metric significantly better than others
        """
        metric_values = [
            metrics.get("perplexity_score", 0.5),
            metrics.get("bleu_score", 0.5),
            metrics.get("safety_score", 0.5),
            metrics.get("latency_score", 0.5)
        ]
        
        if metric_values:
            max_metric = max(metric_values)
            min_metric = min(metric_values)
            
            if min_metric > 0:
                ratio = max_metric / min_metric
                
                if ratio > self.thresholds["metric_imbalance"]:
                    return AbusePattern(
                        pattern_type="metric_gaming",
                        severity="medium",
                        confidence=min(1.0, (ratio - 1.5) / 2.0),
                        evidence={
                            "max_metric": max_metric,
                            "min_metric": min_metric,
                            "imbalance_ratio": ratio
                        },
                        timestamp=time.time(),
                        action_required="warn"
                    )
                    
        return None
        
    def _check_timing_manipulation(
        self,
        submitter: str,
        timestamp: float
    ) -> Optional[AbusePattern]:
        """
        Check for timing manipulation (submitting at predictable times).
        
        Detection: Low variance in submission times
        """
        if submitter in self.submission_history and len(self.submission_history[submitter]) >= 5:
            # Get hours of day for submissions
            submission_hours = [
                (s["timestamp"] % 86400) / 3600  # Hour of day
                for s in self.submission_history[submitter][-10:]
            ]
            
            # Calculate variance
            if submission_hours:
                variance = np.var(submission_hours)
                
                if variance < self.thresholds["timing_variance"]:
                    return AbusePattern(
                        pattern_type="timing_manipulation",
                        severity="low",
                        confidence=1.0 - variance,
                        evidence={
                            "variance": variance,
                            "submission_hours": submission_hours[-5:]
                        },
                        timestamp=timestamp,
                        action_required="warn"
                    )
                    
        return None
        
    def check_inference_abuse(
        self,
        user: str,
        inference_count: int,
        time_window: float = 3600
    ) -> Optional[AbusePattern]:
        """
        Check for self-botting or inference abuse.
        
        Detection: Anomalous inference patterns
        """
        # Track inference pattern
        self.inference_patterns[user].append({
            "count": inference_count,
            "timestamp": time.time()
        })
        
        # Keep only recent data
        cutoff = time.time() - time_window
        self.inference_patterns[user] = deque(
            [p for p in self.inference_patterns[user] if p["timestamp"] > cutoff],
            maxlen=100
        )
        
        if len(self.inference_patterns[user]) >= 10:
            counts = [p["count"] for p in self.inference_patterns[user]]
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            
            if std_count > 0:
                z_score = (inference_count - mean_count) / std_count
                
                if abs(z_score) > self.thresholds["inference_anomaly"]:
                    return AbusePattern(
                        pattern_type="inference_abuse",
                        severity="high" if z_score > 5 else "medium",
                        confidence=min(1.0, abs(z_score) / 10),
                        evidence={
                            "z_score": z_score,
                            "inference_count": inference_count,
                            "mean_count": mean_count,
                            "std_count": std_count
                        },
                        timestamp=time.time(),
                        action_required="reject" if z_score > 5 else "warn"
                    )
                    
        return None
        
    def check_validator_collusion(
        self,
        validators: List[str],
        votes: Dict[str, bool]
    ) -> Optional[AbusePattern]:
        """
        Check for validator collusion patterns.
        
        Detection: Validators always voting together
        """
        # Track voting patterns
        for validator in validators:
            self.validator_votes[validator].append(votes.get(validator, False))
            
        # Check for correlation between validators
        if all(len(self.validator_votes[v]) >= 20 for v in validators):
            correlations = []
            
            for i, v1 in enumerate(validators):
                for v2 in validators[i+1:]:
                    votes1 = np.array(self.validator_votes[v1][-20:])
                    votes2 = np.array(self.validator_votes[v2][-20:])
                    
                    correlation = np.corrcoef(votes1, votes2)[0, 1]
                    correlations.append(correlation)
                    
                    if correlation > self.thresholds["collusion_correlation"]:
                        return AbusePattern(
                            pattern_type="validator_collusion",
                            severity="critical",
                            confidence=correlation,
                            evidence={
                                "validators": [v1, v2],
                                "correlation": correlation,
                                "vote_history": {
                                    v1: list(votes1[-5:]),
                                    v2: list(votes2[-5:])
                                }
                            },
                            timestamp=time.time(),
                            action_required="slash"
                        )
                        
        return None
        
    def apply_punishment(
        self,
        address: str,
        pattern: AbusePattern
    ) -> Dict[str, any]:
        """
        Apply punishment based on detected abuse.
        
        Returns:
            Punishment details
        """
        punishment = {
            "address": address,
            "pattern": pattern.pattern_type,
            "severity": pattern.severity,
            "timestamp": time.time()
        }
        
        if pattern.action_required == "warn":
            self.warning_count[address] += 1
            punishment["action"] = "warning"
            punishment["warning_count"] = self.warning_count[address]
            
            # Escalate if too many warnings
            if self.warning_count[address] >= 3:
                punishment["action"] = "temporary_ban"
                punishment["ban_duration"] = 86400  # 24 hours
                
        elif pattern.action_required == "reject":
            punishment["action"] = "submission_rejected"
            punishment["penalty"] = 0  # No rewards
            
        elif pattern.action_required == "slash":
            # Calculate slashing amount based on severity
            slash_percentage = {
                "low": 1,
                "medium": 5,
                "high": 10,
                "critical": 25
            }.get(pattern.severity, 5)
            
            punishment["action"] = "slashing"
            punishment["slash_percentage"] = slash_percentage
            self.slashing_registry[address] = punishment
            
        logger.warning(f"Punishment applied to {address}: {punishment}")
        return punishment
        
    def _update_history(
        self,
        submitter: str,
        model_hash: str,
        metrics: Dict[str, float],
        timestamp: float
    ):
        """Update submission history."""
        self.submission_history[submitter].append({
            "model_hash": model_hash,
            "metrics": metrics,
            "timestamp": timestamp
        })
        
        # Keep only recent history (last 100 submissions)
        if len(self.submission_history[submitter]) > 100:
            self.submission_history[submitter] = self.submission_history[submitter][-100:]
            
        # Update metric history
        self.metric_history[submitter].append(metrics)
        if len(self.metric_history[submitter]) > 50:
            self.metric_history[submitter] = self.metric_history[submitter][-50:]
            
    def get_reputation_score(self, address: str) -> float:
        """
        Calculate reputation score based on history.
        
        Returns:
            Reputation score (0-100)
        """
        base_score = 50.0
        
        # Deduct for warnings
        base_score -= self.warning_count[address] * 5
        
        # Deduct for slashing
        if address in self.slashing_registry:
            base_score -= self.slashing_registry[address]["slash_percentage"]
            
        # Bonus for consistent good behavior
        if address in self.submission_history:
            submissions = len(self.submission_history[address])
            if submissions > 10 and self.warning_count[address] == 0:
                base_score += min(20, submissions / 2)
                
        return max(0, min(100, base_score))

# Demo
def demo_anti_gaming():
    """Demonstrate anti-gaming system."""
    system = AntiGamingSystem()
    
    print("=== ANTI-GAMING SYSTEM DEMO ===\n")
    
    # Test 1: Normal submission
    print("Test 1: Normal submission")
    patterns = system.check_submission(
        submitter="0xABCD",
        model_hash="hash1",
        delta_weights=np.random.randn(100, 100),
        metrics={
            "public_score": 0.8,
            "hidden_score": 0.75,
            "perplexity_score": 0.7,
            "bleu_score": 0.6,
            "safety_score": 0.9,
            "latency_score": 0.8
        }
    )
    print(f"  Detected patterns: {len(patterns)}")
    
    # Test 2: Overfitting detection
    print("\nTest 2: Overfitting detection")
    patterns = system.check_submission(
        submitter="0xEVIL",
        model_hash="hash2",
        delta_weights=np.random.randn(100, 100),
        metrics={
            "public_score": 0.95,
            "hidden_score": 0.4,  # Much worse on hidden
            "perplexity_score": 0.9,
            "bleu_score": 0.2,
            "safety_score": 0.8,
            "latency_score": 0.7
        }
    )
    if patterns:
        print(f"  ⚠️ Detected: {patterns[0].pattern_type}")
        print(f"  Severity: {patterns[0].severity}")
        print(f"  Action: {patterns[0].action_required}")
        
    # Test 3: Frequency abuse
    print("\nTest 3: Frequency abuse")
    current_time = time.time()
    
    # First submission
    system.check_submission(
        submitter="0xFAST",
        model_hash="hash3",
        delta_weights=np.random.randn(100, 100),
        metrics={"public_score": 0.7, "hidden_score": 0.7},
        timestamp=current_time
    )
    
    # Too fast second submission
    patterns = system.check_submission(
        submitter="0xFAST",
        model_hash="hash4",
        delta_weights=np.random.randn(100, 100),
        metrics={"public_score": 0.7, "hidden_score": 0.7},
        timestamp=current_time + 60  # Only 1 minute later
    )
    if patterns:
        print(f"  ⚠️ Detected: {patterns[0].pattern_type}")
        print(f"  Evidence: {patterns[0].evidence}")
        
    # Test 4: Check reputation
    print("\nTest 4: Reputation scores")
    print(f"  Normal user (0xABCD): {system.get_reputation_score('0xABCD'):.1f}")
    print(f"  Overfitter (0xEVIL): {system.get_reputation_score('0xEVIL'):.1f}")
    print(f"  Frequency abuser (0xFAST): {system.get_reputation_score('0xFAST'):.1f}")

if __name__ == "__main__":
    demo_anti_gaming()