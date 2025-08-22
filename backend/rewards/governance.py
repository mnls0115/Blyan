#!/usr/bin/env python3
"""
Reward System Governance
Implements on-chain governance for quality multipliers and objective metrics.
"""

import time
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityEvidence:
    """Evidence for quality/difficulty assessment."""
    claim_id: str
    metric_type: str  # 'test_loss', 'validation_accuracy', 'anomaly_rate', etc.
    baseline_value: float
    achieved_value: float
    improvement_pct: float
    validation_nodes: List[str]
    evidence_hash: str
    timestamp: float
    
    def verify_hash(self) -> bool:
        """Verify evidence integrity."""
        content = f"{self.claim_id}:{self.metric_type}:{self.baseline_value}:{self.achieved_value}"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.evidence_hash == expected_hash


class MultiplierGovernance:
    """
    Governs quality and difficulty multipliers through objective metrics.
    All decisions are recorded on-chain with evidence.
    """
    
    def __init__(self, chain_interface=None):
        """Initialize governance system."""
        self.chain = chain_interface  # Blockchain interface for recording
        
        # Objective metric thresholds (can be updated via governance vote)
        self.thresholds = {
            'difficulty': {
                'simple': {'test_loss_reduction': 0.01, 'multiplier': 1.0},
                'moderate': {'test_loss_reduction': 0.05, 'multiplier': 1.5},
                'hard': {'test_loss_reduction': 0.10, 'multiplier': 2.0},
                'breakthrough': {'test_loss_reduction': 0.20, 'multiplier': 3.0}
            },
            'quality': {
                'poor': {'validation_consensus': 0.5, 'multiplier': 0.5},
                'acceptable': {'validation_consensus': 0.7, 'multiplier': 0.8},
                'good': {'validation_consensus': 0.85, 'multiplier': 1.0},
                'excellent': {'validation_consensus': 0.95, 'multiplier': 1.5}
            },
            'applicability': {
                'narrow': {'affected_layers': 1, 'multiplier': 0.8},
                'standard': {'affected_layers': 3, 'multiplier': 1.0},
                'broad': {'affected_layers': 5, 'multiplier': 1.2}
            }
        }
        
        # Governance parameters
        self.min_validators = 3
        self.consensus_threshold = 0.66
        self.evidence_retention_days = 90
    
    def calculate_difficulty_multiplier(self, evidence: QualityEvidence) -> Tuple[float, str]:
        """
        Calculate difficulty multiplier from objective metrics.
        
        Returns:
            (multiplier, justification)
        """
        if evidence.metric_type != 'test_loss':
            return 1.5, "Default - no test loss data"
        
        loss_reduction = evidence.improvement_pct / 100
        
        # Find appropriate tier
        if loss_reduction >= 0.20:
            return 3.0, f"Breakthrough: {loss_reduction:.1%} loss reduction"
        elif loss_reduction >= 0.10:
            return 2.0, f"Hard: {loss_reduction:.1%} loss reduction"
        elif loss_reduction >= 0.05:
            return 1.5, f"Moderate: {loss_reduction:.1%} loss reduction"
        else:
            return 1.0, f"Simple: {loss_reduction:.1%} loss reduction"
    
    def calculate_quality_multiplier(self, 
                                   validation_results: List[Dict]) -> Tuple[float, str]:
        """
        Calculate quality multiplier from validation consensus.
        
        Args:
            validation_results: List of validation results from different nodes
        
        Returns:
            (multiplier, justification)
        """
        if len(validation_results) < self.min_validators:
            return 0.8, f"Insufficient validators ({len(validation_results)}/{self.min_validators})"
        
        # Calculate consensus score
        positive_validations = sum(
            1 for v in validation_results 
            if v.get('validated', False)
        )
        consensus_rate = positive_validations / len(validation_results)
        
        # Map to multiplier
        if consensus_rate >= 0.95:
            return 1.5, f"Excellent: {consensus_rate:.0%} consensus"
        elif consensus_rate >= 0.85:
            return 1.0, f"Good: {consensus_rate:.0%} consensus"
        elif consensus_rate >= 0.70:
            return 0.8, f"Acceptable: {consensus_rate:.0%} consensus"
        else:
            return 0.5, f"Poor: {consensus_rate:.0%} consensus"
    
    def record_governance_decision(self, 
                                  claim_id: str,
                                  multipliers: Dict[str, float],
                                  evidence: QualityEvidence,
                                  justification: str) -> str:
        """
        Record governance decision on-chain.
        
        Returns:
            Transaction hash
        """
        decision = {
            'claim_id': claim_id,
            'timestamp': time.time(),
            'multipliers': multipliers,
            'evidence_hash': evidence.evidence_hash,
            'justification': justification,
            'validators': evidence.validation_nodes
        }
        
        # Record on chain (would use actual blockchain interface)
        if self.chain:
            tx_hash = self.chain.record_governance_decision(decision)
        else:
            # Mock for testing
            tx_hash = hashlib.sha256(json.dumps(decision).encode()).hexdigest()[:16]
        
        logger.info(f"Recorded governance decision: {tx_hash}")
        return tx_hash
    
    def propose_threshold_update(self, 
                                category: str,
                                new_thresholds: Dict) -> str:
        """
        Propose update to multiplier thresholds (requires vote).
        """
        proposal = {
            'type': 'threshold_update',
            'category': category,
            'current': self.thresholds.get(category),
            'proposed': new_thresholds,
            'timestamp': time.time(),
            'voting_period_days': 7
        }
        
        # Would submit to governance contract
        proposal_id = hashlib.sha256(json.dumps(proposal).encode()).hexdigest()[:8]
        logger.info(f"Created governance proposal: {proposal_id}")
        
        return proposal_id


class BackpayDecayManager:
    """
    Manages progressive decay of old backpay claims to prevent infinite accumulation.
    """
    
    def __init__(self):
        """Initialize decay manager."""
        self.decay_schedule = [
            {'days': 7, 'retention': 1.0},    # Full value for 7 days
            {'days': 14, 'retention': 0.9},   # 90% value days 7-14
            {'days': 21, 'retention': 0.7},   # 70% value days 14-21
            {'days': 28, 'retention': 0.5},   # 50% value days 21-28
            {'days': 35, 'retention': 0.3},   # 30% value days 28-35
            {'days': 42, 'retention': 0.1},   # 10% value days 35-42
            {'days': float('inf'), 'retention': 0.0}  # Expired after 42 days
        ]
    
    def calculate_decayed_value(self, 
                               original_amount: float,
                               claim_timestamp: float) -> Tuple[float, float]:
        """
        Calculate decayed value of backpay claim.
        
        Returns:
            (decayed_amount, retention_rate)
        """
        age_days = (time.time() - claim_timestamp) / 86400
        
        for tier in self.decay_schedule:
            if age_days <= tier['days']:
                decayed = original_amount * tier['retention']
                return decayed, tier['retention']
        
        return 0.0, 0.0
    
    def apply_decay_to_queue(self, backpay_queue: List[Dict]) -> List[Dict]:
        """
        Apply decay to all items in backpay queue.
        
        Returns:
            Updated queue with decayed values
        """
        updated_queue = []
        total_decayed = 0
        
        for item in backpay_queue:
            original = item['amount_bly']
            decayed, retention = self.calculate_decayed_value(
                original,
                item['timestamp']
            )
            
            if decayed > 0:
                item['amount_bly'] = decayed
                item['decay_applied'] = 1 - retention
                updated_queue.append(item)
                total_decayed += (original - decayed)
            else:
                logger.info(f"Expired backpay claim: {item['request_id']}")
        
        if total_decayed > 0:
            logger.info(f"Decayed {total_decayed:.2f} BLY from backpay queue")
        
        return updated_queue


class DynamicBudgetScaler:
    """
    Dynamically scales budget based on network activity and token economics.
    """
    
    def __init__(self, base_daily_budget: float = 273_972):
        """Initialize budget scaler."""
        self.base_daily_budget = base_daily_budget
        self.total_supply = 1_000_000_000
        self.target_inflation = 0.10  # 10% annual
        
        # Activity metrics (would be pulled from chain)
        self.metrics_window_days = 7
        self.activity_history = []
    
    def calculate_dynamic_budget(self, 
                                current_supply: float,
                                network_metrics: Dict) -> float:
        """
        Calculate dynamic daily budget based on network activity.
        
        Args:
            current_supply: Current circulating supply
            network_metrics: Dict with 'daily_active_nodes', 'total_tokens_processed', etc.
        
        Returns:
            Adjusted daily budget
        """
        # Base budget from inflation target
        annual_inflation_budget = current_supply * self.target_inflation
        base_daily = annual_inflation_budget / 365
        
        # Activity multiplier (0.5x to 1.5x based on network growth)
        baseline_activity = 1000  # Baseline daily active nodes
        current_activity = network_metrics.get('daily_active_nodes', baseline_activity)
        activity_multiplier = min(1.5, max(0.5, current_activity / baseline_activity))
        
        # Congestion adjustment (reduce budget if backlog is huge)
        backlog_ratio = network_metrics.get('backpay_ratio', 0)  # backpay_queue / daily_budget
        if backlog_ratio > 3.0:
            congestion_discount = 0.8  # Reduce new allocation if backlog > 3 days
        elif backlog_ratio > 1.0:
            congestion_discount = 0.9
        else:
            congestion_discount = 1.0
        
        # Calculate final budget
        dynamic_budget = base_daily * activity_multiplier * congestion_discount
        
        # Apply bounds (50% to 150% of base)
        dynamic_budget = min(
            self.base_daily_budget * 1.5,
            max(self.base_daily_budget * 0.5, dynamic_budget)
        )
        
        logger.info(f"Dynamic budget: {dynamic_budget:.0f} BLY "
                   f"(activity: {activity_multiplier:.1f}x, "
                   f"congestion: {congestion_discount:.1f}x)")
        
        return dynamic_budget
    
    def auto_tune_epoch_size(self, 
                            transaction_rate: float,
                            budget_remaining: float) -> int:
        """
        Auto-tune distribution epoch size based on load.
        
        Returns:
            Epoch size in seconds (3600 for hourly, 86400 for daily, etc.)
        """
        # High transaction rate -> more frequent distribution
        if transaction_rate > 100:  # >100 claims/hour
            return 3600  # Hourly
        elif transaction_rate > 10:  # 10-100 claims/hour
            return 3600 * 6  # Every 6 hours
        else:  # <10 claims/hour
            return 86400  # Daily
        
        # Override if budget is critically low
        if budget_remaining < self.base_daily_budget * 0.1:
            return 3600  # Hourly when budget is tight


class FairnessOracle:
    """
    Monitors and enforces fairness across reward distribution.
    """
    
    def __init__(self):
        """Initialize fairness oracle."""
        self.concentration_threshold = 0.25  # Max 25% to single entity
        self.gini_threshold = 0.7  # Gini coefficient threshold
    
    def calculate_gini_coefficient(self, distributions: List[float]) -> float:
        """
        Calculate Gini coefficient for reward distribution.
        
        0 = perfect equality, 1 = perfect inequality
        """
        if not distributions or sum(distributions) == 0:
            return 0
        
        sorted_dist = sorted(distributions)
        n = len(sorted_dist)
        cumsum = np.cumsum(sorted_dist)
        
        return (2 * np.sum((np.arange(1, n+1)) * sorted_dist)) / (n * cumsum[-1]) - (n + 1) / n
    
    def check_concentration(self, 
                           reward_ledger: List[Dict]) -> Dict[str, float]:
        """
        Check if rewards are too concentrated.
        
        Returns:
            Dict of recipients exceeding concentration threshold
        """
        total_by_recipient = {}
        total_distributed = 0
        
        for entry in reward_ledger:
            recipient = entry['recipient']
            amount = entry['amount_bly']
            
            total_by_recipient[recipient] = total_by_recipient.get(recipient, 0) + amount
            total_distributed += amount
        
        # Find concentrated recipients
        concentrated = {}
        for recipient, amount in total_by_recipient.items():
            share = amount / total_distributed if total_distributed > 0 else 0
            if share > self.concentration_threshold:
                concentrated[recipient] = share
        
        return concentrated
    
    def recommend_adjustments(self, 
                             metrics: Dict) -> List[Dict]:
        """
        Recommend adjustments to improve fairness.
        """
        recommendations = []
        
        # Check Gini coefficient
        if 'gini' in metrics and metrics['gini'] > self.gini_threshold:
            recommendations.append({
                'type': 'high_inequality',
                'metric': 'gini_coefficient',
                'value': metrics['gini'],
                'threshold': self.gini_threshold,
                'action': 'Consider progressive multipliers or caps'
            })
        
        # Check concentration
        if 'concentrated_recipients' in metrics:
            for recipient, share in metrics['concentrated_recipients'].items():
                recommendations.append({
                    'type': 'concentration_risk',
                    'recipient': recipient[:8] + '...',  # Anonymize
                    'share': share,
                    'threshold': self.concentration_threshold,
                    'action': 'Apply cooldown or progressive reduction'
                })
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    
    # Test governance
    governance = MultiplierGovernance()
    
    evidence = QualityEvidence(
        claim_id="claim_123",
        metric_type="test_loss",
        baseline_value=2.5,
        achieved_value=2.0,
        improvement_pct=20.0,
        validation_nodes=["node1", "node2", "node3"],
        evidence_hash=hashlib.sha256(b"evidence").hexdigest(),
        timestamp=time.time()
    )
    
    difficulty_mult, justification = governance.calculate_difficulty_multiplier(evidence)
    print(f"Difficulty multiplier: {difficulty_mult}x ({justification})")
    
    # Test decay manager
    decay_mgr = BackpayDecayManager()
    
    # Simulate 10-day old claim
    old_timestamp = time.time() - (10 * 86400)
    decayed, retention = decay_mgr.calculate_decayed_value(1000.0, old_timestamp)
    print(f"10-day old claim: {decayed:.0f} BLY ({retention:.0%} retained)")
    
    # Test dynamic budget
    scaler = DynamicBudgetScaler()
    
    network_metrics = {
        'daily_active_nodes': 1500,
        'backpay_ratio': 0.5
    }
    
    dynamic_budget = scaler.calculate_dynamic_budget(100_000_000, network_metrics)
    print(f"Dynamic daily budget: {dynamic_budget:,.0f} BLY")
    
    # Test fairness oracle
    oracle = FairnessOracle()
    
    distributions = [100, 150, 200, 250, 300, 1000, 5000]  # Unequal distribution
    gini = oracle.calculate_gini_coefficient(distributions)
    print(f"Gini coefficient: {gini:.2f}")