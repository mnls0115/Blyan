#!/usr/bin/env python3
"""
Reward System Safety Mechanisms
Implements sybil resistance, emergency controls, and audit hooks.
"""

import time
import hashlib
import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class EmergencyState(Enum):
    """Emergency system states."""
    NORMAL = "normal"
    SLOWMODE = "slowmode"  # 10x slower distribution
    PAUSED = "paused"      # No distributions
    READONLY = "readonly"   # View only, no changes


@dataclass
class ReputationScore:
    """Reputation score for sybil resistance."""
    node_id: str
    pol_contributions: int  # Proof-of-Learning contributions
    validation_accuracy: float
    stake_amount: float
    account_age_days: int
    violation_count: int
    last_updated: float
    
    def calculate_score(self) -> float:
        """Calculate composite reputation score (0-100)."""
        # Base score from contributions
        contribution_score = min(30, self.pol_contributions * 0.1)
        
        # Accuracy bonus (up to 20 points)
        accuracy_score = self.validation_accuracy * 20
        
        # Stake bonus (logarithmic, up to 20 points)
        import math
        stake_score = min(20, math.log10(max(1, self.stake_amount)) * 5)
        
        # Age bonus (up to 20 points)
        age_score = min(20, self.account_age_days / 30)
        
        # Behavior bonus (up to 10 points)
        behavior_score = max(0, 10 - self.violation_count * 2)
        
        total = contribution_score + accuracy_score + stake_score + age_score + behavior_score
        
        return min(100, max(0, total))


class SybilResistance:
    """
    Implements sybil resistance through reputation and proof-of-learning.
    """
    
    def __init__(self, min_reputation: float = 20.0, min_stake: float = 100.0):
        """Initialize sybil resistance."""
        self.min_reputation = min_reputation
        self.min_stake = min_stake
        self.reputation_cache: Dict[str, ReputationScore] = {}
        self.suspicious_patterns: List[Dict] = []
        
        # Identity clustering detection
        self.identity_clusters: Dict[str, Set[str]] = {}
        self.cluster_threshold = 0.8  # Similarity threshold
    
    def verify_identity(self, node_id: str, claim_data: Dict) -> Tuple[bool, str]:
        """
        Verify node identity and check for sybil behavior.
        
        Returns:
            (is_valid, reason)
        """
        # Get or create reputation
        reputation = self._get_reputation(node_id)
        
        # Check minimum reputation
        rep_score = reputation.calculate_score()
        if rep_score < self.min_reputation:
            return False, f"Insufficient reputation: {rep_score:.1f} < {self.min_reputation}"
        
        # Check minimum stake
        if reputation.stake_amount < self.min_stake:
            return False, f"Insufficient stake: {reputation.stake_amount} < {self.min_stake}"
        
        # Check for suspicious patterns
        if self._detect_sybil_pattern(node_id, claim_data):
            return False, "Suspicious activity pattern detected"
        
        # Check identity clustering
        cluster = self._find_identity_cluster(node_id)
        if len(cluster) > 3:  # More than 3 similar identities
            return False, f"Part of suspicious identity cluster ({len(cluster)} nodes)"
        
        return True, "Identity verified"
    
    def _get_reputation(self, node_id: str) -> ReputationScore:
        """Get or initialize reputation score."""
        if node_id not in self.reputation_cache:
            # Would load from blockchain/database
            self.reputation_cache[node_id] = ReputationScore(
                node_id=node_id,
                pol_contributions=0,
                validation_accuracy=0.0,
                stake_amount=0.0,
                account_age_days=0,
                violation_count=0,
                last_updated=time.time()
            )
        
        return self.reputation_cache[node_id]
    
    def _detect_sybil_pattern(self, node_id: str, claim_data: Dict) -> bool:
        """Detect suspicious sybil patterns."""
        patterns = []
        
        # Pattern 1: Rapid-fire claims
        claim_time = claim_data.get('timestamp', time.time())
        recent_claims = self._get_recent_claims(node_id, window_seconds=60)
        if len(recent_claims) > 10:
            patterns.append("rapid_claims")
        
        # Pattern 2: Identical improvements across nodes
        improvement = claim_data.get('improvement_pct', 0)
        if self._check_duplicate_improvements(improvement, window_seconds=300):
            patterns.append("duplicate_improvements")
        
        # Pattern 3: Coordinated timing
        if self._check_coordinated_timing(node_id, claim_time):
            patterns.append("coordinated_timing")
        
        if patterns:
            self.suspicious_patterns.append({
                'node_id': node_id,
                'patterns': patterns,
                'timestamp': time.time()
            })
            logger.warning(f"Sybil patterns detected for {node_id}: {patterns}")
            return True
        
        return False
    
    def _find_identity_cluster(self, node_id: str) -> Set[str]:
        """Find cluster of similar identities."""
        # Would use more sophisticated clustering in production
        if node_id in self.identity_clusters:
            return self.identity_clusters[node_id]
        
        # Simple clustering based on behavior similarity
        cluster = {node_id}
        # Add logic to find similar nodes
        
        return cluster
    
    def _get_recent_claims(self, node_id: str, window_seconds: int) -> List[Dict]:
        """Get recent claims from node (would query ledger)."""
        # Placeholder - would query actual ledger
        return []
    
    def _check_duplicate_improvements(self, improvement: float, window_seconds: int) -> bool:
        """Check if many nodes report identical improvements."""
        # Placeholder - would check across recent claims
        return False
    
    def _check_coordinated_timing(self, node_id: str, claim_time: float) -> bool:
        """Check for coordinated claim timing."""
        # Placeholder - would analyze timing patterns
        return False
    
    def update_reputation(self, node_id: str, action: str, value: float):
        """Update reputation based on actions."""
        reputation = self._get_reputation(node_id)
        
        if action == "pol_contribution":
            reputation.pol_contributions += 1
        elif action == "validation_success":
            # Update accuracy with exponential moving average
            alpha = 0.1
            reputation.validation_accuracy = (
                alpha * value + (1 - alpha) * reputation.validation_accuracy
            )
        elif action == "stake_added":
            reputation.stake_amount += value
        elif action == "violation":
            reputation.violation_count += 1
        
        reputation.last_updated = time.time()


class EmergencyController:
    """
    Emergency control system for the reward system.
    Can pause, slow, or halt distributions in case of issues.
    """
    
    def __init__(self):
        """Initialize emergency controller."""
        self.state = EmergencyState.NORMAL
        self.state_lock = threading.RLock()
        
        # Emergency triggers
        self.triggers = {
            'exploit_detected': False,
            'budget_overflow': False,
            'governance_override': False,
            'audit_in_progress': False
        }
        
        # Authorized pausers (would be multisig in production)
        self.authorized_pausers: Set[str] = set()
        
        # State history for audit
        self.state_history: List[Dict] = []
        
        # Auto-recovery settings
        self.auto_recovery_enabled = True
        self.pause_timeout_seconds = 3600  # 1 hour max pause
        self.pause_start_time: Optional[float] = None
    
    def request_pause(self, 
                     requester: str, 
                     reason: str,
                     duration_seconds: int = 3600) -> Tuple[bool, str]:
        """
        Request emergency pause.
        
        Returns:
            (success, message)
        """
        with self.state_lock:
            # Verify authorization
            if requester not in self.authorized_pausers:
                return False, "Unauthorized pause request"
            
            # Check if already paused
            if self.state == EmergencyState.PAUSED:
                return False, "System already paused"
            
            # Set pause state
            self.state = EmergencyState.PAUSED
            self.pause_start_time = time.time()
            
            # Record in history
            self.state_history.append({
                'action': 'pause',
                'requester': requester,
                'reason': reason,
                'duration': duration_seconds,
                'timestamp': time.time()
            })
            
            logger.critical(f"EMERGENCY PAUSE by {requester}: {reason}")
            
            # Schedule auto-recovery
            if self.auto_recovery_enabled:
                threading.Timer(duration_seconds, self._auto_recover).start()
            
            return True, f"System paused for {duration_seconds} seconds"
    
    def request_slowmode(self, requester: str, factor: int = 10) -> Tuple[bool, str]:
        """
        Request slowmode (reduced distribution rate).
        
        Returns:
            (success, message)
        """
        with self.state_lock:
            if requester not in self.authorized_pausers:
                return False, "Unauthorized slowmode request"
            
            self.state = EmergencyState.SLOWMODE
            
            self.state_history.append({
                'action': 'slowmode',
                'requester': requester,
                'factor': factor,
                'timestamp': time.time()
            })
            
            logger.warning(f"SLOWMODE activated by {requester} ({factor}x slower)")
            
            return True, f"Slowmode activated ({factor}x slower)"
    
    def resume_normal(self, requester: str) -> Tuple[bool, str]:
        """Resume normal operations."""
        with self.state_lock:
            if requester not in self.authorized_pausers:
                return False, "Unauthorized resume request"
            
            previous_state = self.state
            self.state = EmergencyState.NORMAL
            self.pause_start_time = None
            
            self.state_history.append({
                'action': 'resume',
                'requester': requester,
                'previous_state': previous_state.value,
                'timestamp': time.time()
            })
            
            logger.info(f"Normal operations resumed by {requester}")
            
            return True, "Normal operations resumed"
    
    def _auto_recover(self):
        """Automatically recover from pause."""
        with self.state_lock:
            if self.state == EmergencyState.PAUSED:
                self.state = EmergencyState.NORMAL
                self.pause_start_time = None
                
                self.state_history.append({
                    'action': 'auto_recovery',
                    'timestamp': time.time()
                })
                
                logger.info("Auto-recovery: Normal operations resumed")
    
    def check_can_distribute(self) -> Tuple[bool, str]:
        """
        Check if distributions are allowed.
        
        Returns:
            (can_distribute, reason)
        """
        with self.state_lock:
            if self.state == EmergencyState.PAUSED:
                return False, "System is paused"
            
            if self.state == EmergencyState.READONLY:
                return False, "System is in read-only mode"
            
            if self.state == EmergencyState.SLOWMODE:
                # Still allow, but caller should slow down
                return True, "Slowmode active - reduce rate"
            
            return True, "Normal operations"
    
    def get_slowdown_factor(self) -> int:
        """Get current slowdown factor."""
        if self.state == EmergencyState.SLOWMODE:
            return 10  # 10x slower
        return 1  # Normal speed
    
    def add_authorized_pauser(self, address: str, added_by: str):
        """Add authorized pauser (requires existing pauser)."""
        if added_by in self.authorized_pausers or not self.authorized_pausers:
            self.authorized_pausers.add(address)
            logger.info(f"Added authorized pauser: {address}")


class AuditInterface:
    """
    External audit interface for independent verification.
    """
    
    def __init__(self, ledger_path: str, state_path: str):
        """Initialize audit interface."""
        self.ledger_path = ledger_path
        self.state_path = state_path
        self.audit_reports: List[Dict] = []
    
    def create_audit_snapshot(self) -> str:
        """
        Create immutable audit snapshot.
        
        Returns:
            Snapshot hash
        """
        snapshot = {
            'timestamp': time.time(),
            'ledger_hash': self._hash_file(self.ledger_path),
            'state_hash': self._hash_file(self.state_path),
            'metadata': {
                'version': '1.0.0',
                'network': 'blyan',
                'audit_interface': 'v1'
            }
        }
        
        snapshot_data = json.dumps(snapshot, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_data.encode()).hexdigest()
        
        # Store snapshot
        snapshot['hash'] = snapshot_hash
        self.audit_reports.append(snapshot)
        
        logger.info(f"Created audit snapshot: {snapshot_hash[:16]}...")
        
        return snapshot_hash
    
    def _hash_file(self, filepath: str) -> str:
        """Calculate hash of file contents."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return "file_not_found"
    
    def verify_distribution(self, 
                          claim_id: str,
                          expected_amount: float) -> Tuple[bool, Dict]:
        """
        Verify a specific distribution.
        
        Returns:
            (is_valid, details)
        """
        # Would query ledger and verify
        details = {
            'claim_id': claim_id,
            'expected': expected_amount,
            'actual': 0,
            'verified': False,
            'timestamp': time.time()
        }
        
        # Placeholder verification logic
        # In production, would check ledger, calculate from formulas, etc.
        
        return False, details
    
    def generate_audit_report(self, 
                            start_time: float,
                            end_time: float) -> Dict:
        """Generate comprehensive audit report."""
        report = {
            'period': {
                'start': start_time,
                'end': end_time,
                'duration_days': (end_time - start_time) / 86400
            },
            'distributions': {
                'total_claims': 0,
                'total_bly': 0,
                'by_type': {}
            },
            'compliance': {
                'formula_violations': [],
                'duplicate_payments': [],
                'unauthorized_changes': []
            },
            'recommendations': []
        }
        
        # Would analyze ledger and identify issues
        
        return report
    
    def export_for_external_audit(self, output_dir: str):
        """Export all data for external audit."""
        import shutil
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy relevant files
        files_to_export = [
            self.ledger_path,
            self.state_path,
            'config/reward_policy.yaml'
        ]
        
        for filepath in files_to_export:
            if Path(filepath).exists():
                shutil.copy2(filepath, output_path)
        
        # Create audit manifest
        manifest = {
            'export_time': time.time(),
            'files': files_to_export,
            'snapshots': self.audit_reports,
            'version': '1.0.0'
        }
        
        with open(output_path / 'audit_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Exported audit data to {output_dir}")


class CircuitBreaker:
    """
    Circuit breaker pattern for automatic issue detection and recovery.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed=normal, open=broken, half_open=testing
        
    def record_success(self):
        """Record successful operation."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker: Recovered to normal")
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit breaker: OPEN after {self.failure_count} failures")
    
    def can_proceed(self) -> bool:
        """Check if operations can proceed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker: Testing recovery")
                return True
            return False
        
        if self.state == "half_open":
            return True
        
        return False


if __name__ == "__main__":
    # Example usage
    
    # Test sybil resistance
    sybil = SybilResistance()
    
    # Create test identity
    sybil.update_reputation("node_123", "pol_contribution", 1)
    sybil.update_reputation("node_123", "stake_added", 150)
    
    valid, reason = sybil.verify_identity("node_123", {'timestamp': time.time()})
    print(f"Identity verification: {valid} ({reason})")
    
    # Test emergency controller
    emergency = EmergencyController()
    emergency.add_authorized_pauser("admin_1", "init")
    
    success, msg = emergency.request_pause("admin_1", "Testing emergency pause", 30)
    print(f"Pause request: {success} - {msg}")
    
    can_distribute, reason = emergency.check_can_distribute()
    print(f"Can distribute: {can_distribute} ({reason})")
    
    # Test audit interface
    audit = AuditInterface("data/ledger.json", "data/state.json")
    snapshot_hash = audit.create_audit_snapshot()
    print(f"Audit snapshot: {snapshot_hash[:16]}...")
    
    # Test circuit breaker
    breaker = CircuitBreaker()
    for i in range(6):
        breaker.record_failure()
        print(f"Can proceed after failure {i+1}: {breaker.can_proceed()}")