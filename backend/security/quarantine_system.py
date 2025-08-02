#!/usr/bin/env python3
"""Quarantine and isolation system for malicious nodes and data."""

import json
import time
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading
from collections import defaultdict

class QuarantineLevel(Enum):
    """Quarantine severity levels."""
    MONITORING = "monitoring"
    RESTRICTED = "restricted"
    ISOLATED = "isolated"
    BANNED = "banned"

@dataclass
class QuarantineEntry:
    """Entry in quarantine database."""
    node_id: str
    reason: str
    level: QuarantineLevel
    timestamp: float
    duration: Optional[float]  # None for permanent
    evidence: Dict[str, Any]
    reporter_nodes: List[str]
    auto_release: bool = False

@dataclass
class SuspiciousActivity:
    """Record of suspicious activity."""
    node_id: str
    activity_type: str
    severity: float  # 0.0 to 1.0
    timestamp: float
    details: Dict[str, Any]

class QuarantineManager:
    """Manages quarantined nodes and suspicious activities."""
    
    def __init__(self, quarantine_dir: Path):
        self.quarantine_dir = quarantine_dir
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.quarantined_nodes: Dict[str, QuarantineEntry] = {}
        self.suspicious_activities: List[SuspiciousActivity] = []
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)  # Default trust score
        
        self.quarantine_thresholds = {
            QuarantineLevel.MONITORING: 0.3,
            QuarantineLevel.RESTRICTED: 0.5,
            QuarantineLevel.ISOLATED: 0.7,
            QuarantineLevel.BANNED: 0.9
        }
        
        self.load_quarantine_data()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_entries, daemon=True)
        self.cleanup_thread.start()
    
    def report_suspicious_activity(
        self, 
        node_id: str, 
        activity_type: str, 
        severity: float,
        details: Dict[str, Any],
        reporter_node: str = "system"
    ) -> bool:
        """Report suspicious activity from a node."""
        
        activity = SuspiciousActivity(
            node_id=node_id,
            activity_type=activity_type,
            severity=severity,
            timestamp=time.time(),
            details=details
        )
        
        self.suspicious_activities.append(activity)
        
        # Update trust score
        trust_penalty = severity * 0.2  # Max 0.2 penalty per incident
        self.trust_scores[node_id] = max(0.0, self.trust_scores[node_id] - trust_penalty)
        
        print(f"⚠️ Suspicious activity reported: {node_id} - {activity_type} (severity: {severity:.2f})")
        print(f"   Trust score updated: {self.trust_scores[node_id]:.2f}")
        
        # Auto-quarantine if trust score drops below threshold
        return self._check_auto_quarantine(node_id)
    
    def quarantine_node(
        self, 
        node_id: str, 
        reason: str, 
        level: QuarantineLevel,
        duration: Optional[float] = None,
        evidence: Dict[str, Any] = None,
        reporter_nodes: List[str] = None
    ) -> bool:
        """Quarantine a node."""
        
        if evidence is None:
            evidence = {}
        if reporter_nodes is None:
            reporter_nodes = ["system"]
        
        entry = QuarantineEntry(
            node_id=node_id,
            reason=reason,
            level=level,
            timestamp=time.time(),
            duration=duration,
            evidence=evidence,
            reporter_nodes=reporter_nodes,
            auto_release=duration is not None
        )
        
        self.quarantined_nodes[node_id] = entry
        self.save_quarantine_data()
        
        print(f"🚨 Node quarantined: {node_id}")
        print(f"   Level: {level.value}")
        print(f"   Reason: {reason}")
        print(f"   Duration: {duration/3600:.1f} hours" if duration else "   Duration: Permanent")
        
        return True
    
    def is_node_quarantined(self, node_id: str) -> Tuple[bool, Optional[QuarantineLevel]]:
        """Check if a node is quarantined."""
        if node_id in self.quarantined_nodes:
            entry = self.quarantined_nodes[node_id]
            
            # Check if quarantine has expired
            if entry.auto_release and entry.duration:
                if time.time() - entry.timestamp > entry.duration:
                    self.release_node(node_id, "Auto-release after expiration")
                    return False, None
            
            return True, entry.level
        
        return False, None
    
    def can_node_perform_action(self, node_id: str, action: str) -> Tuple[bool, str]:
        """Check if a quarantined node can perform a specific action."""
        is_quarantined, level = self.is_node_quarantined(node_id)
        
        if not is_quarantined:
            return True, "Node not quarantined"
        
        # Define allowed actions for each quarantine level
        allowed_actions = {
            QuarantineLevel.MONITORING: {
                "read_chain", "submit_small_expert", "peer_communication"
            },
            QuarantineLevel.RESTRICTED: {
                "read_chain", "peer_communication"
            },
            QuarantineLevel.ISOLATED: {
                "read_chain"
            },
            QuarantineLevel.BANNED: set()
        }
        
        if action in allowed_actions.get(level, set()):
            return True, f"Action allowed under {level.value} quarantine"
        else:
            return False, f"Action blocked by {level.value} quarantine"
    
    def release_node(self, node_id: str, reason: str) -> bool:
        """Release a node from quarantine."""
        if node_id not in self.quarantined_nodes:
            return False
        
        del self.quarantined_nodes[node_id]
        
        # Partially restore trust score
        if node_id in self.trust_scores:
            self.trust_scores[node_id] = min(1.0, self.trust_scores[node_id] + 0.1)
        
        self.save_quarantine_data()
        
        print(f"✅ Node released from quarantine: {node_id}")
        print(f"   Reason: {reason}")
        print(f"   New trust score: {self.trust_scores[node_id]:.2f}")
        
        return True
    
    def get_quarantine_status(self, node_id: str) -> Dict[str, Any]:
        """Get detailed quarantine status for a node."""
        if node_id not in self.quarantined_nodes:
            return {
                "quarantined": False,
                "trust_score": self.trust_scores[node_id],
                "recent_activities": self._get_recent_activities(node_id)
            }
        
        entry = self.quarantined_nodes[node_id]
        remaining_time = None
        
        if entry.auto_release and entry.duration:
            elapsed = time.time() - entry.timestamp
            remaining_time = max(0, entry.duration - elapsed)
        
        return {
            "quarantined": True,
            "level": entry.level.value,
            "reason": entry.reason,
            "quarantined_since": entry.timestamp,
            "remaining_time": remaining_time,
            "evidence": entry.evidence,
            "reporter_nodes": entry.reporter_nodes,
            "trust_score": self.trust_scores[node_id],
            "recent_activities": self._get_recent_activities(node_id)
        }
    
    def get_network_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive network security report."""
        total_nodes = len(set(self.trust_scores.keys()) | set(self.quarantined_nodes.keys()))
        quarantined_count = len(self.quarantined_nodes)
        
        # Categorize by trust score
        high_trust = sum(1 for score in self.trust_scores.values() if score >= 0.8)
        medium_trust = sum(1 for score in self.trust_scores.values() if 0.3 <= score < 0.8)
        low_trust = sum(1 for score in self.trust_scores.values() if score < 0.3)
        
        # Recent activity analysis
        recent_cutoff = time.time() - 24 * 3600  # Last 24 hours
        recent_activities = [a for a in self.suspicious_activities if a.timestamp > recent_cutoff]
        
        activity_by_type = defaultdict(int)
        for activity in recent_activities:
            activity_by_type[activity.activity_type] += 1
        
        return {
            "timestamp": time.time(),
            "total_nodes": total_nodes,
            "quarantined_nodes": quarantined_count,
            "quarantine_rate": quarantined_count / max(1, total_nodes),
            "trust_distribution": {
                "high_trust": high_trust,
                "medium_trust": medium_trust,
                "low_trust": low_trust
            },
            "recent_suspicious_activities": len(recent_activities),
            "activity_breakdown": dict(activity_by_type),
            "quarantine_levels": {
                level.value: sum(1 for entry in self.quarantined_nodes.values() if entry.level == level)
                for level in QuarantineLevel
            }
        }
    
    def _check_auto_quarantine(self, node_id: str) -> bool:
        """Check if node should be auto-quarantined based on trust score."""
        trust_score = self.trust_scores[node_id]
        
        # Skip if already quarantined
        if node_id in self.quarantined_nodes:
            return False
        
        # Determine quarantine level based on trust score
        target_level = None
        for level, threshold in sorted(self.quarantine_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if trust_score <= threshold:
                target_level = level
                break
        
        if target_level and target_level != QuarantineLevel.MONITORING:
            # Calculate duration based on severity
            duration_map = {
                QuarantineLevel.RESTRICTED: 24 * 3600,  # 24 hours
                QuarantineLevel.ISOLATED: 7 * 24 * 3600,  # 7 days
                QuarantineLevel.BANNED: None  # Permanent
            }
            
            self.quarantine_node(
                node_id=node_id,
                reason=f"Auto-quarantine due to low trust score: {trust_score:.2f}",
                level=target_level,
                duration=duration_map[target_level],
                evidence={"trust_score": trust_score, "auto_quarantine": True}
            )
            return True
        
        return False
    
    def _get_recent_activities(self, node_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent suspicious activities for a node."""
        cutoff = time.time() - hours * 3600
        recent = [
            a for a in self.suspicious_activities 
            if a.node_id == node_id and a.timestamp > cutoff
        ]
        
        return [
            {
                "activity_type": a.activity_type,
                "severity": a.severity,
                "timestamp": a.timestamp,
                "details": a.details
            }
            for a in sorted(recent, key=lambda x: x.timestamp, reverse=True)
        ]
    
    def _cleanup_expired_entries(self):
        """Background thread to clean up expired quarantine entries."""
        while True:
            try:
                current_time = time.time()
                expired_nodes = []
                
                for node_id, entry in self.quarantined_nodes.items():
                    if entry.auto_release and entry.duration:
                        if current_time - entry.timestamp > entry.duration:
                            expired_nodes.append(node_id)
                
                for node_id in expired_nodes:
                    self.release_node(node_id, "Auto-release after expiration")
                
                # Clean old suspicious activities (keep last 30 days)
                cutoff = current_time - 30 * 24 * 3600
                self.suspicious_activities = [
                    a for a in self.suspicious_activities if a.timestamp > cutoff
                ]
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                print(f"❌ Error in quarantine cleanup: {e}")
                time.sleep(3600)
    
    def save_quarantine_data(self):
        """Save quarantine data to disk."""
        data = {
            "quarantined_nodes": {
                node_id: asdict(entry) for node_id, entry in self.quarantined_nodes.items()
            },
            "trust_scores": dict(self.trust_scores),
            "suspicious_activities": [asdict(activity) for activity in self.suspicious_activities[-1000:]]  # Keep last 1000
        }
        
        # Convert enum to string for JSON serialization
        for node_data in data["quarantined_nodes"].values():
            node_data["level"] = node_data["level"].value
        
        quarantine_file = self.quarantine_dir / "quarantine_data.json"
        with open(quarantine_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_quarantine_data(self):
        """Load quarantine data from disk."""
        quarantine_file = self.quarantine_dir / "quarantine_data.json"
        
        if not quarantine_file.exists():
            return
        
        try:
            with open(quarantine_file, 'r') as f:
                data = json.load(f)
            
            # Load quarantined nodes
            for node_id, entry_data in data.get("quarantined_nodes", {}).items():
                entry_data["level"] = QuarantineLevel(entry_data["level"])
                self.quarantined_nodes[node_id] = QuarantineEntry(**entry_data)
            
            # Load trust scores
            self.trust_scores.update(data.get("trust_scores", {}))
            
            # Load suspicious activities
            for activity_data in data.get("suspicious_activities", []):
                self.suspicious_activities.append(SuspiciousActivity(**activity_data))
            
            print(f"📂 Loaded quarantine data: {len(self.quarantined_nodes)} quarantined nodes")
            
        except Exception as e:
            print(f"❌ Error loading quarantine data: {e}")

class NetworkDefenseCoordinator:
    """Coordinates network defense against malicious activities."""
    
    def __init__(self, quarantine_dir: Path):
        self.quarantine_manager = QuarantineManager(quarantine_dir)
        self.defense_policies = {
            "data_poisoning": {
                "initial_penalty": 0.3,
                "escalation_factor": 1.5,
                "quarantine_threshold": 2
            },
            "backdoor_attack": {
                "initial_penalty": 0.5,
                "escalation_factor": 2.0,
                "quarantine_threshold": 1
            },
            "gradient_manipulation": {
                "initial_penalty": 0.4,
                "escalation_factor": 1.8,
                "quarantine_threshold": 2
            },
            "consensus_manipulation": {
                "initial_penalty": 0.6,
                "escalation_factor": 2.5,
                "quarantine_threshold": 1
            }
        }
    
    def handle_security_incident(
        self, 
        node_id: str, 
        incident_type: str,
        severity: float,
        evidence: Dict[str, Any],
        reporter_nodes: List[str] = None
    ) -> Dict[str, Any]:
        """Handle a security incident and take appropriate action."""
        
        if reporter_nodes is None:
            reporter_nodes = ["system"]
        
        # Report suspicious activity
        self.quarantine_manager.report_suspicious_activity(
            node_id=node_id,
            activity_type=incident_type,
            severity=severity,
            details=evidence
        )
        
        # Check defense policy
        policy = self.defense_policies.get(incident_type, {
            "initial_penalty": 0.3,
            "escalation_factor": 1.5,
            "quarantine_threshold": 2
        })
        
        # Count recent incidents of this type
        recent_incidents = [
            a for a in self.quarantine_manager.suspicious_activities
            if (a.node_id == node_id and 
                a.activity_type == incident_type and
                time.time() - a.timestamp < 24 * 3600)  # Last 24 hours
        ]
        
        incident_count = len(recent_incidents)
        
        # Escalate if threshold reached
        if incident_count >= policy["quarantine_threshold"]:
            # Determine quarantine level based on severity and incident count
            if severity >= 0.9 or incident_count >= 5:
                level = QuarantineLevel.BANNED
                duration = None
            elif severity >= 0.7 or incident_count >= 3:
                level = QuarantineLevel.ISOLATED
                duration = 7 * 24 * 3600  # 7 days
            elif severity >= 0.5 or incident_count >= 2:
                level = QuarantineLevel.RESTRICTED
                duration = 24 * 3600  # 24 hours
            else:
                level = QuarantineLevel.MONITORING
                duration = 12 * 3600  # 12 hours
            
            self.quarantine_manager.quarantine_node(
                node_id=node_id,
                reason=f"{incident_type} incident (severity: {severity:.2f}, count: {incident_count})",
                level=level,
                duration=duration,
                evidence=evidence,
                reporter_nodes=reporter_nodes
            )
            
            action_taken = f"QUARANTINED_{level.value.upper()}"
        else:
            action_taken = "MONITORING_INCREASED"
        
        return {
            "node_id": node_id,
            "incident_type": incident_type,
            "severity": severity,
            "incident_count": incident_count,
            "action_taken": action_taken,
            "trust_score": self.quarantine_manager.trust_scores[node_id]
        }
    
    def get_network_health(self) -> Dict[str, Any]:
        """Get overall network health metrics."""
        report = self.quarantine_manager.get_network_security_report()
        
        # Calculate health score (0.0 to 1.0)
        health_score = 1.0
        
        # Penalize high quarantine rate
        health_score -= min(0.3, report["quarantine_rate"] * 0.5)
        
        # Penalize low trust distribution
        total_trust_nodes = sum(report["trust_distribution"].values())
        if total_trust_nodes > 0:
            high_trust_ratio = report["trust_distribution"]["high_trust"] / total_trust_nodes
            health_score = health_score * (0.5 + 0.5 * high_trust_ratio)
        
        # Penalize recent suspicious activities
        if report["recent_suspicious_activities"] > 10:
            health_score -= min(0.2, (report["recent_suspicious_activities"] - 10) * 0.01)
        
        health_score = max(0.0, health_score)
        
        return {
            **report,
            "network_health_score": health_score,
            "health_status": (
                "EXCELLENT" if health_score >= 0.9 else
                "GOOD" if health_score >= 0.7 else
                "FAIR" if health_score >= 0.5 else
                "POOR" if health_score >= 0.3 else
                "CRITICAL"
            )
        }

# Example usage and testing
def demo_quarantine_system():
    """Demonstrate the quarantine system."""
    defense_coordinator = NetworkDefenseCoordinator(Path("./quarantine_data"))
    
    print("🛡️ Network Defense System Demo\n")
    
    # Simulate normal node
    print("1. Normal node activity:")
    defense_coordinator.quarantine_manager.trust_scores["good_node"] = 0.9
    status = defense_coordinator.quarantine_manager.get_quarantine_status("good_node")
    print(f"   Trust score: {status['trust_score']:.2f}")
    
    # Simulate suspicious activities
    print("\n2. Suspicious activities from malicious node:")
    
    malicious_node = "malicious_node_123"
    
    # First incident - data poisoning
    result1 = defense_coordinator.handle_security_incident(
        node_id=malicious_node,
        incident_type="data_poisoning",
        severity=0.6,
        evidence={"detected_poison": True, "confidence": 0.8}
    )
    print(f"   Incident 1: {result1['action_taken']}")
    
    # Second incident - backdoor attack
    result2 = defense_coordinator.handle_security_incident(
        node_id=malicious_node,
        incident_type="backdoor_attack",
        severity=0.8,
        evidence={"backdoor_detected": True, "trigger_pattern": "SECRET_TRIGGER"}
    )
    print(f"   Incident 2: {result2['action_taken']}")
    
    # Check quarantine status
    print(f"\n3. Quarantine status for {malicious_node}:")
    status = defense_coordinator.quarantine_manager.get_quarantine_status(malicious_node)
    if status["quarantined"]:
        print(f"   Level: {status['level']}")
        print(f"   Reason: {status['reason']}")
        print(f"   Trust score: {status['trust_score']:.2f}")
    
    # Test action permissions
    print(f"\n4. Action permissions for quarantined node:")
    actions = ["read_chain", "submit_expert", "peer_communication", "admin_action"]
    for action in actions:
        can_perform, reason = defense_coordinator.quarantine_manager.can_node_perform_action(
            malicious_node, action
        )
        status_emoji = "✅" if can_perform else "❌"
        print(f"   {status_emoji} {action}: {reason}")
    
    # Network health report
    print(f"\n5. Network health report:")
    health = defense_coordinator.get_network_health()
    print(f"   Health score: {health['network_health_score']:.2f}")
    print(f"   Health status: {health['health_status']}")
    print(f"   Total nodes: {health['total_nodes']}")
    print(f"   Quarantined: {health['quarantined_nodes']}")

if __name__ == "__main__":
    demo_quarantine_system()