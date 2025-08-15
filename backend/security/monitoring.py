#!/usr/bin/env python3
"""
Basic Monitoring and Alerting System
Real-time security monitoring with immediate threat response.
"""

import os
import time
import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    severity: str  # "low", "medium", "high", "critical"
    source: str
    details: Dict[str, Any]
    timestamp: float
    resolved: bool = False
    alert_sent: bool = False


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    threshold: int
    window_seconds: int
    severity: str
    action: str  # "log", "alert", "quarantine"


class SecurityMetricsCollector:
    """Collect security metrics from various system components."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.event_windows = defaultdict(lambda: deque())
        self.last_collection = time.time()
    
    def record_event(self, event_type: str, source: str, details: Dict = None):
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=self._classify_severity(event_type),
            source=source,
            details=details or {},
            timestamp=time.time()
        )
        
        self.metrics[event_type].append(event)
        self.event_windows[event_type].append(event.timestamp)
        
        # Keep only recent events in window (last hour)
        cutoff = time.time() - 3600
        while self.event_windows[event_type] and self.event_windows[event_type][0] < cutoff:
            self.event_windows[event_type].popleft()
    
    def _classify_severity(self, event_type: str) -> str:
        """Classify event severity based on type."""
        critical_events = {
            "genesis_integrity_failure", "expert_tampering_detected", 
            "malicious_peer_connection", "blockchain_corruption"
        }
        high_events = {
            "failed_auth_attempts", "node_quarantine", "unusual_upload_patterns",
            "security_scan_failure", "rate_limit_bypass_attempt"
        }
        medium_events = {
            "pol_validation_failure", "expert_validation_failure",
            "suspicious_behavior_pattern", "quota_exceeded"
        }
        
        if event_type in critical_events:
            return "critical"
        elif event_type in high_events:
            return "high"
        elif event_type in medium_events:
            return "medium"
        else:
            return "low"
    
    def get_event_count(self, event_type: str, window_seconds: int = 300) -> int:
        """Get event count within time window."""
        cutoff = time.time() - window_seconds
        return sum(1 for ts in self.event_windows[event_type] if ts >= cutoff)
    
    def get_recent_events(self, limit: int = 50) -> List[SecurityEvent]:
        """Get recent security events."""
        all_events = []
        for event_list in self.metrics.values():
            all_events.extend(event_list)
        
        # Sort by timestamp, most recent first
        all_events.sort(key=lambda x: x.timestamp, reverse=True)
        return all_events[:limit]


class AlertManager:
    """Manage security alerts and notifications."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/monitoring")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert thresholds
        self.thresholds = {
            "failed_auth_attempts": AlertThreshold(10, 300, "high", "alert"),  # 10 in 5 min
            "unusual_upload_patterns": AlertThreshold(50, 3600, "high", "alert"),  # 50 in 1 hour
            "genesis_integrity_failure": AlertThreshold(1, 60, "critical", "quarantine"),  # 1 in 1 min
            "expert_tampering_detected": AlertThreshold(1, 60, "critical", "quarantine"),  # 1 in 1 min
            "node_quarantine_events": AlertThreshold(3, 3600, "high", "alert"),  # 3 in 1 hour
            "pol_validation_failure": AlertThreshold(20, 1800, "medium", "log"),  # 20 in 30 min
            "rate_limit_exceeded": AlertThreshold(100, 300, "medium", "log"),  # 100 in 5 min
            # Pipeline-related alerts
            "pipeline_rpc_errors": AlertThreshold(int(os.getenv('ALERT_PIPELINE_RPC_ERRORS', '20')), 300, "medium", "alert"),
            "device_profile_staleness": AlertThreshold(int(os.getenv('ALERT_DEVICE_PROFILE_STALENESS', '10')), 600, "medium", "alert"),
            "partition_drift_high": AlertThreshold(int(os.getenv('ALERT_PARTITION_DRIFT', '5')), 600, "high", "alert")
        }
        
        self.alert_history: List[Dict] = []
        self._load_alert_history()
    
    def _load_alert_history(self):
        """Load alert history from storage."""
        history_file = self.storage_dir / "alert_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.alert_history = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load alert history: {e}")
    
    def _save_alert_history(self):
        """Save alert history to storage."""
        history_file = self.storage_dir / "alert_history.json"
        try:
            # Keep only last 1000 alerts
            recent_alerts = self.alert_history[-1000:]
            with open(history_file, 'w') as f:
                json.dump(recent_alerts, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save alert history: {e}")
    
    def check_thresholds(self, metrics_collector: SecurityMetricsCollector):
        """Check if any alert thresholds are exceeded."""
        alerts_triggered = []
        
        for event_type, threshold in self.thresholds.items():
            count = metrics_collector.get_event_count(event_type, threshold.window_seconds)
            
            if count >= threshold.threshold:
                alert = self._create_alert(event_type, count, threshold)
                alerts_triggered.append(alert)
                
                # Take action based on threshold config
                if threshold.action == "alert":
                    self._send_alert(alert)
                elif threshold.action == "quarantine":
                    self._trigger_quarantine(alert)
                elif threshold.action == "log":
                    self._log_alert(alert)
        
        return alerts_triggered
    
    def _create_alert(self, event_type: str, count: int, threshold: AlertThreshold) -> Dict:
        """Create alert dictionary."""
        alert = {
            "alert_id": hashlib.sha256(f"{event_type}{time.time()}".encode()).hexdigest()[:16],
            "event_type": event_type,
            "severity": threshold.severity,
            "count": count,
            "threshold": threshold.threshold,
            "window_seconds": threshold.window_seconds,
            "timestamp": time.time(),
            "message": f"{event_type} threshold exceeded: {count} events in {threshold.window_seconds}s (limit: {threshold.threshold})",
            "action_taken": threshold.action,
            "resolved": False
        }
        
        self.alert_history.append(alert)
        self._save_alert_history()
        return alert
    
    def _send_alert(self, alert: Dict):
        """Send alert notification (placeholder for Slack/PagerDuty integration)."""
        print(f"🚨 SECURITY ALERT [{alert['severity'].upper()}]: {alert['message']}")
        
        # TODO: Implement actual alerting
        # - Slack webhook
        # - PagerDuty integration
        # - Email notifications
        # - SMS alerts for critical events
        
        # Log to file for now
        alert_log = self.storage_dir / "security_alerts.log"
        with open(alert_log, 'a') as f:
            f.write(f"{datetime.now().isoformat()} [{alert['severity'].upper()}] {alert['message']}\n")
    
    def _trigger_quarantine(self, alert: Dict):
        """Trigger automatic quarantine for critical events."""
        print(f"🛡️ QUARANTINE TRIGGERED: {alert['message']}")
        
        # TODO: Implement quarantine actions
        # - Block suspicious IP addresses
        # - Disable compromised API keys
        # - Isolate suspicious nodes
        # - Trigger rollback if needed
        
        self._send_alert(alert)  # Also send alert
    
    def _log_alert(self, alert: Dict):
        """Log alert without notification."""
        logging.warning(f"Security threshold exceeded: {alert['message']}")


class SystemHealthMonitor:
    """Monitor overall system health and performance."""
    
    def __init__(self):
        self.health_metrics = {
            "api_response_time": deque(maxlen=100),
            "blockchain_sync_status": True,
            "expert_validation_success_rate": 1.0,
            "p2p_node_count": 0,
            "memory_usage_percent": 0.0,
            "disk_usage_percent": 0.0
        }
        self.last_health_check = time.time()
    
    def record_api_response_time(self, response_time_ms: float):
        """Record API response time."""
        self.health_metrics["api_response_time"].append(response_time_ms)
    
    def update_health_metrics(self, **kwargs):
        """Update health metrics."""
        for key, value in kwargs.items():
            if key in self.health_metrics:
                self.health_metrics[key] = value
    
    def get_health_status(self) -> Dict:
        """Get current system health status."""
        avg_response_time = 0
        if self.health_metrics["api_response_time"]:
            avg_response_time = sum(self.health_metrics["api_response_time"]) / len(self.health_metrics["api_response_time"])
        
        return {
            "overall_status": self._calculate_overall_status(),
            "api_avg_response_time_ms": avg_response_time,
            "blockchain_sync_status": self.health_metrics["blockchain_sync_status"],
            "expert_validation_success_rate": self.health_metrics["expert_validation_success_rate"],
            "p2p_node_count": self.health_metrics["p2p_node_count"],
            "memory_usage_percent": self.health_metrics["memory_usage_percent"],
            "disk_usage_percent": self.health_metrics["disk_usage_percent"],
            "last_health_check": self.last_health_check,
            "uptime_seconds": time.time() - system_start_time
        }
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system status."""
        if not self.health_metrics["blockchain_sync_status"]:
            return "degraded"
        
        if self.health_metrics["expert_validation_success_rate"] < 0.8:
            return "degraded"
        
        avg_response_time = 0
        if self.health_metrics["api_response_time"]:
            avg_response_time = sum(self.health_metrics["api_response_time"]) / len(self.health_metrics["api_response_time"])
        
        if avg_response_time > 5000:  # 5 seconds
            return "degraded"
        
        if self.health_metrics["memory_usage_percent"] > 90:
            return "degraded"
        
        if self.health_metrics["disk_usage_percent"] > 95:
            return "critical"
        
        return "healthy"


class SecurityMonitoringSystem:
    """Main security monitoring system coordinator."""
    
    def __init__(self):
        self.metrics_collector = SecurityMetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = SystemHealthMonitor()
        
        self.monitoring_active = True
        self.monitoring_thread = None
        
        # Start monitoring loop
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring loop."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print("🔍 Security monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("🔍 Security monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check alert thresholds
                alerts = self.alert_manager.check_thresholds(self.metrics_collector)
                
                # Update health metrics
                self.health_monitor.last_health_check = time.time()
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except Exception as e:
                print(f"Warning: Monitoring loop error: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def record_security_event(self, event_type: str, source: str, details: Dict = None):
        """Record a security event."""
        self.metrics_collector.record_event(event_type, source, details)
    
    def get_security_dashboard(self) -> Dict:
        """Get comprehensive security dashboard data."""
        recent_events = self.metrics_collector.get_recent_events(20)
        health_status = self.health_monitor.get_health_status()
        recent_alerts = self.alert_manager.alert_history[-10:]  # Last 10 alerts
        
        return {
            "system_health": health_status,
            "recent_security_events": [asdict(event) for event in recent_events],
            "recent_alerts": recent_alerts,
            "active_thresholds": {
                event_type: {
                    "threshold": threshold.threshold,
                    "window_seconds": threshold.window_seconds,
                    "current_count": self.metrics_collector.get_event_count(event_type, threshold.window_seconds)
                }
                for event_type, threshold in self.alert_manager.thresholds.items()
            },
            "monitoring_status": {
                "active": self.monitoring_active,
                "uptime": time.time() - system_start_time,
                "thread_alive": self.monitoring_thread.is_alive() if self.monitoring_thread else False
            }
        }


# Global instances
system_start_time = time.time()
security_monitor = SecurityMonitoringSystem()


# Convenience functions for easy integration
def record_security_event(event_type: str, source: str, details: Dict = None):
    """Convenience function to record security events."""
    security_monitor.record_security_event(event_type, source, details)


def record_api_response_time(response_time_ms: float):
    """Convenience function to record API response times."""
    security_monitor.health_monitor.record_api_response_time(response_time_ms)


def get_security_dashboard():
    """Convenience function to get security dashboard."""
    return security_monitor.get_security_dashboard()


def get_system_health():
    """Convenience function to get system health."""
    return security_monitor.health_monitor.get_health_status()