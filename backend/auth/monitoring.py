"""
Monitoring and Observability for API Key System
===============================================

Prometheus metrics, structured logging, and alerting hooks.
"""

import time
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import json

# Configure structured logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Prometheus metrics registry
registry = CollectorRegistry()

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

# Counters
api_key_created = Counter(
    'api_key_created_total',
    'Total number of API keys created',
    ['role', 'version'],
    registry=registry
)

api_key_validated = Counter(
    'api_key_validated_total',
    'Total number of API key validations',
    ['role', 'version', 'result'],
    registry=registry
)

api_key_refreshed = Counter(
    'api_key_refreshed_total',
    'Total number of API key refreshes',
    ['role', 'success'],
    registry=registry
)

api_key_revoked = Counter(
    'api_key_revoked_total',
    'Total number of API keys revoked',
    ['role', 'reason'],
    registry=registry
)

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total number of rate limit violations',
    ['role', 'endpoint'],
    registry=registry
)

auth_errors = Counter(
    'auth_errors_total',
    'Total number of authentication errors',
    ['error_type', 'version'],
    registry=registry
)

# Histograms
key_validation_duration = Histogram(
    'key_validation_duration_seconds',
    'Time taken to validate an API key',
    ['version'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

jwt_decode_duration = Histogram(
    'jwt_decode_duration_seconds',
    'Time taken to decode JWT token',
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01),
    registry=registry
)

redis_operation_duration = Histogram(
    'redis_operation_duration_seconds',
    'Time taken for Redis operations',
    ['operation'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
    registry=registry
)

# Gauges
active_api_keys = Gauge(
    'active_api_keys',
    'Number of active API keys',
    ['role'],
    registry=registry
)

redis_connection_status = Gauge(
    'redis_connection_status',
    'Redis connection status (1=connected, 0=disconnected)',
    registry=registry
)

refresh_queue_size = Gauge(
    'refresh_queue_size',
    'Number of keys pending refresh',
    registry=registry
)

# ============================================================================
# MONITORING CLASSES
# ============================================================================

@dataclass
class AuthEvent:
    """Structured auth event for logging and alerting"""
    timestamp: str
    event_type: str
    user_id: Optional[str]
    role: Optional[str]
    endpoint: Optional[str]
    success: bool
    latency_ms: Optional[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON for structured logging"""
        return json.dumps(asdict(self))

class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MonitoringService:
    """
    Central monitoring service for API key system.
    Collects metrics, logs events, and triggers alerts.
    """
    
    def __init__(self):
        self.alert_thresholds = {
            "error_rate": 0.05,           # 5% error rate
            "refresh_failure_rate": 0.10,  # 10% refresh failure
            "validation_latency_p99": 0.1, # 100ms p99 latency
            "rate_limit_violations": 100,  # 100 violations per hour
        }
        
        self.metrics_buffer = []
        self.last_flush = time.time()
        
    # ========================================================================
    # METRIC COLLECTION
    # ========================================================================
    
    def track_key_creation(self, role: str, version: str = "v2"):
        """Track API key creation"""
        api_key_created.labels(role=role, version=version).inc()
        active_api_keys.labels(role=role).inc()
        
        self.log_event(AuthEvent(
            timestamp=datetime.now().isoformat(),
            event_type="key_created",
            user_id=None,
            role=role,
            endpoint="/auth/register",
            success=True,
            latency_ms=None,
            error_message=None,
            metadata={"version": version}
        ))
    
    def track_key_validation(self, role: str, success: bool, latency: float, version: str = "v2"):
        """Track API key validation"""
        result = "success" if success else "failure"
        api_key_validated.labels(role=role, version=version, result=result).inc()
        key_validation_duration.labels(version=version).observe(latency)
        
        if not success:
            auth_errors.labels(error_type="validation_failed", version=version).inc()
    
    def track_key_refresh(self, role: str, success: bool, error: Optional[str] = None):
        """Track API key refresh"""
        api_key_refreshed.labels(role=role, success=str(success).lower()).inc()
        
        if not success:
            self.log_event(AuthEvent(
                timestamp=datetime.now().isoformat(),
                event_type="refresh_failed",
                user_id=None,
                role=role,
                endpoint="/auth/refresh",
                success=False,
                latency_ms=None,
                error_message=error,
                metadata={}
            ))
            
            # Check if we need to alert
            self.check_refresh_failures()
    
    def track_key_revocation(self, role: str, reason: str):
        """Track API key revocation"""
        api_key_revoked.labels(role=role, reason=reason).inc()
        active_api_keys.labels(role=role).dec()
        
        self.log_event(AuthEvent(
            timestamp=datetime.now().isoformat(),
            event_type="key_revoked",
            user_id=None,
            role=role,
            endpoint="/auth/revoke",
            success=True,
            latency_ms=None,
            error_message=None,
            metadata={"reason": reason}
        ))
    
    def track_rate_limit_violation(self, role: str, endpoint: str):
        """Track rate limit violation"""
        rate_limit_exceeded.labels(role=role, endpoint=endpoint).inc()
        
        # Check if we need to alert
        self.check_rate_limit_violations()
    
    def track_jwt_decode(self, duration: float):
        """Track JWT decode performance"""
        jwt_decode_duration.observe(duration)
    
    def track_redis_operation(self, operation: str, duration: float):
        """Track Redis operation performance"""
        redis_operation_duration.labels(operation=operation).observe(duration)
    
    def update_redis_status(self, connected: bool):
        """Update Redis connection status"""
        redis_connection_status.set(1 if connected else 0)
        
        if not connected:
            self.trigger_alert(
                AlertLevel.CRITICAL,
                "Redis connection lost",
                {"service": "redis", "impact": "auth_degraded"}
            )
    
    def update_refresh_queue(self, size: int):
        """Update refresh queue size"""
        refresh_queue_size.set(size)
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def log_event(self, event: AuthEvent):
        """Log structured auth event"""
        if event.success:
            logger.info(f"AUTH_EVENT: {event.to_json()}")
        else:
            logger.error(f"AUTH_ERROR: {event.to_json()}")
        
        # Buffer for batch processing
        self.metrics_buffer.append(event)
        
        # Flush periodically
        if time.time() - self.last_flush > 60:  # Every minute
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush metrics buffer to persistent storage or stream"""
        if not self.metrics_buffer:
            return
        
        # Here you would send to your metrics backend
        # For now, just log summary
        error_count = sum(1 for e in self.metrics_buffer if not e.success)
        total_count = len(self.metrics_buffer)
        
        if total_count > 0:
            error_rate = error_count / total_count
            logger.info(f"Metrics flush: {total_count} events, {error_rate:.2%} error rate")
        
        self.metrics_buffer.clear()
        self.last_flush = time.time()
    
    # ========================================================================
    # ALERTING
    # ========================================================================
    
    def check_refresh_failures(self):
        """Check refresh failure rate and alert if needed"""
        # Get recent metrics (this would query Prometheus in production)
        # For now, simplified check
        recent_failures = api_key_refreshed.labels(role="basic", success="false")._value._value
        recent_total = (
            api_key_refreshed.labels(role="basic", success="true")._value._value +
            recent_failures
        )
        
        if recent_total > 10:  # Minimum sample size
            failure_rate = recent_failures / recent_total
            if failure_rate > self.alert_thresholds["refresh_failure_rate"]:
                self.trigger_alert(
                    AlertLevel.ERROR,
                    f"High refresh failure rate: {failure_rate:.1%}",
                    {"metric": "refresh_failure_rate", "value": failure_rate}
                )
    
    def check_rate_limit_violations(self):
        """Check rate limit violations and alert if needed"""
        # Simplified check - in production would use time windows
        total_violations = sum(
            rate_limit_exceeded.labels(role=role, endpoint="")._value._value
            for role in ["basic", "contributor", "node_operator"]
        )
        
        if total_violations > self.alert_thresholds["rate_limit_violations"]:
            self.trigger_alert(
                AlertLevel.WARNING,
                f"High rate limit violations: {total_violations}",
                {"metric": "rate_limit_violations", "count": total_violations}
            )
    
    def trigger_alert(self, level: AlertLevel, message: str, metadata: Dict[str, Any]):
        """
        Trigger an alert.
        
        In production, this would integrate with:
        - PagerDuty for critical alerts
        - Slack for warnings
        - Email for info
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message,
            "metadata": metadata
        }
        
        if level == AlertLevel.CRITICAL:
            logger.critical(f"ALERT: {json.dumps(alert)}")
            # Would trigger PagerDuty here
        elif level == AlertLevel.ERROR:
            logger.error(f"ALERT: {json.dumps(alert)}")
            # Would send to Slack #alerts channel
        elif level == AlertLevel.WARNING:
            logger.warning(f"ALERT: {json.dumps(alert)}")
            # Would send to Slack #warnings channel
        else:
            logger.info(f"ALERT: {json.dumps(alert)}")
    
    # ========================================================================
    # HEALTH CHECKS
    # ========================================================================
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        # Calculate key metrics
        total_keys = sum(
            active_api_keys.labels(role=role)._value._value
            for role in ["basic", "contributor", "node_operator", "admin"]
        )
        
        redis_connected = redis_connection_status._value._value == 1
        
        # Calculate error rates
        total_validations = sum(
            api_key_validated.labels(role="", version="", result=result)._value._value
            for result in ["success", "failure"]
        )
        
        failed_validations = api_key_validated.labels(role="", version="", result="failure")._value._value
        validation_error_rate = failed_validations / max(1, total_validations)
        
        # Determine health status
        if not redis_connected:
            status = "degraded"
            issues = ["Redis disconnected"]
        elif validation_error_rate > 0.10:
            status = "degraded"
            issues = [f"High validation error rate: {validation_error_rate:.1%}"]
        else:
            status = "healthy"
            issues = []
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_active_keys": total_keys,
                "redis_connected": redis_connected,
                "validation_error_rate": validation_error_rate,
                "refresh_queue_size": refresh_queue_size._value._value
            },
            "issues": issues
        }
    
    def export_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(registry)

# ============================================================================
# DECORATORS FOR EASY INTEGRATION
# ============================================================================

monitoring = MonitoringService()

def track_performance(operation: str):
    """
    Decorator to track operation performance.
    
    Example:
        @track_performance("key_validation")
        async def validate_key(key: str):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start = time.time()
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration = time.time() - start
                
                # Log performance
                logger.info(f"Operation {operation} took {duration:.3f}s")
                
                # Track in Prometheus
                if "validation" in operation:
                    key_validation_duration.labels(version="v2").observe(duration)
                elif "redis" in operation:
                    redis_operation_duration.labels(operation=operation).observe(duration)
                
                # Log if slow
                if duration > 1.0:
                    monitoring.trigger_alert(
                        AlertLevel.WARNING,
                        f"Slow operation: {operation} took {duration:.1f}s",
                        {"operation": operation, "duration": duration}
                    )
        
        return wrapper
    return decorator

"""
Integration Example:

from backend.auth.monitoring import monitoring, track_performance

class APIKeyManager:
    @track_performance("key_validation")
    async def validate_api_key(self, key: str):
        monitoring.track_key_validation("basic", True, 0.005)
        ...
    
    async def refresh_api_key(self, key: str):
        try:
            # Refresh logic
            monitoring.track_key_refresh("basic", True)
        except Exception as e:
            monitoring.track_key_refresh("basic", False, str(e))
            raise

# Expose Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        monitoring.export_prometheus_metrics(),
        media_type="text/plain"
    )
"""