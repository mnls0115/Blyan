# 🔐 API Key Management Playbook

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Day-to-Day Operations](#day-to-day-operations)
4. [Emergency Procedures](#emergency-procedures)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Security Best Practices](#security-best-practices)
8. [Runbooks](#runbooks)

---

## Overview

This playbook provides comprehensive guidance for operating and maintaining the Blyan API Key Management System V2. It covers routine operations, emergency procedures, and troubleshooting scenarios.

### System Components

| Component | Purpose | Location |
|-----------|---------|----------|
| API Key Manager | JWT token generation/validation | `backend/auth/api_key_system.py` |
| Feature Flags | Zero-downtime rollout control | `backend/auth/feature_flags.py` |
| Monitoring | Prometheus metrics & alerting | `backend/auth/monitoring.py` |
| Revocation | Key blocklist management | `backend/auth/revocation.py` |
| Secret Rotation | JWT secret lifecycle | `backend/auth/secret_rotation.py` |
| Canary Testing | Continuous validation | `backend/auth/canary_testing.py` |

### Critical Dependencies

- **Redis**: Required for distributed caching, revocation, and rate limiting
- **JWT Library**: Core cryptographic operations
- **Prometheus**: Metrics collection and alerting

---

## Architecture

### Data Flow
```
User Request → API Gateway → JWT Validation → Redis Cache Check → 
→ Permission Verification → Rate Limiting → Application Logic
```

### Key Storage Layers
1. **JWT Token**: Self-contained, signed with secret
2. **Redis Cache**: Fast validation, revocation list
3. **Audit Log**: Compliance and forensics

---

## Day-to-Day Operations

### 1. Monitor Key Metrics

#### Dashboard URLs
- Grafana: `http://monitoring.blyan.com:3000/dashboard/api-keys`
- Prometheus: `http://monitoring.blyan.com:9090`

#### Key Metrics to Watch
```bash
# Check current API key statistics
curl http://localhost:8000/metrics | grep api_key

# Key metrics:
# - api_key_created_total: New keys per hour
# - api_key_validated_total: Validation rate
# - key_validation_duration_seconds: p99 < 10ms
# - rate_limit_exceeded_total: Should be < 1% of requests
```

### 2. Gradual V2 Rollout

#### Current rollout status
```bash
# Check feature flag status
redis-cli GET "feature_flags:api_key_v2" | jq .

# View rollout metrics
curl http://localhost:8000/api/admin/feature_flags/metrics
```

#### Adjust rollout percentage
```python
# Increase to 25% traffic
from backend.auth.feature_flags import feature_flags, RolloutStrategy

feature_flags.set_rollout_strategy(
    RolloutStrategy.PERCENTAGE,
    percentage=25
)
```

#### Rollout schedule (recommended)
- Week 1: 5% canary
- Week 2: 25% rollout
- Week 3: 50% rollout
- Week 4: 100% (complete)

### 3. JWT Secret Rotation

#### Quarterly rotation schedule
```bash
# Check last rotation
curl http://localhost:8000/api/admin/secrets/status

# Initiate rotation (overlapping mode)
curl -X POST http://localhost:8000/api/admin/secrets/rotate \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -d '{"reason": "scheduled_quarterly", "immediate": false}'
```

#### Rotation process
1. Create new secret (state: ROTATING)
2. Sign new tokens with new secret
3. Accept both secrets for 24 hours
4. Complete rotation (new secret ACTIVE)
5. Expire old secret after 24 hours

### 4. Canary Testing

#### Check canary health
```bash
# View canary test results
curl http://localhost:8000/api/admin/canary/health

# Expected output:
{
  "status": "healthy",
  "pass_rate": 98.5,
  "recent_tests": 60,
  "consecutive_failures": 0
}
```

#### Add new canary test
```python
from backend.auth.canary_testing import CanaryTest

new_test = CanaryTest(
    test_id="custom_test",
    name="Custom Endpoint Test",
    endpoint="/api/custom",
    method="GET",
    payload=None,
    expected_status=200,
    timeout=5.0,
    critical=True
)
```

---

## Emergency Procedures

### 🚨 1. Complete System Rollback

**Scenario**: V2 system causing widespread failures

```bash
#!/bin/bash
# EMERGENCY ROLLBACK SCRIPT

# 1. Disable V2 immediately
redis-cli SET "feature_flags:api_key_v2" '{"strategy": "off"}'

# 2. Clear V2 cache
redis-cli --scan --pattern "api_key:v2:*" | xargs redis-cli DEL

# 3. Notify team
curl -X POST $SLACK_WEBHOOK -d '{"text": "🚨 API V2 rolled back"}'

# 4. Restart API servers
kubectl rollout restart deployment/api-server

echo "✅ Rollback complete - all traffic on V1"
```

### 🚨 2. Mass Key Revocation

**Scenario**: Security breach requiring immediate key invalidation

```python
#!/usr/bin/env python3
# EMERGENCY REVOCATION SCRIPT

from backend.auth.revocation import revocation_manager, RevocationReason
import asyncio

async def emergency_revoke():
    # Revoke all keys created in last 24 hours
    count = await revocation_manager.emergency_revoke_all(
        created_before=datetime.now() - timedelta(hours=24),
        revoked_by="security_team"
    )
    
    print(f"🚨 Emergency revocation activated")
    print(f"All validations will fail for 1 hour")
    print(f"Run recovery script when ready")

asyncio.run(emergency_revoke())
```

### 🚨 3. Redis Failure Recovery

**Scenario**: Redis cluster down, auth system degraded

```bash
#!/bin/bash
# REDIS FAILURE RECOVERY

# 1. Switch to backup Redis
export REDIS_HOST=redis-backup.blyan.com

# 2. Restore from snapshot
redis-cli --rdb /backups/redis-latest.rdb

# 3. Rebuild bloom filters
python3 -c "
from backend.auth.revocation import revocation_manager
import asyncio
asyncio.run(revocation_manager.rebuild_bloom_filter())
"

# 4. Verify
redis-cli PING
redis-cli DBSIZE

echo "✅ Redis recovered with $(redis-cli DBSIZE) keys"
```

### 🚨 4. Rate Limit Storm

**Scenario**: Sudden spike in rate limit violations

```python
#!/usr/bin/env python3
# RATE LIMIT EMERGENCY ADJUSTMENT

import redis
import json

r = redis.Redis(host='localhost', port=6379)

# Temporarily increase limits
emergency_limits = {
    "basic": 1000,      # 10x normal
    "contributor": 5000, # 5x normal
    "node_operator": -1  # unlimited
}

# Apply for 1 hour
r.setex("rate_limits:emergency", 3600, json.dumps(emergency_limits))

print("⚠️  Emergency rate limits active for 1 hour")
print("Remember to investigate root cause!")
```

---

## Monitoring & Alerts

### Alert Configuration

#### Critical Alerts (PagerDuty)
```yaml
- name: APIKeyValidationFailureHigh
  expr: rate(api_key_validated_total{result="failure"}[5m]) > 0.1
  severity: critical
  action: page_oncall

- name: RedisConnectionLost
  expr: redis_connection_status == 0
  severity: critical
  action: page_oncall

- name: JWTSecretRotationFailed
  expr: increase(secret_rotation_failures[1h]) > 0
  severity: critical
  action: page_oncall
```

#### Warning Alerts (Slack)
```yaml
- name: HighRefreshFailureRate
  expr: rate(api_key_refreshed_total{success="false"}[1h]) > 0.05
  severity: warning
  action: slack_notify

- name: RateLimitViolationsHigh
  expr: rate(rate_limit_exceeded_total[5m]) > 10
  severity: warning
  action: slack_notify
```

### Dashboards

#### Key Performance Indicators (KPIs)
1. **Availability**: Target 99.99% uptime
2. **Latency**: p99 validation < 10ms
3. **Error Rate**: < 0.1% validation failures
4. **Adoption**: V2 usage percentage

#### Real-time Monitoring Commands
```bash
# Watch validation latency
watch -n 1 'curl -s localhost:8000/metrics | grep key_validation_duration'

# Monitor error rate
watch -n 5 'curl -s localhost:8000/metrics | grep auth_errors_total'

# Check Redis performance
redis-cli --latency-history

# View canary test results
watch -n 10 'curl -s localhost:8000/api/admin/canary/health | jq .'
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "Invalid API Key" Errors Spike

**Symptoms**: Sudden increase in 401 errors

**Check**:
```bash
# Check if keys are expired
redis-cli --scan --pattern "api_key:*" | head -10 | xargs -I {} redis-cli TTL {}

# Verify JWT secret is correct
echo $JWT_SECRET | md5sum

# Check revocation list
redis-cli SCARD revoked_bloom
```

**Fix**:
```python
# Clear revocation cache if corrupted
from backend.auth.revocation import revocation_manager
await revocation_manager.rebuild_bloom_filter()
```

#### 2. Slow Validation Performance

**Symptoms**: p99 latency > 50ms

**Check**:
```bash
# Redis latency
redis-cli --latency

# Cache hit rate
redis-cli INFO stats | grep keyspace_hits

# JWT decode performance
curl localhost:8000/metrics | grep jwt_decode_duration
```

**Fix**:
```bash
# Optimize Redis
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli BGREWRITEAOF

# Increase local cache TTL
export API_KEY_CACHE_TTL=300  # 5 minutes
```

#### 3. Feature Flag Not Working

**Symptoms**: All traffic still going to V1 despite percentage set

**Check**:
```python
from backend.auth.feature_flags import feature_flags

# Check configuration
config = feature_flags.config
print(f"Strategy: {config['strategy']}")
print(f"Percentage: {config.get('percentage')}")
print(f"Enabled endpoints: {config.get('enabled_endpoints')}")

# Check metrics
metrics = feature_flags.get_metrics()
print(f"V2 requests: {metrics['v2_requests']}")
```

**Fix**:
```python
# Reset and reconfigure
feature_flags.config = feature_flags.default_config
feature_flags.save_config()

# Set strategy again
feature_flags.set_rollout_strategy(
    RolloutStrategy.PERCENTAGE,
    percentage=10
)

# Enable specific endpoints
feature_flags.enable_endpoints([
    "/api/auth/register",
    "/api/auth/validate",
    "/api/chat"
])
```

#### 4. Memory Leak in Redis

**Symptoms**: Redis memory usage growing unbounded

**Check**:
```bash
# Memory usage by pattern
redis-cli --bigkeys

# Check for orphaned keys
redis-cli --scan --pattern "api_key:*" | wc -l
redis-cli --scan --pattern "revocation_audit:*" | wc -l
```

**Fix**:
```python
# Clean up expired records
from backend.auth.revocation import revocation_manager
cleaned = await revocation_manager.cleanup_expired()
print(f"Cleaned {cleaned} expired records")

# Set TTL on audit records
import redis
r = redis.Redis()
for key in r.scan_iter("revocation_audit:*"):
    if r.ttl(key) == -1:  # No TTL set
        r.expire(key, 86400 * 30)  # 30 days
```

---

## Security Best Practices

### 1. Secret Management

```bash
# Rotate JWT secret quarterly
*/90 * * * * /usr/local/bin/rotate_jwt_secret.sh

# Never log secrets
export LOG_SECRETS=false

# Use hardware security module (HSM) in production
export USE_HSM=true
export HSM_ENDPOINT=hsm.blyan.com:11111
```

### 2. Audit Requirements

```python
# Enable comprehensive audit logging
from backend.auth.monitoring import monitoring

# Log all key operations
monitoring.log_event(AuthEvent(
    timestamp=datetime.now().isoformat(),
    event_type="key_created",
    user_id=user_id,
    role=role,
    endpoint=endpoint,
    success=True,
    latency_ms=latency,
    error_message=None,
    metadata={"ip": request.client.host}
))
```

### 3. Compliance Checklist

- [ ] All API keys have expiration dates
- [ ] Audit logs retained for 7 years
- [ ] PII redacted from logs
- [ ] Encryption at rest for secrets
- [ ] Regular security scans
- [ ] Penetration testing quarterly

---

## Runbooks

### Runbook: Daily Health Check

```bash
#!/bin/bash
# Daily API Key System Health Check

echo "=== API Key System Daily Health Check ==="
echo "Date: $(date)"

# 1. Check Redis
echo -n "Redis Status: "
redis-cli PING || echo "❌ FAILED"

# 2. Check Canary Tests
echo -n "Canary Health: "
curl -s localhost:8000/api/admin/canary/health | jq -r .status

# 3. Check Error Rates
echo "Error Rate (last hour):"
curl -s localhost:8000/metrics | grep -E "auth_errors_total|rate_limit_exceeded"

# 4. Check Secret Rotation Status
echo "Secret Rotation:"
curl -s localhost:8000/api/admin/secrets/status | jq .

# 5. Check Feature Flags
echo "V2 Rollout Status:"
redis-cli GET "feature_flags:api_key_v2" | jq -r .strategy

echo "=== Health Check Complete ==="
```

### Runbook: Weekly Maintenance

```python
#!/usr/bin/env python3
# Weekly API Key System Maintenance

import asyncio
from datetime import datetime, timedelta
from backend.auth.revocation import revocation_manager
from backend.auth.monitoring import monitoring

async def weekly_maintenance():
    print(f"Starting weekly maintenance - {datetime.now()}")
    
    # 1. Clean expired revocations
    cleaned = await revocation_manager.cleanup_expired()
    print(f"Cleaned {cleaned} expired revocations")
    
    # 2. Rebuild bloom filter for accuracy
    await revocation_manager.rebuild_bloom_filter()
    print("Bloom filter rebuilt")
    
    # 3. Export metrics for reporting
    metrics = monitoring.get_health_status()
    print(f"System health: {metrics['status']}")
    
    # 4. Test canary user
    from backend.auth.canary_testing import canary_suite
    result = await canary_suite.run_suite()
    print(f"Canary tests: {result['statistics']['pass_rate']:.1f}% pass rate")
    
    print("Weekly maintenance complete")

asyncio.run(weekly_maintenance())
```

### Runbook: Incident Response

```bash
#!/bin/bash
# API Key Incident Response Runbook

INCIDENT_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/var/log/incidents/$INCIDENT_ID"

echo "🚨 Starting Incident Response - ID: $INCIDENT_ID"

# 1. Create incident directory
mkdir -p $LOG_DIR

# 2. Capture current state
echo "Capturing system state..."
redis-cli INFO > $LOG_DIR/redis_info.txt
curl -s localhost:8000/metrics > $LOG_DIR/prometheus_metrics.txt
curl -s localhost:8000/api/admin/canary/health > $LOG_DIR/canary_health.json
docker logs api-server --tail 1000 > $LOG_DIR/api_logs.txt

# 3. Check for emergency flags
echo "Checking emergency flags..."
redis-cli GET "emergency_revocation" > $LOG_DIR/emergency_status.txt

# 4. Analyze recent errors
echo "Analyzing recent errors..."
grep ERROR /var/log/api/api.log | tail -100 > $LOG_DIR/recent_errors.txt

# 5. Generate report
cat > $LOG_DIR/incident_report.md << EOF
# Incident Report: $INCIDENT_ID

## Timestamp
$(date)

## System Status
- Redis: $(redis-cli PING)
- API: $(curl -s -o /dev/null -w "%{http_code}" localhost:8000/health)
- Canary: $(curl -s localhost:8000/api/admin/canary/health | jq -r .status)

## Recent Changes
$(git log --oneline -10)

## Action Items
- [ ] Review error logs
- [ ] Check monitoring dashboards
- [ ] Verify Redis connectivity
- [ ] Test with canary user
- [ ] Check rate limits

EOF

echo "✅ Incident response complete - Report: $LOG_DIR/incident_report.md"
```

---

## Appendix: Quick Commands

### Most Used Commands

```bash
# Check V2 adoption rate
redis-cli GET "feature_flags:api_key_v2" | jq '.metrics.v2_percentage'

# Force expire a specific key
redis-cli DEL "api_key:v2:USER_KEY_ID"

# View recent revocations
redis-cli --scan --pattern "revocation_audit:*" | tail -10

# Test as canary user
curl -H "Authorization: Bearer CANARY_KEY" localhost:8000/api/chat

# Emergency disable V2
redis-cli SET "feature_flags:api_key_v2" '{"strategy": "off"}'

# Check JWT secret version
curl localhost:8000/api/admin/secrets/status | jq .active.version_id

# Manual canary test
curl -X POST localhost:8000/api/admin/canary/run

# Export metrics for analysis
curl localhost:8000/metrics > metrics_$(date +%Y%m%d).txt
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2024-01-08  
**Review Schedule**: Quarterly  
**Owner**: Platform Engineering Team

**For questions or updates, contact**: platform-eng@blyan.com