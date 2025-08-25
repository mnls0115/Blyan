# Server Recovery Guide

## Quick Recovery Steps

When the server fails to start or `/health` returns 500:

### 1. Immediate Recovery (Production Server)

```bash
# SSH to server
ssh root@165.227.221.225

# Kill stray processes
pkill -f uvicorn
pkill -9 -f "python.*api.server"

# Start in minimal mode
cd /root/blyan
./scripts/minimal_start.sh
```

### 2. Using Environment Flags

Set these environment variables to disable problematic subsystems:

```bash
export BLYAN_MINIMAL_MODE=true           # Master switch for minimal mode
# Blockchain is always enabled for GPU nodes
export DISABLE_PIPELINE_ROUND=true       # Disable pipeline round service
export DISABLE_GRPC=true                 # Disable gRPC services
export P2P_ENABLE=false                  # Disable P2P coordination
export BLYAN_DISABLE_SECURITY_MONITOR=true  # Disable security monitoring
export SKIP_POL=true                     # Skip Proof-of-Learning
export ENABLE_POL=false                  # Disable PoL validation
```

### 3. Systemd Service Recovery

```bash
# Restart with minimal mode
systemctl stop blyan
export BLYAN_MINIMAL_MODE=true
systemctl start blyan

# Check logs
journalctl -u blyan -f

# Verify health endpoint
curl http://localhost:8000/health
```

## Diagnostics

### Check Current Issues

```bash
# View recent errors
tail -n 200 /root/blyan/api.log | grep -E "ERROR|Exception|Traceback"

# Check systemd logs
journalctl -u blyan -n 100 --no-pager

# Test health endpoint
curl -v http://localhost:8000/health

# Check port usage
lsof -i:8000

# System resources
htop
df -h
free -h
```

### API Schema Validation

```bash
# Validate OpenAPI schema
curl -s http://localhost:8000/openapi.json | python -m json.tool | head -20

# Check metrics endpoint
curl -s http://localhost:8000/metrics | head -20
```

## Gradual Service Restoration

Once the server is running in minimal mode:

### Step 1: Enable Blockchain Only
```bash
export BLYAN_MINIMAL_MODE=false
# Blockchain is always enabled
systemctl restart blyan
```

### Step 2: Enable P2P (if needed)
```bash
export P2P_ENABLE=true
systemctl restart blyan
```

### Step 3: Enable Security Monitoring
```bash
export BLYAN_DISABLE_SECURITY_MONITOR=false
systemctl restart blyan
```

### Step 4: Full Service
```bash
# Remove all restrictive flags
unset BLYAN_MINIMAL_MODE
# Blockchain is always enabled (no need to unset)
unset DISABLE_PIPELINE_ROUND
unset DISABLE_GRPC
unset P2P_ENABLE
unset BLYAN_DISABLE_SECURITY_MONITOR
systemctl restart blyan
```

## Prevention Measures

### 1. Health Check Monitoring

Add to monitoring system:
```bash
# Cron job for health monitoring
*/5 * * * * curl -f http://localhost:8000/health || systemctl restart blyan
```

### 2. Automatic Recovery Script

Create `/root/blyan/scripts/auto_recovery.sh`:
```bash
#!/bin/bash
# Auto-recovery script

HEALTH_URL="http://localhost:8000/health"
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "Health check passed"
        exit 0
    fi
    
    echo "Health check failed, attempt $((RETRY_COUNT + 1))"
    
    # Progressive degradation
    if [ $RETRY_COUNT -eq 0 ]; then
        systemctl restart blyan
    elif [ $RETRY_COUNT -eq 1 ]; then
        export P2P_ENABLE=false
        systemctl restart blyan
    else
        export BLYAN_MINIMAL_MODE=true
        systemctl restart blyan
    fi
    
    sleep 30
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo "Failed to recover after $MAX_RETRIES attempts"
exit 1
```

### 3. Systemd Service Hardening

Update `/etc/systemd/system/blyan.service`:
```ini
[Service]
# ... existing config ...
Restart=always
RestartSec=5
StartLimitBurst=5
StartLimitInterval=60

# Health check
ExecStartPost=/bin/sleep 10
ExecStartPost=/usr/bin/curl -f http://localhost:8000/health
```

## Long-term Improvements

### Implemented ✅
- [x] Health endpoint timeout handling (2-second timeout)
- [x] Graceful degraded mode support
- [x] Environment-based subsystem control
- [x] Minimal mode startup script
- [x] Integration tests for health endpoint

### Roadmap Items
- [ ] Implement circuit breakers for subsystems
- [ ] Add Prometheus metrics for init failures
- [ ] Create subsystem health dashboard
- [ ] Implement automatic failover to backup nodes
- [ ] Add distributed tracing for debugging
- [ ] Create chaos engineering tests
- [ ] Implement progressive rollout for updates

## Testing Recovery

Run integration tests:
```bash
cd /root/blyan
python tests/test_health_endpoint.py
```

Test minimal start script:
```bash
./scripts/minimal_start.sh &
sleep 5
curl http://localhost:8000/health
pkill -f "python.*api.server"
```

## Contact for Critical Issues

If recovery fails after all attempts:
1. Check GitHub issues: https://github.com/mnls0115/Blyan/issues
2. Review recent commits for breaking changes
3. Rollback to last known good commit if necessary:
   ```bash
   git log --oneline -10
   git checkout <good-commit-hash>
   systemctl restart blyan
   ```