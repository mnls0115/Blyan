# Proxy Validation Checklist

## Quick Test Commands

### 1. Test Node Addition
```bash
# Add your Runpod node
curl -X POST http://localhost:9000/_admin/nodes \
  -H "X-Admin-Token: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "runpod-primary",
    "baseURL": "https://YOUR-RUNPOD-URL.proxy.runpod.net:8000"
  }'
```

### 2. Check Status
```bash
# Public status (no auth)
curl http://localhost:9000/_status | jq

# Admin status (with metrics)
curl -H "X-Admin-Token: YOUR_TOKEN" \
  http://localhost:9000/_admin/status | jq
```

### 3. Test Health Checks
```bash
# Watch health check logs
journalctl -u blyan-proxy -f | grep "healthy\|unhealthy"
```

### 4. Test API Routing
```bash
# Through proxy (will route to healthy node)
curl -X POST http://localhost:9000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# With target override (bypass selection)
curl -X POST "http://localhost:9000/api/chat?target=https://specific-node.com:8000" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

### 5. Test SSE Stream
```bash
# Start SSE stream
curl -N -H "Accept: text/event-stream" \
  http://localhost:9000/api/queue/stream?ticket_id=YOUR_TICKET
```

### 6. Test WebSocket
```bash
# Install wscat if needed
npm install -g wscat

# Test WebSocket connection
wscat -c ws://localhost:9000/api/ws
```

## Validation Steps

### Initial Setup
- [ ] Proxy starts without errors
- [ ] Admin token generated and saved
- [ ] State file created at `./data/proxy_state.json`

### Node Management
- [ ] Can add nodes via admin API
- [ ] Nodes appear in `/_status` endpoint
- [ ] State persists across restarts
- [ ] Can remove nodes
- [ ] Can set active node index

### Health Checks
- [ ] Health checks run every 5 seconds
- [ ] Nodes marked healthy/unhealthy correctly
- [ ] Response times tracked
- [ ] Failed nodes marked unhealthy

### Request Routing
- [ ] Requests route to healthy nodes
- [ ] Failover works when active node dies
- [ ] Round-robin policy distributes evenly
- [ ] Least-latency routes to fastest node
- [ ] Target override parameter works

### SSE Support
- [ ] SSE headers set correctly
- [ ] No buffering on streams
- [ ] Connections stay alive
- [ ] Events flow through proxy

### WebSocket Support
- [ ] WebSocket upgrades work
- [ ] Connections stay alive
- [ ] Heartbeat keeps connections open
- [ ] Multiple WS connections supported

### Error Handling
- [ ] 503 when no nodes available
- [ ] 502 when upstream fails
- [ ] Graceful degradation
- [ ] Errors logged properly

### Performance
- [ ] Handles concurrent requests
- [ ] Memory usage stable
- [ ] CPU usage reasonable
- [ ] No memory leaks over time

## Load Testing

### Basic Load Test
```bash
# Install Apache Bench if needed
apt-get install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -p test.json -T application/json \
  http://localhost:9000/api/chat
```

### SSE Load Test
```bash
# Run 10 SSE connections in parallel
for i in {1..10}; do
  curl -N http://localhost:9000/api/queue/stream &
done
```

## Monitoring

### Check Metrics
```bash
# Node metrics
curl http://localhost:9000/_admin/status | jq '.metrics'

# Request counts
curl http://localhost:9000/_admin/status | jq '.metrics.totalRequests'

# Failed requests
curl http://localhost:9000/_admin/status | jq '.metrics.failedRequests'
```

### Watch Logs
```bash
# Proxy logs
journalctl -u blyan-proxy -f

# Filter for errors
journalctl -u blyan-proxy -f | grep ERROR

# Filter for specific node
journalctl -u blyan-proxy -f | grep "runpod"
```

## Troubleshooting

### Proxy Won't Start
1. Check port 9000 is free: `lsof -i:9000`
2. Check Node.js version: `node --version` (need v16+)
3. Check logs: `journalctl -u blyan-proxy -n 50`

### Nodes Always Unhealthy
1. Check node URLs are accessible
2. Test health endpoint directly: `curl NODE_URL/health`
3. Check timeout settings (increase if needed)
4. Verify health paths in config

### SSE Not Working
1. Check Nginx buffering is disabled
2. Verify headers: `curl -I http://localhost:9000/api/queue/stream`
3. Test without proxy: direct to node
4. Check Cloudflare settings if applicable

### WebSocket Failures
1. Check Nginx upgrade headers
2. Test with wscat directly
3. Check firewall rules
4. Verify WebSocket endpoint on nodes

## Production Readiness

### Security
- [ ] Admin token is strong (32+ chars)
- [ ] Admin endpoints IP-restricted in Nginx
- [ ] HTTPS configured with valid cert
- [ ] No sensitive data in logs

### Reliability
- [ ] Systemd service auto-restarts
- [ ] State persists across restarts
- [ ] Multiple nodes configured
- [ ] Failover tested and works

### Performance
- [ ] Can handle expected load
- [ ] Response times acceptable
- [ ] Memory usage bounded
- [ ] CPU usage reasonable

### Monitoring
- [ ] Metrics endpoint accessible
- [ ] Logs aggregated (if applicable)
- [ ] Alerts configured (optional)
- [ ] Health checks monitored