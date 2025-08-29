# Blyan Dynamic API Proxy

Production-ready reverse proxy for managing multiple GPU nodes with automatic failover, health checks, and SSE support.

## Features

- **Dynamic node registry** - Add/remove nodes without restart
- **Health checks** - Automatic monitoring and failover
- **Multiple selection policies** - Active-first, round-robin, least-latency
- **SSE/WebSocket support** - Full streaming capability
- **Admin API** - Secure management endpoints
- **State persistence** - Survives restarts
- **Metrics** - Request tracking and node statistics

## Quick Start

### Local Development

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start proxy
npm start
```

### Production Deployment (DigitalOcean/Linux)

```bash
# Run deployment script as root
sudo ./deploy.sh
```

The script will:
1. Install Node.js, Nginx, and dependencies
2. Create system user and directories
3. Configure the proxy service
4. Setup Nginx reverse proxy
5. Optional: Configure SSL with Let's Encrypt

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_PORT` | 9000 | Port for proxy server |
| `ADMIN_TOKEN` | (generated) | Authentication token for admin API |
| `SELECTION_POLICY` | active-first | Node selection strategy |
| `HEALTH_INTERVAL_MS` | 5000 | Health check frequency |
| `STATE_FILE` | ./data/proxy_state.json | Persistent state location |

### Initial Nodes

Add nodes via environment:
```bash
NODE_0=primary|https://gpu1.example.com:8000
NODE_1=backup|https://gpu2.example.com:8000
```

## API Endpoints

### Public Endpoints

- `GET /health` - Proxy health status
- `GET /_status` - Basic node status

### Admin Endpoints (requires X-Admin-Token header)

- `GET /_admin/status` - Full status with metrics
- `POST /_admin/nodes` - Add new node
- `DELETE /_admin/nodes/:id` - Remove node
- `POST /_admin/active/:index` - Set preferred node
- `POST /_admin/health-check` - Force health check

### Add Node Example

```bash
curl -X POST http://localhost:9000/_admin/nodes \
  -H "X-Admin-Token: YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "runpod-gpu",
    "baseURL": "https://abc123.proxy.runpod.net:8000",
    "metadata": {"gpu": "RTX 6000"}
  }'
```

## Node Selection Policies

### active-first (default)
Uses the designated active node if healthy, falls back to first healthy node.

### round-robin
Distributes requests evenly across all healthy nodes.

### least-latency
Routes to the node with lowest response time.

## Frontend Integration

No changes needed! Frontend continues to use `/api/*` endpoints:

```javascript
// Frontend code remains unchanged
fetch('/api/chat', {
  method: 'POST',
  body: JSON.stringify({ prompt: 'Hello' })
});
```

The proxy automatically routes to the best available node.

### Override Option

Keep the `?api=` parameter for direct node access:
```
https://blyan.com/chat.html?api=https://direct-node.example.com
```

## Monitoring

### View Logs
```bash
# Systemd
journalctl -u blyan-proxy -f

# Docker
docker logs -f blyan-proxy
```

### Check Status
```bash
curl http://localhost:9000/_status | jq
```

### Metrics
```bash
curl -H "X-Admin-Token: YOUR_TOKEN" \
  http://localhost:9000/_admin/status | jq
```

## Architecture

```
[Cloudflare] → [Nginx] → [Node.js Proxy] → [GPU Nodes]
                ↓                 ↓
            [Frontend]     [Health Checks]
                           [Node Registry]
                           [Metrics]
```

## Troubleshooting

### Proxy won't start
- Check port 9000 is available
- Verify Node.js version >= 16
- Check logs: `journalctl -u blyan-proxy`

### Nodes showing unhealthy
- Verify node URLs are accessible
- Check health endpoints respond
- Review health check logs

### SSE not working
- Ensure Nginx buffering is disabled
- Check Cloudflare caching rules
- Verify X-Accel-Buffering headers

## Security

- Admin API requires authentication token
- Nginx can restrict admin endpoints by IP
- HTTPS recommended for production
- Tokens never logged or exposed

## License

MIT