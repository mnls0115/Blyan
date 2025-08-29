# Cloudflare Configuration for Blyan Proxy

## Overview

When using Cloudflare with the Blyan dynamic proxy, specific configurations are needed to ensure SSE (Server-Sent Events), WebSockets, and long-running API calls work correctly.

## Recommended Configuration

### Option 1: Bypass Cloudflare for API (Recommended for SSE/WebSocket)

Create a DNS-only subdomain for API traffic:

1. **Add DNS Record**:
   - Type: A
   - Name: `api`
   - Content: Your DigitalOcean IP
   - Proxy status: **DNS only (grey cloud)**

2. **Update Frontend Config**:
   ```javascript
   // frontend/config.js
   API_CONFIG.baseURL = 'https://api.blyan.com';
   ```

3. **Update Nginx**:
   ```nginx
   server_name api.blyan.com;
   ```

### Option 2: Configure Cloudflare Page Rules

If you must proxy API traffic through Cloudflare:

1. **Create Page Rule** for `*blyan.com/api/*`:
   - Cache Level: **Bypass**
   - Disable Performance: **ON**
   - Disable Security: **OFF** (keep security)
   - Disable Apps: **ON**
   - Rocket Loader: **OFF**
   - Mirage: **OFF**
   - Polish: **OFF**

2. **Configure Cloudflare Settings**:
   ```
   Speed → Optimization:
   - Auto Minify: Uncheck JavaScript for /api/*
   - Rocket Loader: OFF for /api/*
   - Brotli: ON (safe for streams)
   
   Caching → Configuration:
   - Browser Cache TTL: Respect Existing Headers
   - Always Online: OFF for /api/*
   
   Network:
   - WebSockets: ON
   - HTTP/2: ON
   - HTTP/3 (QUIC): ON (optional)
   ```

3. **Add Transform Rule** (Business plan or higher):
   ```
   When: URI Path starts with "/api/"
   Then: 
   - Add Header: X-Accel-Buffering = "no"
   - Add Header: Cache-Control = "no-cache, no-store"
   ```

## SSE-Specific Configuration

### Cloudflare Workers (Optional Enhancement)

Create a Worker to handle SSE streams properly:

```javascript
// cloudflare-worker.js
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // Only process API requests
  if (!url.pathname.startsWith('/api/')) {
    return fetch(request)
  }
  
  // Forward to origin with SSE headers
  const response = await fetch(request, {
    cf: {
      // Disable caching
      cacheTtl: 0,
      cacheEverything: false,
      // Disable Cloudflare features that interfere with streams
      mirage: false,
      polish: 'off',
      minify: false,
      apps: false
    }
  })
  
  // Clone response to modify headers
  const newResponse = new Response(response.body, response)
  
  // Add SSE-friendly headers
  newResponse.headers.set('Cache-Control', 'no-cache')
  newResponse.headers.set('X-Accel-Buffering', 'no')
  
  // Remove Cloudflare headers that might interfere
  newResponse.headers.delete('CF-Cache-Status')
  newResponse.headers.delete('CF-RAY')
  
  return newResponse
}
```

Deploy to route: `*blyan.com/api/*`

## WebSocket Configuration

### Required Settings

1. **Network Tab**:
   - WebSockets: **Enabled** ✓

2. **Page Rule** for WebSocket endpoints:
   ```
   URL: *blyan.com/api/ws/*
   Settings:
   - Cache Level: Bypass
   - Disable Performance: ON
   ```

3. **Nginx WebSocket Headers** (already configured):
   ```nginx
   proxy_set_header Upgrade $http_upgrade;
   proxy_set_header Connection $connection_upgrade;
   ```

## Monitoring & Debugging

### Test SSE Connection

```bash
# Direct (bypassing Cloudflare)
curl -N -H "Accept: text/event-stream" \
  https://YOUR-DO-IP:443/api/queue/stream

# Through Cloudflare
curl -N -H "Accept: text/event-stream" \
  https://blyan.com/api/queue/stream

# Check headers
curl -I https://blyan.com/api/health
```

### Expected Headers for SSE

Good response should include:
```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
Connection: keep-alive
```

Should NOT include:
```
CF-Cache-Status: HIT  (should be DYNAMIC or BYPASS)
Content-Encoding: br   (might break SSE)
```

### Debug Cloudflare Issues

1. **Check Ray ID**: 
   ```bash
   curl -I https://blyan.com/api/health | grep CF-RAY
   ```
   Then lookup in Cloudflare dashboard → Analytics → Ray ID Lookup

2. **Test with CF bypassed**:
   ```bash
   curl --resolve blyan.com:443:YOUR-DO-IP https://blyan.com/api/health
   ```

3. **Check WebSocket upgrade**:
   ```bash
   wscat -c wss://blyan.com/api/ws
   ```

## Performance Considerations

### Cloudflare Pros:
- DDoS protection
- Global CDN for static assets
- SSL termination
- Analytics

### Cloudflare Cons for API:
- Adds 20-50ms latency
- 100-second timeout limit (Enterprise: 600s)
- SSE buffering issues on Free/Pro plans
- WebSocket connection limits

### Hybrid Approach (Recommended)

```
Static assets (HTML/CSS/JS): blyan.com → Cloudflare → Nginx
API traffic: api.blyan.com → Direct to Nginx (DNS-only)
```

This gives you Cloudflare benefits for static content while maintaining direct, low-latency API access.

## Timeout Limits

| Component | Default Timeout | Configurable |
|-----------|----------------|--------------|
| Cloudflare Free/Pro | 100 seconds | No |
| Cloudflare Business | 300 seconds | No |  
| Cloudflare Enterprise | 600 seconds | Yes (up to 6000s) |
| Nginx (configured) | 600 seconds | Yes |
| Node.js Proxy | 600 seconds | Yes |

For long-running operations beyond Cloudflare limits, consider:
1. Polling with job IDs
2. WebSocket connections (no timeout)
3. SSE with heartbeat events every 30s

## Troubleshooting Checklist

- [ ] WebSockets enabled in Cloudflare Network settings
- [ ] Page rule created for /api/* with Cache Bypass
- [ ] Rocket Loader disabled for API routes
- [ ] SSL/TLS mode set to Full (strict)
- [ ] Browser Integrity Check not blocking API clients
- [ ] Rate limiting rules not too aggressive
- [ ] Transform rules adding proper headers (Business+ plan)
- [ ] Worker deployed for SSE handling (optional)
- [ ] DNS-only mode for api subdomain (if using Option 1)

## Testing Script

Save as `test-cloudflare.sh`:

```bash
#!/bin/bash
DOMAIN="blyan.com"
API_DOMAIN="api.blyan.com"

echo "Testing Cloudflare configuration..."

# Test main domain (should be proxied)
echo -n "Main domain CF: "
curl -s -I https://$DOMAIN | grep -q "CF-RAY" && echo "✓ Proxied" || echo "✗ Not proxied"

# Test API subdomain (should be direct)
echo -n "API domain CF: "
curl -s -I https://$API_DOMAIN | grep -q "CF-RAY" && echo "✗ Proxied (not recommended)" || echo "✓ Direct"

# Test SSE headers
echo -n "SSE headers: "
HEADERS=$(curl -s -I https://$API_DOMAIN/queue/stream)
echo "$HEADERS" | grep -q "X-Accel-Buffering: no" && echo "✓ Correct" || echo "✗ Missing"

# Test WebSocket upgrade
echo -n "WebSocket: "
curl -s -I -H "Upgrade: websocket" https://$API_DOMAIN/ws | grep -q "HTTP/1.1 101" && echo "✓ Works" || echo "✗ Failed"

echo "Done!"
```

## Support Contacts

- **Cloudflare Issues**: Use Ray ID when contacting support
- **SSE/WebSocket**: Consider Enterprise plan for better streaming support
- **Alternative**: Use dedicated streaming service (Pusher, Ably) for real-time features