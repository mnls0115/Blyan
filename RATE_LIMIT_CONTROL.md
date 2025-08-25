# Rate Limit Data Control Guide

## How Rate Limiting Works

### 1. **Data Storage Locations**

Rate limit data is stored in TWO places:

#### Redis (Primary - if available)
```bash
# Keys pattern:
rate_limit:ip:222.118.97.244:/chat    # IP-based limits
rate_limit:wallet:0x123...:/chat      # Wallet-based limits
ssot:hourly:222.118.97.244           # SSOT hourly limits
ssot:daily:222.118.97.244            # SSOT daily limits
abuse:222.118.97.244                 # Abuse detection
```

#### File System (Fallback)
```bash
/root/blyan/data/rate_limit/*.json   # File-based storage
/root/blyan/data/*.json              # Legacy location
```

### 2. **How Data is Created**

When you make a request to `/chat`:

```python
# In backend/core/rate_limiter.py
1. Get client IP from request
2. Check if Redis is available
3. If Redis:
   - Store: redis.setex(f"rate_limit:{ip}:/chat", 18000, timestamps)
4. If no Redis:
   - Store: data/rate_limit/{hash}.json with timestamps
```

### 3. **Rate Limit Rules**

- **Free Tier**: 20 requests per 5 hours (18,000 seconds)
- **Premium Tier**: 1000 requests per hour
- **Window Type**: Sliding window (not fixed)

### 4. **Manual Control Commands**

#### Check Current Rate Limit Data
```bash
# On server
ssh root@165.227.221.225

# Check Redis
redis-cli --scan --pattern "rate_limit:*"
redis-cli GET "rate_limit:ip:YOUR_IP:/chat"

# Check files
ls -la /root/blyan/data/rate_limit/
cat /root/blyan/data/rate_limit/*.json
```

#### Clear Specific IP
```bash
# Clear specific IP from Redis
redis-cli DEL "rate_limit:ip:222.118.97.244:/chat"

# Clear all rate limits for an IP
redis-cli --scan --pattern "rate_limit:ip:222.118.97.244:*" | xargs redis-cli DEL
```

#### Clear ALL Rate Limits
```bash
# Nuclear option - clear everything
redis-cli FLUSHDB  # Clears entire Redis database
rm -rf /root/blyan/data/rate_limit/*
```

### 5. **Why You Got Rate Limited**

The issue was **corrupted timestamps**:
- Server's Python `time.time()` returned year 2055 timestamps
- These timestamps (1756130646) were 30 years in the future
- When checking limits, all requests appeared "recent"
- Result: Instant rate limit even with 1 request

### 6. **Prevention**

To prevent this issue:

```python
# Add timestamp validation in rate_limiter.py
def _add_request(self, client_id: str, endpoint: str, timestamp: float):
    # Sanity check - timestamp should be reasonable
    current_year_approx = 1724000000  # 2024
    max_future = current_year_approx + 31536000 * 2  # Max 2 years future
    
    if timestamp > max_future:
        logger.error(f"Invalid timestamp {timestamp} - using current time")
        timestamp = time.time()
```

### 7. **Monitoring**

Check rate limit status without triggering limits:
```bash
# From client
curl https://blyan.com/api/rate-limit/status

# On server - watch Redis
redis-cli MONITOR  # See all Redis commands in real-time
```

### 8. **Configuration**

Rate limits are configured in:
- `/root/blyan/backend/core/rate_limiter.py` - Main limits
- `/root/blyan/api/server.py` - Middleware enforcement
- Environment variables can override:
  ```bash
  export RATE_LIMIT_FREE_TIER=50      # Change free tier limit
  export RATE_LIMIT_WINDOW_HOURS=1    # Change window to 1 hour
  ```

## Quick Fix Commands

```bash
# Complete reset (copy-paste to server)
redis-cli FLUSHDB && \
rm -rf /root/blyan/data/rate_limit/* && \
systemctl restart blyan && \
echo "✅ All rate limits cleared!"
```

## Testing

After clearing, test with:
```bash
# Should work now
curl -X POST https://blyan.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```