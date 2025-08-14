# Blyan Security Setup Guide

## 🔒 Main Node Protection System

This guide explains how to secure your Blyan main node and prevent unauthorized takeover attempts.

## Quick Start

### 1. Initial Setup (Digital Ocean Server ONLY)

```bash
# On your main server (e.g., Digital Ocean)
export BLYAN_MAIN_NODE_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
python scripts/setup_main_node.py
```

Save the generated token securely - you'll need it for all operations!

### 2. Configure Environment

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env` with your values:
```env
BLYAN_MAIN_NODE_SECRET=<your-secret>
BLYAN_MAIN_NODE_TOKEN=<your-token-from-setup>
BLYAN_NODE_ID=main-do-nyc1
BLYAN_ENFORCE_AUTH=true
```

### 3. Start Protected Server

```bash
source .env  # Load environment variables
./server.sh start
```

## Security Features Implemented

### ✅ Authentication Enforcement
- **Chain Write Protection**: `/mine`, `/upload_parameters`, `/upload_moe_experts` require main node token
- **P2P Worker Protection**: All worker endpoints require authentication
- **Automatic Header Injection**: Coordinator automatically includes auth headers

### ✅ Attack Prevention
- **Hostile Takeover Detection**: Logs attempts to claim main node role
- **Unauthorized Access Logging**: All failed auth attempts are logged to `logs/security_events.json`
- **Token Verification**: Cryptographic verification of node identity

### ✅ Configuration Security
- **Secret Management**: Tokens never stored in git
- **Environment Isolation**: Separate configs for dev/staging/production
- **Audit Trail**: Complete security event logging

## How It Works

### Authentication Flow
```
1. Main Node Registration
   └─> Generates unique auth token
   
2. API Request
   └─> Includes X-Node-ID and X-Node-Auth-Token headers
   
3. Token Verification
   └─> Checks against stored hash
   
4. Access Control
   └─> Allow/Deny based on node role
```

### Node Roles
- **main**: Can write blocks, coordinate workers
- **validator**: Can validate but not write
- **worker**: Can only contribute compute

## Monitoring Security

Check security events:
```bash
tail -f logs/security_events.json | jq '.'
```

Common event types:
- `hostile_takeover_attempt`: Someone tried to claim main node
- `unauthorized_api_access`: Invalid token for protected endpoint
- `unauthorized_worker_access`: Invalid access to worker node

## Network-Level Protection (Recommended)

### 1. IP Allowlist (Digital Ocean Firewall)
```bash
# Allow only your worker nodes
ufw allow from <worker-ip-1> to any port 8000
ufw allow from <worker-ip-2> to any port 8000
ufw deny 8000
```

### 2. VPC/Private Network
- Keep main node ↔ worker communication on private network
- Only expose public API endpoints

### 3. TLS/SSL (Production)
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    # ... rest of config
}
```

#### Pipeline RPC TLS/mTLS

- Set `BLYAN_TLS_CERT` to a CA bundle path to enable TLS verification for pipeline HTTP clients.
- For mTLS, also set `BLYAN_TLS_CLIENT_CERT` and `BLYAN_TLS_CLIENT_KEY` on the client side.
- You can automate certificate obtain/renew with `scripts/ssl_manager.py` which installs certbot and configures nginx.

## Testing Authentication

### Test Protected Endpoint (Should Fail)
```bash
curl -X POST http://localhost:8000/mine \
  -H "Content-Type: application/json" \
  -d '{"data": "test"}'
# Expected: 401 Unauthorized
```

### Test With Valid Token
```bash
curl -X POST http://localhost:8000/mine \
  -H "Content-Type: application/json" \
  -H "X-Node-ID: main-do-nyc1" \
  -H "X-Node-Auth-Token: <your-token>" \
  -d '{"data": "test"}'
# Expected: Success
```

## Troubleshooting

### "Missing authentication headers"
- Ensure BLYAN_NODE_ID and BLYAN_MAIN_NODE_TOKEN are set
- Check .env file is loaded: `echo $BLYAN_NODE_ID`

### "Invalid main node credentials"
- Token doesn't match registered token
- Run `setup_main_node.py` again if needed

### "No main node registered"
- Run initial setup: `python scripts/setup_main_node.py`

## Emergency Procedures

### Token Compromised
1. Delete `config/node_auth.json`
2. Re-run `setup_main_node.py` with new secret
3. Update all worker nodes with new token

### Disable Auth (Development ONLY)
```bash
export BLYAN_ENFORCE_AUTH=false
```

⚠️ **NEVER disable auth in production!**

## Summary

Your main node is now protected against:
- ✅ Unauthorized blockchain writes
- ✅ Hostile takeover attempts
- ✅ Worker node hijacking
- ✅ API abuse

Next steps:
1. Set up network-level protection (firewall/VPC)
2. Enable TLS/SSL for production
3. Monitor security events regularly
4. Rotate tokens periodically