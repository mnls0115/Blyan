# API Key Roles and Usage Guide

This document explains the different API key types in Blyan and their specific use cases, permissions, and implementation patterns.

## Overview

Blyan uses a role-based API key system to control access to different features and endpoints. Each key type has specific permissions and rate limits.

## API Key Types

### 1. `basic` - Basic User Keys
**Purpose**: General users for chat and basic inference  
**Permissions**: 
- Chat API access
- Basic inference requests
- Rate-limited free tier access

**Usage Example**:
```javascript
// Frontend - Basic chat usage
const response = await apiAuth.makeAuthenticatedRequest('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
        prompt: "Hello",
        use_moe: true
    })
});
```

**Rate Limits**: 5 free messages, then requires wallet/payment

---

### 2. `contributor` - Content Contributors
**Purpose**: Users who contribute datasets, models, or expert knowledge  
**Permissions**:
- All basic permissions
- Dataset upload and validation
- Expert model submissions
- Access to contributor rewards

**Usage Example**:
```javascript
// Dataset upload
const response = await fetch('/api/datasets/upload', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${contributorApiKey}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        dataset_name: "medical_qa_v1",
        stage: "validation",
        quality_tier: "community"
    })
});
```

**Rate Limits**: Higher limits, reward-based usage

---

### 3. `node_operator` - GPU Node Operators
**Purpose**: Users running GPU nodes for distributed inference  
**Permissions**:
- Node registration and management
- Expert model hosting
- P2P network participation
- Mining and reward collection

**Usage Example**:
```bash
# Environment setup for GPU nodes
export BLYAN_API_KEY=node_operator_key_here
export MAIN_SERVER_URL=https://blyan.com/api

# Docker deployment
docker run --gpus all --rm -it \
  -e BLYAN_API_KEY=${BLYAN_API_KEY} \
  -e MAIN_SERVER_URL=${MAIN_SERVER_URL} \
  ghcr.io/blyan-network/expert-node:latest
```

```python
# Python node registration
from backend.p2p.distributed_inference import register_node

await register_node({
    "node_id": "gpu_node_001", 
    "api_key": node_operator_key,
    "available_experts": ["layer0.expert0", "layer1.expert1"],
    "gpu_specs": {"memory": "24GB", "model": "RTX 4090"}
})
```

**Rate Limits**: No rate limits, incentivized usage

---

### 4. `admin` - System Administrators
**Purpose**: System management and maintenance  
**Permissions**:
- All system endpoints
- User management
- System configuration
- Emergency controls

**Usage Example**:
```javascript
// Admin-only operations
const response = await fetch('/api/admin/system/status', {
    headers: {
        'Authorization': `Bearer ${adminApiKey}`
    }
});

// Emergency shutdown
await fetch('/api/admin/emergency/pause', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${adminApiKey}`
    }
});
```

**Rate Limits**: Unlimited

---

## Key Generation

### Programmatic Generation

```javascript
// Generate different key types
const keyTypes = ['basic', 'contributor', 'node_operator', 'admin'];

for (const keyType of keyTypes) {
    const response = await fetch('/api/auth/register_api_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: `${keyType}-key-${Date.now()}`,
            key_type: keyType
        })
    });
    
    const { api_key } = await response.json();
    console.log(`${keyType} key:`, api_key);
}
```

### Manual Generation via Frontend

1. **Basic Users**: Automatic generation on first chat
2. **Contributors**: Via contribute page "Generate Node Key" button
3. **Node Operators**: Via contribute page or direct API call
4. **Admins**: Manual generation by existing admins

---

## Frontend Implementation Patterns

### 1. Conditional Authentication
```javascript
// api-auth.js pattern
async makeAuthenticatedRequest(url, options = {}) {
    if (!this.hasValidApiKey()) {
        // Free tier - no authentication needed
        return fetch(url, { ...options, headers: { 'Content-Type': 'application/json' }});
    }
    
    // Authenticated request with API key
    return fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.apiKey}`,
            ...options.headers
        }
    });
}
```

### 2. Role-Based UI
```javascript
// Show different UI based on API key type
function updateUIForRole() {
    const authStatus = apiAuth.getAuthStatus();
    
    if (authStatus.keyInfo?.type === 'node_operator') {
        showNodeOperatorDashboard();
    } else if (authStatus.keyInfo?.type === 'contributor') {
        showContributorFeatures();
    } else {
        showBasicChatInterface();
    }
}
```

---

## Backend Endpoint Protection

### Role-Based Decorators
```python
from backend.security.api_auth import require_role

@app.post("/datasets/upload")
@require_role(["contributor", "admin"])
async def upload_dataset(request: DatasetRequest):
    # Only contributors and admins can upload
    pass

@app.post("/p2p/register")
@require_role(["node_operator", "admin"])
async def register_node(request: NodeRequest):
    # Only node operators can register
    pass

@app.get("/admin/users")
@require_role(["admin"])
async def list_users():
    # Admin-only endpoint
    pass
```

### Rate Limiting by Role
```python
# Different rate limits per role
RATE_LIMITS = {
    "basic": "5/hour",
    "contributor": "100/hour", 
    "node_operator": "unlimited",
    "admin": "unlimited"
}
```

---

## Security Best Practices

### 1. Key Storage
- **Frontend**: Use secure localStorage with metadata
- **Backend**: Hash keys before database storage
- **Node Operators**: Environment variables only

### 2. Key Rotation
```javascript
// Automatic rotation for security
if (keyAge > 90 * 24 * 60 * 60 * 1000) { // 90 days
    await apiAuth.refreshApiKey();
}
```

### 3. Scope Limitation
- Keys should have minimal required permissions
- Use specific key types for specific functions
- Regular audit of key usage patterns

---

## Troubleshooting Common Issues

### 1. 403 Forbidden Errors
```javascript
// Check key type and permissions
const authStatus = apiAuth.getAuthStatus();
console.log('Key type:', authStatus.keyInfo?.type);
console.log('Trying to access endpoint requiring:', requiredRole);
```

### 2. Free Tier Limitations
```javascript
// Handle free tier gracefully
if (response.status === 403) {
    const usedChats = parseInt(localStorage.getItem('usedFreeChats') || '0');
    if (usedChats >= 5) {
        showUpgradePrompt();
    }
}
```

### 3. Node Registration Issues
```bash
# Verify environment variables
echo "API Key: ${BLYAN_API_KEY:0:10}..."
echo "Server URL: ${MAIN_SERVER_URL}"

# Test connectivity
curl -H "Authorization: Bearer $BLYAN_API_KEY" \
     "$MAIN_SERVER_URL/p2p/test"
```

---

## Migration Guide

### From No Auth to Role-Based Auth
1. Identify user types in your system
2. Generate appropriate keys for existing users
3. Update frontend to handle multiple key types
4. Implement graceful fallbacks for free users

### Key Type Changes
```javascript
// Upgrading from basic to contributor
async function upgradeToContributor(basicKey) {
    // Verify contribution requirements
    const eligibility = await checkContributorEligibility(basicKey);
    
    if (eligibility.qualified) {
        const newKey = await generateContributorKey();
        apiAuth.storeApiKey(newKey, { type: 'contributor' });
        showContributorFeatures();
    }
}
```

---

## Monitoring and Analytics

### Key Usage Tracking
```javascript
// Track key usage patterns
function logKeyUsage(endpoint, keyType, success) {
    analytics.track('api_key_usage', {
        endpoint,
        key_type: keyType,
        success,
        timestamp: new Date().toISOString()
    });
}
```

### Rate Limit Monitoring
```python
# Backend monitoring
@app.middleware("http")
async def track_rate_limits(request, call_next):
    key_type = get_key_type_from_request(request)
    
    # Log near-limit usage
    if is_approaching_rate_limit(key_type):
        logger.warning(f"Key type {key_type} approaching rate limit")
    
    return await call_next(request)
```

This guide provides comprehensive coverage of API key roles and usage patterns across the Blyan ecosystem.