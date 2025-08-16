# API Key Management - Testing Checklist

## Pre-Deployment Testing

### 1. Unit Tests (Backend)

#### Role Management
- [ ] Test all role string variations (basic, contributor, node_operator, admin)
- [ ] Test role normalization (node-operator → node_operator)
- [ ] Test invalid role rejection
- [ ] Test role hierarchy enforcement

#### Key Generation
- [ ] Test JWT generation with correct claims
- [ ] Test TTL assignment per role
- [ ] Test metadata inclusion
- [ ] Test unique JTI generation

#### Key Validation
- [ ] Test valid key acceptance
- [ ] Test expired key rejection (401)
- [ ] Test malformed key rejection (401)
- [ ] Test revoked key rejection (401)
- [ ] Test wrong issuer rejection

#### Permission Checks
- [ ] Test scope-based authorization
- [ ] Test role-based authorization
- [ ] Test key creation permissions matrix
- [ ] Test 403 for insufficient permissions

#### Refresh Flow
- [ ] Test refresh within window (success)
- [ ] Test refresh too early (400)
- [ ] Test refresh with expired key (401)
- [ ] Test refresh with invalid key (401)
- [ ] Test old key revocation after refresh

#### Rate Limiting
- [ ] Test rate limit enforcement per role
- [ ] Test rate limit headers in response
- [ ] Test 429 response when exceeded
- [ ] Test unlimited for admin role

### 2. Integration Tests (Backend + Redis)

#### Redis Operations
- [ ] Test key storage with TTL
- [ ] Test key retrieval
- [ ] Test key revocation (deletion)
- [ ] Test Redis connection failure handling
- [ ] Test Redis reconnection logic

#### Distributed Scenarios
- [ ] Test multiple backend nodes sharing Redis
- [ ] Test key validation across nodes
- [ ] Test rate limiting across nodes
- [ ] Test concurrent refresh attempts

### 3. Frontend Tests

#### Storage Management
- [ ] Test credential storage in localStorage
- [ ] Test credential retrieval on page load
- [ ] Test expired credential cleanup
- [ ] Test storage quota handling

#### Request Flow
- [ ] Test authenticated request with valid key
- [ ] Test free tier request without key
- [ ] Test automatic refresh on 401
- [ ] Test retry with exponential backoff
- [ ] Test request timeout handling

#### Error Handling
- [ ] Test 401 → refresh → retry flow
- [ ] Test 403 → show permission error
- [ ] Test 429 → show rate limit error
- [ ] Test 500 → show server error
- [ ] Test network error → retry logic

#### UI Integration
- [ ] Test auth status display
- [ ] Test role-based UI elements
- [ ] Test error notifications
- [ ] Test loading states during refresh

### 4. End-to-End Tests

#### User Journeys

##### Free Tier User
```
1. Visit site without auth
2. Use chat (should work)
3. Try to access history (should fail gracefully)
4. See upgrade prompt after 5 messages
```

##### Basic User Registration
```
1. Click "Generate API Key"
2. Receive basic key
3. Key stored in browser
4. Can access chat with history
5. Rate limited after 100 requests/hour
```

##### Contributor Upgrade
```
1. Basic user requests contributor key
2. Admin approves upgrade
3. New key issued with contributor role
4. Can upload datasets
5. Higher rate limits applied
```

##### Node Operator Flow
```
1. Register as node operator
2. Receive node_operator key
3. Configure environment variables
4. Start GPU node with key
5. Node registers successfully
6. Can manage node operations
```

##### Key Refresh Flow
```
1. Use key normally
2. 24 hours before expiry
3. Automatic refresh triggered
4. Old key revoked
5. New key stored
6. Seamless continuation
```

### 5. Security Tests

#### Authentication
- [ ] Test Bearer token extraction
- [ ] Test missing auth header handling
- [ ] Test malformed auth header
- [ ] Test SQL injection in key
- [ ] Test XSS in metadata

#### Authorization
- [ ] Test privilege escalation attempts
- [ ] Test scope bypass attempts
- [ ] Test role spoofing
- [ ] Test replay attacks
- [ ] Test timing attacks

#### Cryptography
- [ ] Test JWT signature validation
- [ ] Test key hashing in logs
- [ ] Test secret rotation
- [ ] Test algorithm confusion

### 6. Performance Tests

#### Load Testing
- [ ] Test 1000 concurrent validations
- [ ] Test 10000 keys in Redis
- [ ] Test refresh under load
- [ ] Test rate limiter accuracy
- [ ] Measure validation latency

#### Stress Testing
- [ ] Test Redis memory limits
- [ ] Test backend CPU usage
- [ ] Test network saturation
- [ ] Test cascading failures

### 7. Environment-Specific Tests

#### Development
```bash
# Test with local Redis
export REDIS_HOST=localhost
export JWT_SECRET=dev-secret
npm test
```

#### Staging
```bash
# Test with staging Redis cluster
export REDIS_HOST=redis-staging.example.com
export JWT_SECRET=$STAGING_SECRET
npm run test:staging
```

#### Production
```bash
# Smoke tests only
export ENVIRONMENT=production
npm run test:smoke
```

## Monitoring Checklist

### Metrics to Track
- [ ] API key creation rate
- [ ] API key validation latency (p50, p95, p99)
- [ ] Refresh success rate
- [ ] 401/403 error rate
- [ ] Rate limit violations
- [ ] Redis connection errors
- [ ] JWT decode failures

### Alerts to Configure
- [ ] High 401 rate (> 1% of requests)
- [ ] High 403 rate (> 0.5% of requests)
- [ ] Refresh failures (> 5 in 5 minutes)
- [ ] Redis disconnection
- [ ] Unusual key creation spike
- [ ] Rate limit bypass attempts

### Dashboards
- [ ] API key lifecycle funnel
- [ ] Role distribution pie chart
- [ ] Request rate by role
- [ ] Error rate by endpoint
- [ ] Geographic distribution
- [ ] Latency heatmap

## Rollback Plan

### If Issues Occur

1. **Immediate Actions**
   ```bash
   # Revert to previous version
   kubectl rollout undo deployment/api
   
   # Clear Redis cache if corrupted
   redis-cli FLUSHDB
   
   # Restore from backup if needed
   ./scripts/restore_api_keys.sh
   ```

2. **Communication**
   - [ ] Status page update
   - [ ] Customer notification
   - [ ] Internal incident channel

3. **Recovery**
   - [ ] Validate all services healthy
   - [ ] Test key generation
   - [ ] Test key validation
   - [ ] Monitor error rates

## Sign-off Checklist

### Before Production Deployment

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Security review completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Rollback plan tested
- [ ] Team training completed

### Post-Deployment Verification

- [ ] Health checks passing
- [ ] Key generation working
- [ ] Key validation working
- [ ] Rate limiting active
- [ ] Metrics flowing
- [ ] No elevated error rates
- [ ] Customer reports monitored

---

**Note**: This checklist should be reviewed and updated after each major release or incident.