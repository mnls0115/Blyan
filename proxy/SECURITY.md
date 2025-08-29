# Security Considerations for Blyan Proxy

## Admin Endpoint Security

### Current Configuration (Localhost Only)
The default Nginx configuration restricts `/_admin/` endpoints to localhost access only:

```nginx
location /_admin/ {
    allow 127.0.0.1;
    deny all;
    # ...
}
```

This is the **recommended configuration** for most deployments where admin operations are performed via SSH on the server.

### Remote Admin Access (If Needed)

If you need to access admin endpoints remotely, implement **defense in depth**:

#### 1. IP Allowlist (Nginx)
```nginx
location /_admin/ {
    # Your admin IPs
    allow 1.2.3.4;        # Office IP
    allow 5.6.7.8;        # Home IP  
    allow 127.0.0.1;      # Local
    deny all;
    
    # Still require token
    proxy_pass http://blyan_proxy;
}
```

#### 2. Strong Admin Token
- Use a cryptographically secure token (32+ characters)
- Generate with: `openssl rand -hex 32`
- Never commit to version control
- Rotate periodically

#### 3. HTTPS Only
- Admin endpoints should NEVER be accessed over HTTP
- Ensure SSL/TLS is properly configured
- Use strong ciphers and protocols (TLS 1.2+)

#### 4. Rate Limiting
Add rate limiting to prevent brute force:

```nginx
limit_req_zone $binary_remote_addr zone=admin:10m rate=5r/m;

location /_admin/ {
    limit_req zone=admin burst=5 nodelay;
    # ... rest of config
}
```

#### 5. Audit Logging
Log all admin operations:

```javascript
// In server.js admin endpoints
console.log(`[ADMIN] ${req.method} ${req.path} from ${req.ip} - User: ${req.headers['x-user-id'] || 'unknown'}`);
```

### Additional Hardening Options

#### VPN/Bastion Access
For maximum security, don't expose admin endpoints at all:
1. Access server via VPN or bastion host
2. Use SSH port forwarding: `ssh -L 9000:localhost:9000 server`
3. Access admin API locally: `curl http://localhost:9000/_admin/...`

#### Separate Admin Service
Run admin operations on a different port/service:
- Main proxy on port 9000 (no admin endpoints)
- Admin service on port 9001 (localhost only)
- Complete separation of concerns

#### mTLS (Mutual TLS)
For API-to-API admin access:
- Require client certificates for `/_admin/`
- Strong identity verification
- No passwords/tokens needed

## Token Security

### Storage
- **Environment Variables**: Use `.env` file with 600 permissions
- **Secrets Management**: Consider HashiCorp Vault, AWS Secrets Manager, etc.
- **Never**: Hardcode in source, commit to git, log in plaintext

### Transmission
- **Always use HTTPS** for token transmission
- **Authorization header** preferred over query parameters
- **Short-lived tokens** for temporary access (JWTs with expiry)

### Rotation
- Rotate admin tokens periodically
- Implement token versioning if needed
- Log token usage for audit trail

## Node Authentication

### Current Design
- Nodes authenticate to main service with `BLYAN_API_KEY`
- Token in Authorization header
- Each node has unique ID for tracking

### Best Practices
1. **Unique tokens per node** for granular revocation
2. **Token rotation** on schedule or compromise
3. **Secure token distribution** (not in Docker images)
4. **Monitor** for unauthorized registration attempts

## Network Security

### Firewall Rules
```bash
# Only allow proxy port and SSH
ufw allow 22/tcp
ufw allow 9000/tcp
ufw allow 443/tcp
ufw allow 80/tcp
ufw enable
```

### DDoS Protection
- Use Cloudflare or similar CDN
- Implement rate limiting at proxy level
- Monitor for traffic anomalies

## Monitoring & Alerting

### What to Monitor
- Failed authentication attempts
- Unusual admin API usage patterns
- Node registration/deregistration
- Proxy errors and timeouts
- Memory/CPU usage spikes

### Logging
- Centralize logs (ELK stack, CloudWatch, etc.)
- Retain logs for audit compliance
- Alert on security events

## Incident Response

### If Token Compromised
1. Immediately rotate token in `.env`
2. Restart proxy service
3. Review logs for unauthorized access
4. Check for data exfiltration
5. Update all legitimate clients

### If Node Compromised
1. Remove node from registry
2. Revoke node's API key
3. Audit node's recent activity
4. Check for lateral movement
5. Rebuild node from clean state

## Security Checklist

### Initial Deployment
- [ ] Admin token generated securely
- [ ] Admin endpoints restricted (IP or localhost)
- [ ] HTTPS configured with valid certificate
- [ ] Firewall rules in place
- [ ] Logs configured and monitored

### Ongoing Operations
- [ ] Regular token rotation schedule
- [ ] Security updates applied promptly
- [ ] Access logs reviewed periodically
- [ ] Backup of proxy state
- [ ] Incident response plan tested

## Compliance Considerations

### Data Protection
- Proxy doesn't store user data (stateless)
- Consider GDPR/privacy requirements
- Implement request/response logging policies

### Audit Trail
- Keep immutable logs of admin actions
- Track node registrations and changes
- Document security incidents

## Contact

For security issues, contact: security@blyan.com (or your security team)

**Do not** report security vulnerabilities via public GitHub issues.