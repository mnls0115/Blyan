# Version Pinning Policy

## Overview

The Blyan Network uses semantic versioning (semver) combined with content hashing to ensure reproducible inference across the distributed network.

## Version Format

Models and tokenizers use the following format:

```
name@version+hash
```

Examples:
- `gpt_oss_20b@v1.2.3+sha256:abc123def456`
- `gpt_oss_120b@v2.0.0+sha256:789xyz`

## Semantic Versioning

We follow semver conventions:

- **Major version (X.0.0)**: Breaking changes, incompatible architecture
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, performance improvements

### Compatibility Rules

- Same major version = Compatible (can use as fallback)
- Different major version = Incompatible (reject unless forced)

## Content Hashing

All model weights and tokenizers include SHA256 hashes for integrity:

- Model hash: SHA256 of concatenated weight tensors
- Tokenizer hash: SHA256 of vocabulary and config

## Resolution Policy

### 1. Explicit Version Request

When a client requests a specific version:

```python
# Request
{
  "model": "gpt_oss_20b",
  "model_version": "v1.2.3",
  "model_hash": "sha256:abc123"
}
```

Resolution:
1. Query blockchain for exact version
2. Validate hash if provided
3. Reject if validation fails (unless `allow_fallback=true`)

### 2. Latest Version Request

When no version specified:

```python
# Request
{
  "model": "gpt_oss_20b"
}
```

Resolution:
1. Query blockchain for latest stable version
2. Use meta-chain (Chain A) for authoritative version
3. Fall back to configured default if chain unavailable

### 3. Fallback Behavior

When requested version unavailable:

1. **Compatible fallback**: Use same major version if available
2. **Warning mode**: Log warning but proceed with compatible version
3. **Strict mode**: Reject request (return 400 Bad Request)

## Migration Hooks

When models are upgraded:

### Pre-migration
```python
async def pre_migration(old_version: VersionSpec, new_version: VersionSpec):
    # Validate compatibility
    # Backup current state
    # Notify active nodes
```

### Post-migration
```python
async def post_migration(old_version: VersionSpec, new_version: VersionSpec):
    # Update chain metadata
    # Invalidate caches
    # Trigger re-validation
```

## Configuration

In `config/server.yaml`:

```yaml
versioning:
  default_model: gpt_oss_20b
  default_version: v1.0.0
  enforce_hash_validation: true  # Require hash match
  allow_version_fallback: true   # Allow compatible versions
  cache_ttl_seconds: 300         # Version cache TTL
```

## API Usage

### Request with Version Pinning

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "model": "gpt_oss_20b@v1.2.3+sha256:abc123",
    "max_tokens": 100
  }'
```

### Response with Version Info

```json
{
  "text": "Hello! How can I help you?",
  "receipt": {
    "model_version": "v1.2.3",
    "model_hash": "sha256:abc123",
    "tokenizer_version": "v1.0.0",
    "tokenizer_hash": "sha256:def456"
  }
}
```

## Blockchain Integration

Model versions are stored on-chain:

1. **Meta-chain (A)**: Model architecture and version registry
2. **Parameter-chain (B)**: Actual model weights with version tags
3. **Version blocks**: Special blocks linking versions to weight blocks

## Best Practices

1. **Always specify version in production**: Avoid surprises from auto-updates
2. **Include hash for critical inference**: Ensure exact reproducibility
3. **Test before major upgrades**: Validate compatibility with test traffic
4. **Monitor version metrics**: Track which versions are most used
5. **Gradual rollout**: Use percentage-based routing for new versions

## Metrics

The system exports version-related metrics:

- `blyan_server_version_mismatch_total`: Requests with version conflicts
- `blyan_server_version_defaulted_total`: Requests using default version
- `blyan_server_version_resolution_latency_ms`: Time to resolve version

## Security Considerations

1. **Hash validation prevents tampering**: Modified models will fail hash check
2. **Blockchain provides audit trail**: All version changes recorded on-chain
3. **Signed version manifests**: Cryptographic proof of version authenticity
4. **Rollback capability**: Can revert to previous version if issues detected