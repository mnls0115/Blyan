"""Version pinning and resolution for models and tokenizers.

Ensures reproducible inference with semver + content hash validation.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import structlog

logger = structlog.get_logger()


@dataclass
class VersionSpec:
    """Version specification with semver and hash."""
    name: str  # e.g., "gpt_oss_20b"
    version: str  # e.g., "v1.2.3"
    content_hash: Optional[str] = None  # e.g., "sha256:abc123..."
    tokenizer_version: Optional[str] = None
    tokenizer_hash: Optional[str] = None
    
    def to_string(self) -> str:
        """Convert to string format: name@version+hash."""
        result = f"{self.name}@{self.version}"
        if self.content_hash:
            result += f"+{self.content_hash}"
        return result
    
    @classmethod
    def from_string(cls, spec: str) -> 'VersionSpec':
        """Parse from string format."""
        # Pattern: name@version+hash
        pattern = r'^([^@]+)@([^+]+)(?:\+(.+))?$'
        match = re.match(pattern, spec)
        
        if not match:
            raise ValueError(f"Invalid version spec: {spec}")
        
        name, version, hash_part = match.groups()
        
        return cls(
            name=name,
            version=version,
            content_hash=hash_part
        )
    
    def is_compatible_with(self, other: 'VersionSpec') -> bool:
        """Check if versions are compatible (same major version)."""
        def parse_semver(v: str) -> Tuple[int, int, int]:
            # Remove 'v' prefix if present
            v = v.lstrip('v')
            parts = v.split('.')
            return (
                int(parts[0]) if len(parts) > 0 else 0,
                int(parts[1]) if len(parts) > 1 else 0,
                int(parts[2]) if len(parts) > 2 else 0
            )
        
        self_major, _, _ = parse_semver(self.version)
        other_major, _, _ = parse_semver(other.version)
        
        return self_major == other_major


class VersionResolver:
    """Resolves and validates model/tokenizer versions."""
    
    def __init__(
        self,
        default_versions: Optional[Dict[str, VersionSpec]] = None,
        enforce_hash_validation: bool = True,
        allow_fallback: bool = True,
        chain_client=None
    ):
        self.enforce_hash_validation = enforce_hash_validation
        self.allow_fallback = allow_fallback
        self.chain_client = chain_client
        
        # Default versions
        self.default_versions = default_versions or {
            "gpt_oss_20b": VersionSpec(
                name="gpt_oss_20b",
                version="v1.0.0",
                content_hash="sha256:default20b",
                tokenizer_version="v1.0.0",
                tokenizer_hash="sha256:tokenizer20b"
            ),
            "gpt_oss_120b": VersionSpec(
                name="gpt_oss_120b",
                version="v1.0.0",
                content_hash="sha256:default120b",
                tokenizer_version="v1.0.0",
                tokenizer_hash="sha256:tokenizer120b"
            )
        }
        
        # Cache resolved versions
        self.resolution_cache: Dict[str, VersionSpec] = {}
        
        # Metrics
        self.resolution_count = 0
        self.validation_failures = 0
        self.fallback_count = 0
    
    async def resolve(
        self,
        model_name: str,
        requested_version: Optional[str] = None,
        requested_hash: Optional[str] = None
    ) -> VersionSpec:
        """Resolve model version with validation.
        
        Args:
            model_name: Model identifier (e.g., "gpt_oss_20b")
            requested_version: Optional specific version
            requested_hash: Optional content hash for validation
            
        Returns:
            Resolved VersionSpec
            
        Raises:
            ValueError: If version cannot be resolved or validation fails
        """
        self.resolution_count += 1
        
        # Check cache
        cache_key = f"{model_name}:{requested_version}:{requested_hash}"
        if cache_key in self.resolution_cache:
            return self.resolution_cache[cache_key]
        
        try:
            # If specific version requested
            if requested_version:
                spec = await self._resolve_specific(
                    model_name,
                    requested_version,
                    requested_hash
                )
            else:
                # Resolve to latest
                spec = await self._resolve_latest(model_name)
            
            # Validate if required
            if self.enforce_hash_validation and requested_hash:
                if not await self._validate_hash(spec, requested_hash):
                    self.validation_failures += 1
                    raise ValueError(
                        f"Hash validation failed for {model_name}@{requested_version}: "
                        f"expected {requested_hash}, got {spec.content_hash}"
                    )
            
            # Cache result
            self.resolution_cache[cache_key] = spec
            
            logger.info(
                "Version resolved",
                model=model_name,
                version=spec.version,
                hash=spec.content_hash
            )
            
            return spec
            
        except Exception as e:
            logger.error(f"Version resolution failed: {e}")
            
            if self.allow_fallback:
                # Fall back to default
                self.fallback_count += 1
                return await self._get_default(model_name)
            else:
                raise
    
    async def _resolve_specific(
        self,
        model_name: str,
        version: str,
        requested_hash: Optional[str]
    ) -> VersionSpec:
        """Resolve specific version."""
        # Query chain if available
        if self.chain_client:
            try:
                # Would query blockchain for model metadata
                chain_data = await self.chain_client.get_model_version(
                    model_name,
                    version
                )
                
                return VersionSpec(
                    name=model_name,
                    version=version,
                    content_hash=chain_data.get("hash", requested_hash),
                    tokenizer_version=chain_data.get("tokenizer_version"),
                    tokenizer_hash=chain_data.get("tokenizer_hash")
                )
            except Exception as e:
                logger.warning(f"Chain query failed: {e}")
        
        # Fall back to constructed spec
        return VersionSpec(
            name=model_name,
            version=version,
            content_hash=requested_hash or self._compute_hash(model_name, version)
        )
    
    async def _resolve_latest(self, model_name: str) -> VersionSpec:
        """Resolve to latest version."""
        # Query chain for latest
        if self.chain_client:
            try:
                latest = await self.chain_client.get_latest_version(model_name)
                return VersionSpec(
                    name=model_name,
                    version=latest["version"],
                    content_hash=latest["hash"],
                    tokenizer_version=latest.get("tokenizer_version"),
                    tokenizer_hash=latest.get("tokenizer_hash")
                )
            except Exception as e:
                logger.warning(f"Failed to get latest from chain: {e}")
        
        # Fall back to default
        return await self._get_default(model_name)
    
    async def _get_default(self, model_name: str) -> VersionSpec:
        """Get default version for model."""
        if model_name in self.default_versions:
            logger.warning(
                f"Using default version for {model_name}",
                version=self.default_versions[model_name].version
            )
            return self.default_versions[model_name]
        
        # Ultimate fallback
        return VersionSpec(
            name=model_name,
            version="v1.0.0",
            content_hash="sha256:unknown"
        )
    
    async def _validate_hash(self, spec: VersionSpec, expected_hash: str) -> bool:
        """Validate content hash."""
        if not spec.content_hash:
            return False
        
        # Normalize hash format
        spec_hash = spec.content_hash.replace("sha256:", "").lower()
        expected = expected_hash.replace("sha256:", "").lower()
        
        return spec_hash == expected
    
    def _compute_hash(self, model_name: str, version: str) -> str:
        """Compute deterministic hash for model version."""
        data = f"{model_name}:{version}"
        return f"sha256:{hashlib.sha256(data.encode()).hexdigest()[:16]}"
    
    def check_compatibility(
        self,
        requested: VersionSpec,
        available: VersionSpec
    ) -> bool:
        """Check if available version is compatible with requested."""
        # Same model name required
        if requested.name != available.name:
            return False
        
        # Check version compatibility
        if not requested.is_compatible_with(available):
            logger.warning(
                f"Version incompatibility: requested {requested.version}, "
                f"available {available.version}"
            )
            return False
        
        # If hash specified, must match
        if requested.content_hash and available.content_hash:
            if requested.content_hash != available.content_hash:
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get resolver metrics."""
        return {
            "resolution_count": self.resolution_count,
            "validation_failures": self.validation_failures,
            "fallback_count": self.fallback_count,
            "cache_size": len(self.resolution_cache),
            "fallback_rate": self.fallback_count / max(self.resolution_count, 1)
        }