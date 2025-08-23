"""
Configuration and Feature Flags for Block Runtime

Controls runtime behavior and enables gradual rollout of features.
"""

import os
from typing import Optional
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlags:
    """Feature flags for block runtime."""
    
    # Core runtime
    block_runtime_enabled: bool = False
    block_runtime_canary_percent: float = 0.0
    
    # Fetch strategies
    block_runtime_hedged_fetch: bool = False
    block_runtime_prefetch: bool = False
    block_runtime_prefetch_layers: int = 2
    
    # Caching
    block_runtime_memory_cache_mb: int = 4096
    block_runtime_disk_cache_mb: int = 20480
    block_runtime_cache_ttl_seconds: int = 3600
    
    # Verification
    block_runtime_enable_verification: bool = True
    block_runtime_strict_verification: bool = False
    
    # Performance
    block_runtime_max_concurrent_fetches: int = 10
    block_runtime_fetch_timeout_ms: int = 5000
    block_runtime_hedged_delay_ms: int = 100
    
    # Streaming
    block_runtime_stream_batch_size: int = 5
    block_runtime_stream_flush_interval_ms: int = 50
    block_runtime_stream_max_queue_size: int = 100
    
    # Metrics
    block_runtime_enable_metrics: bool = True
    block_runtime_detailed_metrics: bool = True
    block_runtime_metrics_export_interval: int = 60
    
    @classmethod
    def from_env(cls) -> "FeatureFlags":
        """Load feature flags from environment variables."""
        flags = cls()
        
        # Map of attribute name to env var name
        env_mappings = {
            "block_runtime_enabled": "BLOCK_RUNTIME_ENABLED",
            "block_runtime_canary_percent": "BLOCK_RUNTIME_CANARY_PERCENT",
            "block_runtime_hedged_fetch": "BLOCK_RUNTIME_HEDGED_FETCH",
            "block_runtime_prefetch": "BLOCK_RUNTIME_PREFETCH",
            "block_runtime_prefetch_layers": "BLOCK_RUNTIME_PREFETCH_LAYERS",
            "block_runtime_memory_cache_mb": "BLOCK_RUNTIME_MEMORY_CACHE_MB",
            "block_runtime_disk_cache_mb": "BLOCK_RUNTIME_DISK_CACHE_MB",
            "block_runtime_cache_ttl_seconds": "BLOCK_RUNTIME_CACHE_TTL_SECONDS",
            "block_runtime_enable_verification": "BLOCK_RUNTIME_ENABLE_VERIFICATION",
            "block_runtime_strict_verification": "BLOCK_RUNTIME_STRICT_VERIFICATION",
            "block_runtime_max_concurrent_fetches": "BLOCK_RUNTIME_MAX_CONCURRENT_FETCHES",
            "block_runtime_fetch_timeout_ms": "BLOCK_RUNTIME_FETCH_TIMEOUT_MS",
            "block_runtime_hedged_delay_ms": "BLOCK_RUNTIME_HEDGED_DELAY_MS",
            "block_runtime_stream_batch_size": "BLOCK_RUNTIME_STREAM_BATCH_SIZE",
            "block_runtime_stream_flush_interval_ms": "BLOCK_RUNTIME_STREAM_FLUSH_INTERVAL_MS",
            "block_runtime_stream_max_queue_size": "BLOCK_RUNTIME_STREAM_MAX_QUEUE_SIZE",
            "block_runtime_enable_metrics": "BLOCK_RUNTIME_ENABLE_METRICS",
            "block_runtime_detailed_metrics": "BLOCK_RUNTIME_DETAILED_METRICS",
            "block_runtime_metrics_export_interval": "BLOCK_RUNTIME_METRICS_EXPORT_INTERVAL"
        }
        
        for attr_name, env_name in env_mappings.items():
            if env_name in os.environ:
                value = os.environ[env_name]
                attr_type = type(getattr(flags, attr_name))
                
                try:
                    if attr_type == bool:
                        setattr(flags, attr_name, value.lower() in ("true", "1", "yes"))
                    elif attr_type == int:
                        setattr(flags, attr_name, int(value))
                    elif attr_type == float:
                        setattr(flags, attr_name, float(value))
                    else:
                        setattr(flags, attr_name, value)
                except ValueError as e:
                    logger.warning(f"Invalid value for {env_name}: {value}, using default")
        
        return flags
    
    @classmethod
    def from_file(cls, path: Path) -> "FeatureFlags":
        """Load feature flags from JSON file."""
        if not path.exists():
            logger.warning(f"Feature flags file not found: {path}")
            return cls()
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            flags = cls()
            for key, value in data.items():
                if hasattr(flags, key):
                    setattr(flags, key, value)
                else:
                    logger.warning(f"Unknown feature flag: {key}")
            
            return flags
            
        except Exception as e:
            logger.error(f"Error loading feature flags from {path}: {e}")
            return cls()
    
    def save_to_file(self, path: Path) -> None:
        """Save feature flags to JSON file."""
        data = {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def should_use_block_runtime(self, session_id: Optional[str] = None) -> bool:
        """
        Determine if block runtime should be used for a given session.
        
        Supports canary rollout based on session ID hashing.
        """
        if not self.block_runtime_enabled:
            return False
        
        if self.block_runtime_canary_percent >= 1.0:
            return True
        
        if self.block_runtime_canary_percent <= 0.0:
            return False
        
        if session_id is None:
            # No session ID, use global flag
            return self.block_runtime_enabled
        
        # Hash session ID to determine if in canary
        import hashlib
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        threshold = int(self.block_runtime_canary_percent * (2 ** 128))
        
        return hash_value < threshold


# Global feature flags instance
_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Get the global feature flags instance."""
    global _feature_flags
    
    if _feature_flags is None:
        # Try loading from file first
        config_file = Path("./config/block_runtime_flags.json")
        if config_file.exists():
            _feature_flags = FeatureFlags.from_file(config_file)
            logger.info(f"Loaded feature flags from {config_file}")
        else:
            # Fall back to environment variables
            _feature_flags = FeatureFlags.from_env()
            logger.info("Loaded feature flags from environment")
    
    return _feature_flags


def reload_feature_flags() -> None:
    """Reload feature flags from source."""
    global _feature_flags
    _feature_flags = None
    get_feature_flags()


def set_feature_flag(name: str, value: any) -> None:
    """Set a specific feature flag value."""
    flags = get_feature_flags()
    if hasattr(flags, name):
        setattr(flags, name, value)
        logger.info(f"Set feature flag {name} = {value}")
    else:
        raise ValueError(f"Unknown feature flag: {name}")