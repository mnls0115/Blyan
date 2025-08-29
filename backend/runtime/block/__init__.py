"""
Block Runtime Package

Unified inference runtime for standardized expert management.
"""

from .runtime import BlockRuntime
from .types import (
    RequestSpec,
    RuntimeConfig,
    CacheConfig,
    FetchStrategy,
    ExpertMetadata,
    ExpertData,
    StreamToken
)
from .expert_store import LayerStore as ExpertStore
from .execution_engine import ExecutionEngine
from .streamer import Streamer, StreamSession
from .metrics import MetricsCollector
from .errors import (
    BlockRuntimeError,
    ExpertNotFoundError,
    ExpertVerificationError,
    FetchTimeoutError,
    CacheError,
    StreamingError,
    SessionNotFoundError,
    ResourceExhaustedError,
    InvalidRequestError
)

__all__ = [
    # Main runtime
    "BlockRuntime",
    
    # Types
    "RequestSpec",
    "RuntimeConfig", 
    "CacheConfig",
    "FetchStrategy",
    "ExpertMetadata",
    "ExpertData",
    "StreamToken",
    
    # Components
    "ExpertStore",
    "ExecutionEngine",
    "Streamer",
    "StreamSession",
    "MetricsCollector",
    
    # Errors
    "BlockRuntimeError",
    "ExpertNotFoundError",
    "ExpertVerificationError",
    "FetchTimeoutError",
    "CacheError",
    "StreamingError",
    "SessionNotFoundError",
    "ResourceExhaustedError",
    "InvalidRequestError"
]

# Version
__version__ = "0.1.0"