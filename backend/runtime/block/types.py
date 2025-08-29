"""
Block Runtime Type Definitions

Core types for the standardized inference runtime layer.
"""

from typing import TypedDict, Optional, Any, Protocol, Callable, Awaitable
from dataclasses import dataclass
import torch
from enum import Enum


class RequestSpec(TypedDict):
    """Specification for an inference request."""
    model_id: str
    input_ids: torch.Tensor
    layer_plan: dict[int, list[int]]  # layer -> expert IDs
    sampling: dict[str, Any]
    session_id: str
    max_tokens: Optional[int]
    stream: bool


@dataclass
class ExpertMetadata:
    """Metadata for a single expert."""
    layer_id: int
    expert_id: int
    cid: str
    shard_id: Optional[str]
    offset: int
    length: int
    merkle_root: str
    merkle_proof: Optional[list[tuple[str, bool]]] = None  # List of (hash, is_left) for proof
    compression: Optional[str] = None
    precision: str = "fp16"  # fp16, bf16, int8, fp8, etc.


# Backward/forward-compatibility alias for dense layer metadata
# Many runtimes treat a single dense layer similarly to a legacy "expert" unit.
# Expose LayerMetadata as an alias to ExpertMetadata so imports remain stable.
LayerMetadata = ExpertMetadata


@dataclass
class ExpertData:
    """Loaded expert data with metadata."""
    metadata: ExpertMetadata
    weights: torch.Tensor
    verified: bool
    cache_hit: bool
    fetch_latency_ms: float


@dataclass
class LayerData:
    """Loaded layer data with metadata (dense-layer friendly type)."""
    metadata: LayerMetadata
    weights: Any  # torch.Tensor in practice
    verified: bool
    cache_hit: bool
    fetch_latency_ms: float


class FetchStrategy(Enum):
    """Expert fetch strategies."""
    STANDARD = "standard"
    HEDGED = "hedged"
    PREFETCH = "prefetch"


@dataclass
class StreamToken:
    """Token streaming data."""
    token_id: int
    logprobs: Optional[dict[int, float]]
    timestamp_ms: int
    expert_usage: dict[str, float]  # expert_id -> usage weight


class BlockRuntime(Protocol):
    """Protocol for the block runtime interface."""
    
    async def run_inference(
        self,
        req: RequestSpec,
        stream_cb: Callable[[StreamToken], Awaitable[None]]
    ) -> None:
        """Run inference with streaming callback."""
        ...
    
    async def cancel(self, session_id: str) -> None:
        """Cancel an ongoing inference session."""
        ...
    
    def get_metrics(self) -> dict[str, Any]:
        """Get runtime metrics."""
        ...


@dataclass
class CacheConfig:
    """Configuration for expert caching."""
    memory_cache_size_mb: int = 4096
    disk_cache_size_mb: int = 20480
    ttl_seconds: int = 3600
    eviction_policy: str = "lru"
    admit_threshold: float = 0.7  # Hit-cost product threshold


@dataclass
class RuntimeConfig:
    """Configuration for the block runtime."""
    cache_config: CacheConfig
    fetch_strategy: FetchStrategy = FetchStrategy.STANDARD
    max_concurrent_fetches: int = 10
    fetch_timeout_ms: int = 5000
    hedged_delay_ms: int = 100
    prefetch_early_layers: int = 2
    enable_verification: bool = True
    enable_metrics: bool = True
