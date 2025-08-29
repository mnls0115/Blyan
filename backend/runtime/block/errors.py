"""
Block Runtime Error Definitions

Custom exceptions for the block runtime layer.
"""

from typing import Any


class BlockRuntimeError(Exception):
    """Base exception for block runtime errors."""
    pass


class ExpertNotFoundError(BlockRuntimeError):
    """Raised when an expert cannot be found."""
    def __init__(self, layer_id: int, expert_id: int):
        self.layer_id = layer_id
        self.expert_id = expert_id
        super().__init__(f"Expert not found: layer={layer_id}, expert={expert_id}")


class ExpertVerificationError(BlockRuntimeError):
    """Raised when expert verification fails."""
    def __init__(self, cid: str, reason: str):
        self.cid = cid
        self.reason = reason
        super().__init__(f"Expert verification failed for CID {cid}: {reason}")


# Dense-layer aligned exceptions (aliases for clarity in dense runtime)
class LayerNotFoundError(BlockRuntimeError):
    """Raised when a dense layer cannot be found."""
    def __init__(self, layer_name: str | int):
        self.layer_name = layer_name
        super().__init__(f"Layer not found: {layer_name}")


class LayerVerificationError(BlockRuntimeError):
    """Raised when dense layer verification fails."""
    def __init__(self, layer_name: str | int, reason: str):
        self.layer_name = layer_name
        self.reason = reason
        super().__init__(f"Layer verification failed for {layer_name}: {reason}")


class FetchTimeoutError(BlockRuntimeError):
    """Raised when expert fetch times out."""
    def __init__(self, expert_id: str, timeout_ms: int):
        self.expert_id = expert_id
        self.timeout_ms = timeout_ms
        super().__init__(f"Fetch timeout for expert {expert_id} after {timeout_ms}ms")


class CacheError(BlockRuntimeError):
    """Raised when cache operations fail."""
    pass


class StreamingError(BlockRuntimeError):
    """Raised when streaming fails."""
    pass


class SessionNotFoundError(BlockRuntimeError):
    """Raised when a session cannot be found."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class ResourceExhaustedError(BlockRuntimeError):
    """Raised when resources are exhausted."""
    def __init__(self, resource: str, limit: Any):
        self.resource = resource
        self.limit = limit
        super().__init__(f"Resource exhausted: {resource} (limit: {limit})")


class InvalidRequestError(BlockRuntimeError):
    """Raised when request specification is invalid."""
    pass
