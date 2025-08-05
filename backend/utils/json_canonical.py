"""
Canonical JSON serialization for blockchain consensus data.
CRITICAL: This module ensures deterministic JSON output for hash consistency.
DO NOT modify the serialization format without understanding consensus implications.
"""
import json
from typing import Any, Dict, List, Union


def dumps_canonical(obj: Union[Dict, List, Any]) -> str:
    """
    Serialize object to canonical JSON format for consensus operations.
    
    CRITICAL: This function MUST be used for:
    - Block header/payload serialization before hashing
    - Merkle tree node serialization
    - Any data that affects consensus/hash calculations
    
    DO NOT use orjson or other JSON libraries for consensus data!
    
    Args:
        obj: Object to serialize (must be JSON-serializable)
        
    Returns:
        Canonical JSON string with deterministic ordering
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def loads_canonical(json_str: str) -> Union[Dict, List, Any]:
    """
    Parse JSON string (compatible with canonical format).
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed object
    """
    return json.loads(json_str)


# Performance-oriented JSON operations (non-consensus only)
try:
    import orjson
    
    def dumps_fast(obj: Any) -> bytes:
        """
        Fast JSON serialization for non-consensus data.
        
        USE ONLY FOR:
        - API responses
        - Cache serialization
        - Logging/metrics
        - UI data
        
        NEVER USE FOR:
        - Block data
        - Hash calculations
        - Merkle trees
        - Consensus operations
        """
        return orjson.dumps(obj)
    
    def loads_fast(data: Union[str, bytes]) -> Any:
        """Fast JSON parsing for non-consensus data."""
        return orjson.loads(data)
    
    HAS_ORJSON = True
    
except ImportError:
    # Fallback to standard json if orjson not installed
    def dumps_fast(obj: Any) -> bytes:
        return json.dumps(obj).encode('utf-8')
    
    def loads_fast(data: Union[str, bytes]) -> Any:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    
    HAS_ORJSON = False