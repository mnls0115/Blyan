"""
Shared Authentication Utilities
================================
Centralized user authentication and address extraction logic.
"""

from typing import Optional
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


def extract_user_address(http_request: Request) -> Optional[str]:
    """
    Extract user address from HTTP request.
    
    Priority order:
    1. X-User-Address header
    2. API key info from request state
    3. None if not found
    
    Args:
        http_request: FastAPI Request object
        
    Returns:
        User address string or None
    """
    if not http_request:
        return None
    
    # Try header first
    user_address = http_request.headers.get("X-User-Address")
    if user_address:
        return user_address
    
    # Try API key info
    api_key_info = getattr(http_request.state, "api_key_info", None)
    if api_key_info:
        return f"api_key_{api_key_info.key_id}"
    
    return None


def get_request_fingerprint(http_request: Request) -> str:
    """
    Generate a composite fingerprint for request tracking.
    
    Args:
        http_request: FastAPI Request object
        
    Returns:
        Composite key for tracking
    """
    user_address = extract_user_address(http_request)
    client_host = http_request.client.host if http_request.client else "unknown"
    
    if user_address:
        return f"{user_address}|{client_host}"
    return f"anonymous|{client_host}"


def validate_api_key(api_key: str) -> Optional[dict]:
    """
    Validate API key and return key info if valid.
    
    Args:
        api_key: API key string
        
    Returns:
        Dict with key info or None if invalid
    """
    # This would connect to actual API key validation
    # For now, return mock validation
    if not api_key or len(api_key) < 32:
        return None
    
    return {
        "key_id": api_key[:8],
        "valid": True,
        "rate_limit": 100,
        "quota_remaining": 1000
    }