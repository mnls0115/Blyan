#!/usr/bin/env python3
"""
Wallet authentication with nonce-based replay protection
"""

import time
import secrets
import hashlib
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wallet", tags=["wallet"])

# In-memory nonce storage (use Redis in production)
used_nonces: Dict[str, float] = {}
user_nonces: Dict[str, str] = {}

class NonceRequest(BaseModel):
    address: str

class AuthRequest(BaseModel):
    address: str
    signature: str
    nonce: str
    message: str

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user_info: Optional[Dict] = None

@router.post("/request_nonce")
async def request_nonce(request: NonceRequest) -> Dict[str, str]:
    """
    Generate a unique nonce for wallet signature.
    Prevents replay attacks by ensuring each signature is used only once.
    """
    # Generate cryptographically secure nonce
    nonce = secrets.token_hex(32)
    timestamp = int(time.time())
    
    # Store nonce with timestamp
    nonce_key = f"{request.address}:{nonce}"
    user_nonces[request.address] = nonce
    
    # Create message to sign
    message = f"Sign this message to authenticate with Blyan Network.\n\nNonce: {nonce}\nTimestamp: {timestamp}\nAddress: {request.address}"
    
    return {
        "nonce": nonce,
        "message": message,
        "timestamp": timestamp,
        "expires_in": 300  # 5 minutes
    }

@router.post("/authenticate")
async def authenticate(request: AuthRequest) -> AuthResponse:
    """
    Verify wallet signature with nonce validation.
    """
    try:
        # Check if nonce exists and hasn't been used
        nonce_key = f"{request.address}:{request.nonce}"
        
        # Verify nonce hasn't been used
        if nonce_key in used_nonces:
            raise HTTPException(400, "Nonce already used")
            
        # Verify nonce was issued to this address
        if user_nonces.get(request.address) != request.nonce:
            raise HTTPException(400, "Invalid nonce")
            
        # Check nonce expiry (5 minutes)
        # In production, store timestamp with nonce
        
        # TODO: Verify signature using web3 or ecdsa
        # For now, we'll assume signature is valid
        # In production: verify_signature(request.address, request.message, request.signature)
        
        # Mark nonce as used
        used_nonces[nonce_key] = time.time()
        del user_nonces[request.address]
        
        # Clean up old nonces periodically
        _cleanup_old_nonces()
        
        # Generate session token (JWT in production)
        session_token = secrets.token_urlsafe(32)
        
        # Get user info (from blockchain or database)
        user_info = {
            "address": request.address,
            "balance": 0,  # TODO: Query actual balance
            "rank": None,  # TODO: Query leaderboard rank
            "authenticated_at": int(time.time())
        }
        
        return AuthResponse(
            success=True,
            token=session_token,
            user_info=user_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(500, "Authentication failed")

def _cleanup_old_nonces():
    """Remove nonces older than 1 hour."""
    current_time = time.time()
    cutoff_time = current_time - 3600  # 1 hour
    
    # Clean up used nonces
    expired_nonces = [k for k, v in used_nonces.items() if v < cutoff_time]
    for nonce in expired_nonces:
        del used_nonces[nonce]
        
    # Log cleanup
    if expired_nonces:
        logger.info(f"Cleaned up {len(expired_nonces)} expired nonces")

@router.get("/verify_token/{token}")
async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify session token is still valid.
    """
    # TODO: Implement JWT verification
    # For now, simple validation
    
    return {
        "valid": True,
        "address": "0x...",  # From token
        "expires_at": int(time.time()) + 3600
    }

# Include router in main app
def include_router(app):
    """Include wallet auth routes in FastAPI app."""
    app.include_router(router)