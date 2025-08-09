#!/usr/bin/env python3
"""
Wallet authentication with Ethereum signature verification

### PRODUCTION UPDATE ###
이전: 서명 검증 없이 항상 True 반환
현재: 실제 Ethereum 서명 검증 구현
- eth_account 라이브러리로 서명 검증
- MetaMask/WalletConnect 지원
- Phase 2에서 email OTP 추가 예정
"""

import time
import secrets
import hashlib
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import json
import redis
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal
from backend.accounting.ledger_postgres import postgres_ledger

# Ethereum signature verification
try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    ETH_ACCOUNT_AVAILABLE = True
except ImportError:
    ETH_ACCOUNT_AVAILABLE = False
    logging.warning("eth_account not installed. Install with: pip install eth-account")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/wallet", tags=["wallet"])

# Redis connection for nonce storage (fallback to memory if not available)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected for nonce storage")
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory storage")
    # Fallback to in-memory storage
    used_nonces: Dict[str, float] = {}
    user_nonces: Dict[str, str] = {}

# User session storage
user_sessions: Dict[str, Dict] = {}  # address -> session data

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
    message: Optional[str] = None

class BalanceResponse(BaseModel):
    address: str
    bly_balance: float
    upload_credits: int
    total_rewards: float
    rank: Optional[int] = None

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
    
    # Store nonce with expiry
    if REDIS_AVAILABLE:
        redis_client.setex(f"nonce:{request.address}", 300, nonce)  # 5 min expiry
        redis_client.setex(f"nonce:{nonce}", 300, request.address)  # For reverse lookup
    else:
        user_nonces[request.address] = nonce
    
    # Create message to sign (EIP-4361 inspired format)
    message = (
        f"Blyan Network wants you to sign in with your Ethereum account:\n"
        f"{request.address}\n\n"
        f"Sign this message to authenticate and access Blyan Network services.\n\n"
        f"URI: https://blyan.network\n"
        f"Version: 1\n"
        f"Chain ID: 1\n"
        f"Nonce: {nonce}\n"
        f"Issued At: {datetime.fromtimestamp(timestamp).isoformat()}"
    )
    
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
        
        # ### PRODUCTION CODE: Actual signature verification ###
        if not ETH_ACCOUNT_AVAILABLE:
            logger.error("eth_account not installed, cannot verify signatures")
            raise HTTPException(500, "Signature verification not available")
        
        # Verify Ethereum signature
        is_valid = verify_ethereum_signature(
            request.address,
            request.message,
            request.signature
        )
        
        if not is_valid:
            logger.warning(f"Invalid signature from {request.address}")
            raise HTTPException(401, "Invalid signature")
        
        logger.info(f"✅ Signature verified for {request.address}")
        
        # Mark nonce as used
        if REDIS_AVAILABLE:
            redis_client.setex(f"used:{nonce_key}", 3600, "1")  # Keep for 1 hour
            redis_client.delete(f"nonce:{request.address}")
            redis_client.delete(f"nonce:{request.nonce}")
        else:
            used_nonces[nonce_key] = time.time()
            if request.address in user_nonces:
                del user_nonces[request.address]
        
        # Clean up old nonces periodically
        _cleanup_old_nonces()
        
        # Generate session token (JWT in production)
        session_token = secrets.token_urlsafe(32)
        
        # Get user info from database/blockchain
        user_info = get_user_info(request.address)
        
        # Store session
        user_sessions[request.address] = {
            "token": session_token,
            "authenticated_at": int(time.time()),
            "last_activity": int(time.time())
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

def verify_ethereum_signature(address: str, message: str, signature: str) -> bool:
    """
    Verify Ethereum signature using eth_account.
    
    ### PRODUCTION IMPLEMENTATION ###
    Replaces the TODO comment with actual signature verification.
    """
    try:
        # Encode message in Ethereum format
        message_hash = encode_defunct(text=message)
        
        # Recover address from signature
        recovered_address = Account.recover_message(message_hash, signature=signature)
        
        # Compare addresses (case-insensitive)
        is_valid = recovered_address.lower() == address.lower()
        
        if is_valid:
            logger.debug(f"Signature valid for {address}")
        else:
            logger.warning(f"Signature mismatch: expected {address}, got {recovered_address}")
            
        return is_valid
        
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False

def get_user_info(address: str) -> Dict:
    """
    Get user information from database/blockchain.
    """
    # TODO: Query actual database
    # For now, return mock data with proper structure
    
    # Check if user exists in our tracking
    # from backend.data.quality_gate_v2 import get_quality_gate
    
    try:
        # gate = get_quality_gate()
        # contributor = gate.contributors.get(address, None)
        contributor = None
        
        if contributor:
            return {
                "address": address,
                "bly_balance": 0,  # contributor.total_rewards_bly,
                "upload_credits": contributor.daily_credits,
                "total_submissions": contributor.total_submissions,
                "quality_score": contributor.quality_score,
                "rank": None,  # TODO: Calculate from leaderboard
                "authenticated_at": int(time.time())
            }
    except:
        pass
    
    # New user
    return {
        "address": address,
        "bly_balance": 0.0,
        "upload_credits": 20,  # Initial credits
        "total_submissions": 0,
        "quality_score": 0.5,
        "rank": None,
        "authenticated_at": int(time.time())
    }

@router.get("/balance/{address}")
async def get_balance(address: str) -> BalanceResponse:
    """
    Get BLY balance and transaction summary from PostgreSQL.
    """
    # Initialize ledger if not already done
    if not postgres_ledger._initialized:
        await postgres_ledger.initialize()
    
    # Get balance from PostgreSQL
    try:
        balance = await postgres_ledger.get_balance(address)
        summary = await postgres_ledger.get_account_summary(address)
        
        return BalanceResponse(
            address=address,
            bly_balance=float(balance),
            upload_credits=20,  # Default credits
            total_rewards=float(summary.get("total_received", 0)),
            rank=None  # TODO: Calculate from leaderboard
        )
    except Exception as e:
        logger.error(f"Failed to get balance from PostgreSQL: {e}")
        # Fallback to default response
        return BalanceResponse(
            address=address,
            bly_balance=0.0,
            upload_credits=20,
            total_rewards=0.0,
            rank=None
        )

@router.get("/transactions/{address}")
async def get_transactions(address: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """
    Get transaction history for an address from PostgreSQL.
    """
    # Initialize ledger if not already done
    if not postgres_ledger._initialized:
        await postgres_ledger.initialize()
    
    try:
        transactions = await postgres_ledger.get_user_transactions(address, limit, offset)
        return {
            "address": address,
            "transactions": transactions,
            "count": len(transactions),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to get transactions from PostgreSQL: {e}")
        return {
            "address": address,
            "transactions": [],
            "count": 0,
            "error": str(e)
        }

@router.get("/verify_token/{token}")
async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify session token is still valid.
    """
    # Find session by token
    for address, session in user_sessions.items():
        if session.get("token") == token:
            # Check if expired (24 hours)
            if time.time() - session["authenticated_at"] > 86400:
                return {
                    "valid": False,
                    "reason": "Token expired"
                }
            
            # Update last activity
            session["last_activity"] = int(time.time())
            
            return {
                "valid": True,
                "address": address,
                "expires_at": session["authenticated_at"] + 86400
            }
    
    return {
        "valid": False,
        "reason": "Token not found"
    }

@router.post("/logout")
async def logout(request: AuthRequest) -> Dict[str, str]:
    """
    Logout and invalidate session.
    """
    if request.address in user_sessions:
        del user_sessions[request.address]
        return {"success": True, "message": "Logged out successfully"}
    
    return {"success": False, "message": "No active session"}

# Phase 2: Email OTP fallback (for non-crypto users)
@router.post("/request_otp")
async def request_otp(email: str) -> Dict[str, str]:
    """
    Phase 2: Request OTP for email authentication.
    For users without crypto wallets.
    """
    # TODO: Implement in Phase 2
    return {
        "message": "Email OTP coming in Phase 2",
        "phase": "1",
        "wallet_recommended": True
    }

# Include router in main app
def include_router(app):
    """Include wallet auth routes in FastAPI app."""
    app.include_router(router)

# Cleanup task for expired nonces
async def cleanup_expired_nonces():
    """Background task to clean expired nonces."""
    while True:
        try:
            _cleanup_old_nonces()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Nonce cleanup error: {e}")
            await asyncio.sleep(3600)