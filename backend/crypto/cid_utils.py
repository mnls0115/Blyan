"""
CID (Content Identifier) Utilities

Provides strict CID validation and verification for IPFS and custom CIDs.
"""

import base58
import hashlib
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def decode_ipfs_cid(cid: str) -> Optional[Tuple[str, bytes]]:
    """
    Decode an IPFS CID (v0 or v1) and extract the hash.
    
    Args:
        cid: IPFS CID string (e.g., "QmXyz...")
        
    Returns:
        Tuple of (hash_type, hash_bytes) or None if invalid
    """
    if not cid:
        return None
    
    try:
        # CIDv0 (starts with Qm)
        if cid.startswith("Qm"):
            # Decode base58
            decoded = base58.b58decode(cid)
            
            # CIDv0 format: [0x12][0x20][32-byte SHA256 hash]
            # 0x12 = SHA256, 0x20 = 32 bytes
            if len(decoded) < 34:
                logger.warning(f"CID too short: {len(decoded)} bytes")
                return None
            
            if decoded[0] != 0x12 or decoded[1] != 0x20:
                logger.warning(f"Invalid CIDv0 prefix: {decoded[:2].hex()}")
                return None
            
            # Extract SHA256 hash
            hash_bytes = decoded[2:34]
            return ("sha256", hash_bytes)
        
        # CIDv1 (starts with 'b' for base32)
        elif cid.startswith("b"):
            # CIDv1 is complex: multibase + version + multicodec + multihash
            # For production, we should either:
            # 1. Implement full CIDv1 parsing (complex)
            # 2. Reject CIDv1 and only support CIDv0
            
            # For now, reject CIDv1 in production
            import os
            if os.environ.get('BLOCK_RUNTIME_ALLOW_CIDV1', 'false').lower() != 'true':
                logger.error(f"CIDv1 not supported in production: {cid[:16]}...")
                return None
            
            # Debug mode: simplified validation
            logger.warning("CIDv1 detected in debug mode, using simplified validation")
            return ("cidv1", cid.encode())
        
        else:
            logger.warning(f"Unknown CID format: {cid[:10]}...")
            return None
            
    except Exception as e:
        logger.error(f"Failed to decode CID: {e}")
        return None


def verify_cid(data: bytes, cid: str) -> bool:
    """
    Verify that data matches the given CID.
    
    Args:
        data: Raw data bytes
        cid: CID to verify against
        
    Returns:
        True if data matches CID
    """
    if not cid:
        logger.warning("No CID provided for verification")
        return False
    
    # Decode CID
    decoded = decode_ipfs_cid(cid)
    if not decoded:
        logger.warning(f"Could not decode CID: {cid}")
        return False
    
    hash_type, expected_hash = decoded
    
    # For CIDv1, check if allowed
    if hash_type == "cidv1":
        import os
        if os.environ.get('BLOCK_RUNTIME_ALLOW_CIDV1', 'false').lower() == 'true':
            logger.warning("CIDv1 validation simplified (debug mode)")
            return True
        else:
            logger.error("CIDv1 not supported in production")
            return False
    
    # Compute hash of data
    if hash_type == "sha256":
        computed_hash = hashlib.sha256(data).digest()
        
        # Compare hashes
        if computed_hash == expected_hash:
            logger.debug(f"CID verification successful: {cid[:16]}...")
            return True
        else:
            logger.error(
                f"CID verification failed: "
                f"expected {expected_hash.hex()[:16]}..., "
                f"got {computed_hash.hex()[:16]}..."
            )
            return False
    
    else:
        logger.warning(f"Unsupported hash type: {hash_type}")
        return False


def compute_cid(data: bytes, cid_version: int = 0) -> str:
    """
    Compute CID for given data.
    
    Args:
        data: Raw data bytes
        cid_version: CID version (0 or 1)
        
    Returns:
        CID string
    """
    # Compute SHA256 hash
    hash_bytes = hashlib.sha256(data).digest()
    
    if cid_version == 0:
        # CIDv0: [0x12=SHA256][0x20=32 bytes][hash]
        cid_bytes = bytes([0x12, 0x20]) + hash_bytes
        return base58.b58encode(cid_bytes).decode('ascii')
    
    else:
        # CIDv1 would need more complex encoding
        logger.warning("CIDv1 generation not fully implemented, using v0")
        cid_bytes = bytes([0x12, 0x20]) + hash_bytes
        return base58.b58encode(cid_bytes).decode('ascii')


def is_valid_cid_format(cid: str) -> bool:
    """
    Check if string is a valid CID format (without verifying content).
    
    Args:
        cid: CID string to check
        
    Returns:
        True if format is valid
    """
    if not cid or not isinstance(cid, str):
        return False
    
    # Check CIDv0 format
    if cid.startswith("Qm"):
        try:
            decoded = base58.b58decode(cid)
            return len(decoded) >= 34
        except:
            return False
    
    # Check CIDv1 format (simplified)
    if cid.startswith("b") and len(cid) > 10:
        return True
    
    # Check if it's a raw hex hash (fallback)
    if len(cid) == 64:
        try:
            bytes.fromhex(cid)
            return True
        except:
            return False
    
    return False