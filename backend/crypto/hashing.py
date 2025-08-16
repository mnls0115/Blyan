"""Hashing utilities for blocks and transactions"""
import hashlib
import json
from typing import List, Any


def hash_block(block: dict) -> str:
    """
    Calculate block hash
    
    Args:
        block: Block dict
        
    Returns:
        Hex-encoded hash
    """
    # Hash includes all fields except the hash itself
    block_copy = block.copy()
    block_copy.pop('hash', None)
    
    # Canonical JSON encoding
    block_bytes = json.dumps(block_copy, sort_keys=True).encode()
    return hashlib.sha256(block_bytes).hexdigest()


def hash_transaction(tx: dict) -> str:
    """
    Calculate transaction hash
    
    Args:
        tx: Transaction dict
        
    Returns:
        Hex-encoded hash
    """
    # Canonical JSON encoding
    tx_bytes = json.dumps(tx, sort_keys=True).encode()
    return hashlib.sha256(tx_bytes).hexdigest()


def merkle_root(items: List[str]) -> str:
    """
    Calculate Merkle root of items
    
    Args:
        items: List of hex-encoded hashes
        
    Returns:
        Hex-encoded Merkle root
    """
    if not items:
        return hashlib.sha256(b'').hexdigest()
    
    if len(items) == 1:
        return items[0]
    
    # Convert to bytes
    hashes = [bytes.fromhex(h) for h in items]
    
    while len(hashes) > 1:
        next_level = []
        
        # Pair up hashes
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                # Hash pair
                combined = hashes[i] + hashes[i + 1]
            else:
                # Odd number - duplicate last
                combined = hashes[i] + hashes[i]
            
            next_hash = hashlib.sha256(combined).digest()
            next_level.append(next_hash)
        
        hashes = next_level
    
    return hashes[0].hex()


def verify_merkle_proof(leaf: str, proof: List[tuple], root: str) -> bool:
    """
    Verify a Merkle proof
    
    Args:
        leaf: Hex-encoded leaf hash
        proof: List of (hash, is_left) tuples
        root: Expected root hash
        
    Returns:
        True if proof is valid
    """
    current = bytes.fromhex(leaf)
    
    for sibling_hex, is_left in proof:
        sibling = bytes.fromhex(sibling_hex)
        
        if is_left:
            combined = sibling + current
        else:
            combined = current + sibling
        
        current = hashlib.sha256(combined).digest()
    
    return current.hex() == root