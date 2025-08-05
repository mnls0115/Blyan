import hashlib
import time

# ---------------------------------------------------------------------
# PoL-First Security: Computational proof that contributes to AI learning
# instead of wasteful hash calculations
# ---------------------------------------------------------------------

def find_pol_nonce(data: bytes, contributor_id: str, difficulty: int = 1) -> int:
    """
    Generate PoL nonce based on AI contribution proof instead of computational waste.
    This is a lightweight challenge that proves legitimate participation.
    """
    # Simple anti-spam nonce (3-second CPU challenge)
    base_string = f"{data.hex()}{contributor_id}{int(time.time() // 60)}"  # 1-minute window
    nonce = 0
    
    # Much easier than traditional PoW - focuses on anti-spam, not waste
    while nonce < difficulty * 1000:  # Max 1000 iterations for anti-spam
        test_string = f"{base_string}{nonce}"
        digest = hashlib.sha256(test_string.encode()).hexdigest()
        # Only require last digit to be '0' - very lightweight
        if digest[-1] == '0':
            return nonce
        nonce += 1
    
    # Return simple nonce if no match found (anti-spam purpose only)
    return nonce


def verify_pol_nonce(data: bytes, nonce: int, contributor_id: str, timestamp_window: int = 60) -> bool:
    """
    Verify PoL nonce - much faster than traditional PoW verification.
    Focuses on legitimate participation rather than computational waste.
    """
    current_time = int(time.time() // 60)  # 1-minute granularity
    
    # Check multiple time windows to account for clock drift
    for time_offset in range(-2, 3):  # ±2 minute tolerance
        window_time = current_time + time_offset
        base_string = f"{data.hex()}{contributor_id}{window_time}"
        test_string = f"{base_string}{nonce}"
        digest = hashlib.sha256(test_string.encode()).hexdigest()
        
        if digest[-1] == '0':
            return True
    
    return False


def generate_contribution_proof(expert_hash: str, performance_improvement: float) -> dict:
    """
    Generate proof of AI contribution for rate limiting bypass.
    This replaces economic barriers with technical merit.
    """
    return {
        "expert_hash": expert_hash,
        "performance_improvement": performance_improvement,
        "contribution_type": "pol_proof",
        "timestamp": time.time(),
        "proof_hash": hashlib.sha256(f"{expert_hash}{performance_improvement}".encode()).hexdigest()
    }


def validate_contribution_proof(proof: dict, minimum_improvement: float = 0.01) -> bool:
    """
    Validate AI contribution proof for quota bypass.
    Technical merit replaces economic payment.
    """
    if proof["performance_improvement"] >= minimum_improvement:
        # Verify proof hash integrity
        expected_hash = hashlib.sha256(
            f"{proof['expert_hash']}{proof['performance_improvement']}".encode()
        ).hexdigest()
        return proof["proof_hash"] == expected_hash
    return False