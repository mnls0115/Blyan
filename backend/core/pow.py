import hashlib

# ---------------------------------------------------------------------
# Very simple PoW: find nonce such that sha256(data || nonce) starts with
# `difficulty` leading zeroes in hex representation.
# ---------------------------------------------------------------------

def find_nonce(data: bytes, difficulty: int = 4) -> int:
    prefix = "0" * difficulty
    nonce = 0
    while True:
        digest = hashlib.sha256(data + str(nonce).encode()).hexdigest()
        if digest.startswith(prefix):
            return nonce
        nonce += 1


def verify_pow(data: bytes, nonce: int, difficulty: int = 4) -> bool:
    prefix = "0" * difficulty
    digest = hashlib.sha256(data + str(nonce).encode()).hexdigest()
    return digest.startswith(prefix) 