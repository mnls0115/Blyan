"""Ed25519 signature implementation for P2P messages and blocks"""
import nacl.signing
import nacl.encoding
import nacl.hash
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
import base64
import json
from typing import Union, Dict, Any
from pathlib import Path


class KeyPair:
    """Ed25519 key pair management"""
    
    def __init__(self, private_key: bytes = None):
        if private_key:
            self.signing_key = SigningKey(private_key)
        else:
            self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key
        
    @classmethod
    def from_file(cls, path: Path) -> 'KeyPair':
        """Load keypair from file"""
        if path.exists():
            with open(path, 'rb') as f:
                private_key = f.read()
            return cls(private_key)
        else:
            # Generate new and save
            keypair = cls()
            keypair.save(path)
            return keypair
    
    def save(self, path: Path):
        """Save private key to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(bytes(self.signing_key))
        path.chmod(0o600)  # Private key protection
    
    @property
    def public_key(self) -> bytes:
        """Get public key bytes"""
        return bytes(self.verify_key)
    
    @property
    def public_key_hex(self) -> str:
        """Get hex-encoded public key"""
        return self.verify_key.encode(encoder=nacl.encoding.HexEncoder).decode()
    
    @property
    def address(self) -> str:
        """Get address from public key (first 20 bytes of hash)"""
        hasher = nacl.hash.sha256
        pub_hash = hasher(bytes(self.verify_key), encoder=nacl.encoding.RawEncoder)
        return pub_hash[:20].hex()
    
    def sign(self, message: bytes) -> bytes:
        """Sign a message"""
        signed = self.signing_key.sign(message)
        return signed.signature
    
    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify a signature"""
        try:
            self.verify_key.verify(message, signature)
            return True
        except BadSignatureError:
            return False


def sign_message(message: Union[bytes, str, dict], keypair: KeyPair) -> Dict[str, Any]:
    """
    Sign a message with Ed25519
    
    Args:
        message: Message to sign (bytes, string, or dict)
        keypair: KeyPair instance
        
    Returns:
        Dict with message, signature, and public key
    """
    # Convert message to bytes
    if isinstance(message, dict):
        msg_bytes = json.dumps(message, sort_keys=True).encode()
    elif isinstance(message, str):
        msg_bytes = message.encode()
    else:
        msg_bytes = message
    
    # Sign the message
    signature = keypair.sign(msg_bytes)
    
    return {
        'message': base64.b64encode(msg_bytes).decode() if isinstance(message, bytes) else message,
        'signature': base64.b64encode(signature).decode(),
        'public_key': keypair.public_key_hex,
        'address': keypair.address
    }


def verify_signature(signed_data: Dict[str, Any]) -> bool:
    """
    Verify an Ed25519 signature
    
    Args:
        signed_data: Dict with message, signature, and public_key
        
    Returns:
        True if signature is valid
    """
    try:
        # Decode components
        if isinstance(signed_data['message'], str) and signed_data['message'].startswith('ey'):
            # Base64 encoded
            msg_bytes = base64.b64decode(signed_data['message'])
        elif isinstance(signed_data['message'], dict):
            msg_bytes = json.dumps(signed_data['message'], sort_keys=True).encode()
        else:
            msg_bytes = signed_data['message'].encode()
        
        signature = base64.b64decode(signed_data['signature'])
        public_key_hex = signed_data['public_key']
        
        # Create verify key
        verify_key = VerifyKey(bytes.fromhex(public_key_hex))
        
        # Verify
        verify_key.verify(msg_bytes, signature)
        return True
        
    except (BadSignatureError, KeyError, ValueError):
        return False


def sign_block(block: dict, keypair: KeyPair) -> dict:
    """
    Sign a block with all its transactions
    
    Args:
        block: Block dict to sign
        keypair: Miner's keypair
        
    Returns:
        Signed block
    """
    # Create block header for signing (exclude signature field)
    header = {
        'height': block['height'],
        'parent_hash': block['parent_hash'],
        'timestamp': block['timestamp'],
        'transactions': block.get('transactions', []),
        'state_root': block.get('state_root', ''),
        'miner': keypair.address
    }
    
    # Sign the header
    header_bytes = json.dumps(header, sort_keys=True).encode()
    signature = keypair.sign(header_bytes)
    
    # Add signature to block
    block['signature'] = base64.b64encode(signature).decode()
    block['miner'] = keypair.address
    block['public_key'] = keypair.public_key_hex
    
    return block


def verify_block_signature(block: dict) -> bool:
    """
    Verify a block's signature
    
    Args:
        block: Signed block dict
        
    Returns:
        True if signature is valid
    """
    try:
        # Recreate header for verification
        header = {
            'height': block['height'],
            'parent_hash': block['parent_hash'],
            'timestamp': block['timestamp'],
            'transactions': block.get('transactions', []),
            'state_root': block.get('state_root', ''),
            'miner': block['miner']
        }
        
        header_bytes = json.dumps(header, sort_keys=True).encode()
        signature = base64.b64decode(block['signature'])
        
        # Verify with public key
        verify_key = VerifyKey(bytes.fromhex(block['public_key']))
        verify_key.verify(header_bytes, signature)
        
        # Also verify miner address matches public key
        hasher = nacl.hash.sha256
        pub_hash = hasher(bytes(verify_key), encoder=nacl.encoding.RawEncoder)
        expected_address = pub_hash[:20].hex()
        
        return expected_address == block['miner']
        
    except (BadSignatureError, KeyError, ValueError):
        return False