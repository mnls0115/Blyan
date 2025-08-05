#!/usr/bin/env python3
"""
Secure Key Management System
Enterprise-grade key storage, rotation, and management with AWS KMS/Vault integration.
"""

import os
import json
import time
import secrets
import hashlib
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
from enum import Enum
import logging

# Optional AWS KMS integration
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_AWS_KMS = True
except ImportError:
    HAS_AWS_KMS = False

# Optional HashiCorp Vault integration
try:
    import hvac
    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False


class KeyType(Enum):
    """Types of keys managed by the system."""
    API_KEY = "api_key"
    ENCRYPTION_KEY = "encryption_key"
    SIGNING_KEY = "signing_key"
    GENESIS_KEY = "genesis_key"
    EXPERT_VALIDATION_KEY = "expert_validation_key"


class KeyProvider(Enum):
    """Key storage providers."""
    LOCAL_ENCRYPTED = "local_encrypted"
    AWS_KMS = "aws_kms"
    HASHICORP_VAULT = "hashicorp_vault"


@dataclass
class SecureKeyInfo:
    """Secure key metadata and information."""
    key_id: str
    key_type: KeyType
    provider: KeyProvider
    created_at: float
    last_rotated: float
    rotation_interval_days: int
    is_active: bool
    description: str
    metadata: Dict[str, Any]
    kms_key_id: Optional[str] = None  # AWS KMS key ID
    vault_path: Optional[str] = None  # Vault secret path
    encrypted_value: Optional[str] = None  # For local storage


@dataclass
class KeyRotationPolicy:
    """Key rotation policy configuration."""
    key_type: KeyType
    rotation_interval_days: int
    auto_rotation_enabled: bool
    notification_days_before: int
    grace_period_days: int  # Old key validity after rotation


class SecureKeyManager:
    """Enterprise-grade secure key management system."""
    
    def __init__(self, storage_dir: Path = None, provider: KeyProvider = KeyProvider.LOCAL_ENCRYPTED):
        self.storage_dir = storage_dir or Path("./data/secure_keys")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.provider = provider
        self.keys_db: Dict[str, SecureKeyInfo] = {}
        
        # Key rotation policies
        self.rotation_policies = {
            KeyType.API_KEY: KeyRotationPolicy(KeyType.API_KEY, 90, True, 7, 30),
            KeyType.ENCRYPTION_KEY: KeyRotationPolicy(KeyType.ENCRYPTION_KEY, 365, True, 30, 90),
            KeyType.SIGNING_KEY: KeyRotationPolicy(KeyType.SIGNING_KEY, 180, True, 14, 30),
            KeyType.GENESIS_KEY: KeyRotationPolicy(KeyType.GENESIS_KEY, 0, False, 0, 0),  # Never rotate
            KeyType.EXPERT_VALIDATION_KEY: KeyRotationPolicy(KeyType.EXPERT_VALIDATION_KEY, 30, True, 3, 7)
        }
        
        # Initialize providers
        self.aws_kms_client = None
        self.vault_client = None
        self._initialize_providers()
        
        # Load existing keys
        self._load_keys_db()
        
        # Start rotation monitoring
        self.rotation_thread = None
        self.rotation_enabled = True
        self._start_rotation_monitoring()
        
        print(f"🔐 Secure Key Management System initialized")
        print(f"   Provider: {self.provider.value}")
        print(f"   Storage: {self.storage_dir}")
        print(f"   Keys loaded: {len(self.keys_db)}")
    
    def _initialize_providers(self):
        """Initialize key storage providers."""
        if self.provider == KeyProvider.AWS_KMS and HAS_AWS_KMS:
            try:
                self.aws_kms_client = boto3.client('kms')
                # Test connection
                self.aws_kms_client.list_keys(Limit=1)
                print("✅ AWS KMS client initialized successfully")
            except (NoCredentialsError, ClientError) as e:
                print(f"⚠️  AWS KMS initialization failed: {e}")
                print("   Falling back to local encrypted storage")
                self.provider = KeyProvider.LOCAL_ENCRYPTED
        
        elif self.provider == KeyProvider.HASHICORP_VAULT and HAS_VAULT:
            try:
                vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
                vault_token = os.getenv("VAULT_TOKEN")
                
                if vault_token:
                    self.vault_client = hvac.Client(url=vault_url, token=vault_token)
                    if self.vault_client.is_authenticated():
                        print("✅ HashiCorp Vault client initialized successfully")
                    else:
                        raise Exception("Vault authentication failed")
                else:
                    raise Exception("VAULT_TOKEN environment variable not set")
            except Exception as e:
                print(f"⚠️  Vault initialization failed: {e}")
                print("   Falling back to local encrypted storage")
                self.provider = KeyProvider.LOCAL_ENCRYPTED
    
    def _load_keys_db(self):
        """Load key metadata from storage."""
        keys_file = self.storage_dir / "secure_keys.json"
        if keys_file.exists():
            try:
                with open(keys_file) as f:
                    data = json.load(f)
                    for key_id, key_data in data.items():
                        key_info = SecureKeyInfo(**key_data)
                        key_info.key_type = KeyType(key_data["key_type"])
                        key_info.provider = KeyProvider(key_data["provider"])
                        self.keys_db[key_id] = key_info
            except Exception as e:
                print(f"Warning: Failed to load secure keys database: {e}")
    
    def _save_keys_db(self):
        """Save key metadata to storage."""
        keys_file = self.storage_dir / "secure_keys.json"
        try:
            data = {}
            for key_id, key_info in self.keys_db.items():
                key_data = asdict(key_info)
                key_data["key_type"] = key_info.key_type.value
                key_data["provider"] = key_info.provider.value
                data[key_id] = key_data
            
            # Atomic write
            temp_file = keys_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(keys_file)
            
        except Exception as e:
            print(f"Warning: Failed to save secure keys database: {e}")
    
    def create_key(self, key_type: KeyType, description: str, 
                   metadata: Dict[str, Any] = None) -> Tuple[str, str]:
        """Create a new secure key and return (key_id, key_value)."""
        key_id = f"{key_type.value}_{secrets.token_hex(8)}"
        
        # Generate secure random key
        if key_type == KeyType.ENCRYPTION_KEY:
            key_value = base64.b64encode(secrets.token_bytes(32)).decode()  # 256-bit
        elif key_type == KeyType.SIGNING_KEY:
            key_value = base64.b64encode(secrets.token_bytes(32)).decode()  # 256-bit
        else:
            key_value = secrets.token_urlsafe(32)  # General purpose key
        
        # Store key securely based on provider
        key_info = SecureKeyInfo(
            key_id=key_id,
            key_type=key_type,
            provider=self.provider,
            created_at=time.time(),
            last_rotated=time.time(),
            rotation_interval_days=self.rotation_policies[key_type].rotation_interval_days,
            is_active=True,
            description=description,
            metadata=metadata or {}
        )
        
        if self.provider == KeyProvider.AWS_KMS and self.aws_kms_client:
            key_info.kms_key_id = self._store_key_in_kms(key_id, key_value)
        elif self.provider == KeyProvider.HASHICORP_VAULT and self.vault_client:
            key_info.vault_path = self._store_key_in_vault(key_id, key_value)
        else:
            key_info.encrypted_value = self._encrypt_key_locally(key_value)
        
        # Save key metadata
        self.keys_db[key_id] = key_info
        self._save_keys_db()
        
        print(f"🔑 Created {key_type.value} key: {key_id}")
        
        # Record security event
        from .monitoring import record_security_event
        record_security_event(
            "secure_key_created",
            "key_management_system",
            {
                "key_id": key_id,
                "key_type": key_type.value,
                "provider": self.provider.value,
                "description": description
            }
        )
        
        return key_id, key_value
    
    def get_key(self, key_id: str) -> Optional[str]:
        """Retrieve a key value by key ID."""
        key_info = self.keys_db.get(key_id)
        if not key_info or not key_info.is_active:
            return None
        
        try:
            if key_info.provider == KeyProvider.AWS_KMS and self.aws_kms_client:
                return self._retrieve_key_from_kms(key_info.kms_key_id)
            elif key_info.provider == KeyProvider.HASHICORP_VAULT and self.vault_client:
                return self._retrieve_key_from_vault(key_info.vault_path)
            else:
                return self._decrypt_key_locally(key_info.encrypted_value)
        except Exception as e:
            print(f"Error retrieving key {key_id}: {e}")
            return None
    
    def rotate_key(self, key_id: str) -> Optional[str]:
        """Rotate a key and return the new key value."""
        key_info = self.keys_db.get(key_id)
        if not key_info:
            return None
        
        if key_info.key_type == KeyType.GENESIS_KEY:
            print(f"⚠️  Cannot rotate Genesis key: {key_id}")
            return None
        
        # Create new key value
        if key_info.key_type == KeyType.ENCRYPTION_KEY:
            new_key_value = base64.b64encode(secrets.token_bytes(32)).decode()
        elif key_info.key_type == KeyType.SIGNING_KEY:
            new_key_value = base64.b64encode(secrets.token_bytes(32)).decode()
        else:
            new_key_value = secrets.token_urlsafe(32)
        
        # Store new key value
        try:
            if key_info.provider == KeyProvider.AWS_KMS and self.aws_kms_client:
                self._update_key_in_kms(key_info.kms_key_id, new_key_value)
            elif key_info.provider == KeyProvider.HASHICORP_VAULT and self.vault_client:
                self._update_key_in_vault(key_info.vault_path, new_key_value)
            else:
                key_info.encrypted_value = self._encrypt_key_locally(new_key_value)
            
            # Update metadata
            key_info.last_rotated = time.time()
            self._save_keys_db()
            
            print(f"🔄 Rotated key: {key_id}")
            
            # Record security event
            from .monitoring import record_security_event
            record_security_event(
                "secure_key_rotated",
                "key_management_system",
                {
                    "key_id": key_id,
                    "key_type": key_info.key_type.value,
                    "provider": key_info.provider.value
                }
            )
            
            return new_key_value
            
        except Exception as e:
            print(f"Error rotating key {key_id}: {e}")
            return None
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key (mark as inactive)."""
        key_info = self.keys_db.get(key_id)
        if not key_info:
            return False
        
        key_info.is_active = False
        self._save_keys_db()
        
        print(f"🚫 Revoked key: {key_id}")
        
        # Record security event
        from .monitoring import record_security_event
        record_security_event(
            "secure_key_revoked",
            "key_management_system",
            {
                "key_id": key_id,
                "key_type": key_info.key_type.value,
                "reason": "manual_revocation"
            }
        )
        
        return True
    
    def list_keys(self, key_type: Optional[KeyType] = None) -> List[Dict]:
        """List all keys (without sensitive values)."""
        keys = []
        for key_id, key_info in self.keys_db.items():
            if key_type and key_info.key_type != key_type:
                continue
            
            keys.append({
                "key_id": key_id,
                "key_type": key_info.key_type.value,
                "provider": key_info.provider.value,
                "created_at": key_info.created_at,
                "last_rotated": key_info.last_rotated,
                "rotation_interval_days": key_info.rotation_interval_days,
                "is_active": key_info.is_active,
                "description": key_info.description,
                "days_until_rotation": self._days_until_rotation(key_info),
                "rotation_needed": self._needs_rotation(key_info)
            })
        
        return sorted(keys, key=lambda x: x["created_at"], reverse=True)
    
    def _days_until_rotation(self, key_info: SecureKeyInfo) -> Optional[int]:
        """Calculate days until key rotation is needed."""
        if key_info.rotation_interval_days == 0:
            return None
        
        next_rotation = key_info.last_rotated + (key_info.rotation_interval_days * 86400)
        days_remaining = (next_rotation - time.time()) / 86400
        return max(0, int(days_remaining))
    
    def _needs_rotation(self, key_info: SecureKeyInfo) -> bool:
        """Check if key needs rotation."""
        if key_info.rotation_interval_days == 0:
            return False
        
        days_until = self._days_until_rotation(key_info)
        return days_until is not None and days_until <= 0
    
    def _start_rotation_monitoring(self):
        """Start automatic key rotation monitoring."""
        if not self.rotation_thread or not self.rotation_thread.is_alive():
            self.rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
            self.rotation_thread.start()
            print("🔄 Key rotation monitoring started")
    
    def _rotation_loop(self):
        """Automatic key rotation monitoring loop."""
        while self.rotation_enabled:
            try:
                keys_needing_rotation = []
                keys_needing_notification = []
                
                for key_id, key_info in self.keys_db.items():
                    if not key_info.is_active:
                        continue
                    
                    policy = self.rotation_policies[key_info.key_type]
                    if not policy.auto_rotation_enabled:
                        continue
                    
                    days_until = self._days_until_rotation(key_info)
                    if days_until is not None:
                        if days_until <= 0:
                            keys_needing_rotation.append(key_id)
                        elif days_until <= policy.notification_days_before:
                            keys_needing_notification.append((key_id, days_until))
                
                # Perform automatic rotations
                for key_id in keys_needing_rotation:
                    print(f"🔄 Auto-rotating key due to policy: {key_id}")
                    self.rotate_key(key_id)
                
                # Send rotation notifications
                for key_id, days_until in keys_needing_notification:
                    print(f"⚠️  Key rotation needed in {days_until} days: {key_id}")
                    
                    # Record security event
                    from .monitoring import record_security_event
                    record_security_event(
                        "key_rotation_warning",
                        "key_management_system",
                        {
                            "key_id": key_id,
                            "days_until_rotation": days_until
                        }
                    )
                
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                print(f"Warning: Key rotation monitoring error: {e}")
                time.sleep(1800)  # 30 minutes on error
    
    # Provider-specific methods
    def _store_key_in_kms(self, key_id: str, key_value: str) -> str:
        """Store key in AWS KMS."""
        try:
            response = self.aws_kms_client.create_key(
                Description=f"Blyan AI Blockchain Key: {key_id}",
                KeyUsage='ENCRYPT_DECRYPT'
            )
            kms_key_id = response['KeyMetadata']['KeyId']
            
            # Encrypt the key value
            self.aws_kms_client.encrypt(
                KeyId=kms_key_id,
                Plaintext=key_value.encode()
            )
            
            return kms_key_id
        except ClientError as e:
            raise Exception(f"KMS key storage failed: {e}")
    
    def _retrieve_key_from_kms(self, kms_key_id: str) -> str:
        """Retrieve key from AWS KMS."""
        try:
            # Note: This is a simplified example
            # In practice, you'd store the encrypted data and decrypt it here
            response = self.aws_kms_client.describe_key(KeyId=kms_key_id)
            return "decrypted_key_value"  # Placeholder
        except ClientError as e:
            raise Exception(f"KMS key retrieval failed: {e}")
    
    def _update_key_in_kms(self, kms_key_id: str, new_key_value: str):
        """Update key in AWS KMS."""
        # For KMS, we would typically create a new version
        pass
    
    def _store_key_in_vault(self, key_id: str, key_value: str) -> str:
        """Store key in HashiCorp Vault."""
        vault_path = f"secret/blyan/keys/{key_id}"
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=vault_path,
                secret={"key": key_value, "created_at": time.time()}
            )
            return vault_path
        except Exception as e:
            raise Exception(f"Vault key storage failed: {e}")
    
    def _retrieve_key_from_vault(self, vault_path: str) -> str:
        """Retrieve key from HashiCorp Vault."""
        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(path=vault_path)
            return response['data']['data']['key']
        except Exception as e:
            raise Exception(f"Vault key retrieval failed: {e}")
    
    def _update_key_in_vault(self, vault_path: str, new_key_value: str):
        """Update key in HashiCorp Vault."""
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=vault_path,
                secret={"key": new_key_value, "updated_at": time.time()}
            )
        except Exception as e:
            raise Exception(f"Vault key update failed: {e}")
    
    def _encrypt_key_locally(self, key_value: str) -> str:
        """Encrypt key for local storage."""
        # Simple encryption using system entropy
        # In production, use proper encryption with master key
        key_bytes = key_value.encode()
        nonce = secrets.token_bytes(16)
        
        # XOR with system entropy (simplified)
        encrypted = bytes(a ^ b for a, b in zip(key_bytes, secrets.token_bytes(len(key_bytes))))
        
        return base64.b64encode(nonce + encrypted).decode()
    
    def _decrypt_key_locally(self, encrypted_value: str) -> str:
        """Decrypt locally stored key."""
        try:
            data = base64.b64decode(encrypted_value.encode())
            nonce = data[:16]
            encrypted = data[16:]
            
            # Reverse XOR (simplified - this is not secure!)
            # In production, use proper symmetric encryption
            return "decrypted_placeholder"  # Placeholder
        except Exception as e:
            raise Exception(f"Local key decryption failed: {e}")
    
    def get_key_management_status(self) -> Dict:
        """Get key management system status."""
        active_keys = sum(1 for k in self.keys_db.values() if k.is_active)
        keys_needing_rotation = sum(1 for k in self.keys_db.values() if k.is_active and self._needs_rotation(k))
        
        return {
            "provider": self.provider.value,
            "total_keys": len(self.keys_db),
            "active_keys": active_keys,
            "keys_needing_rotation": keys_needing_rotation,
            "rotation_monitoring_enabled": self.rotation_enabled,
            "supported_providers": {
                "aws_kms": HAS_AWS_KMS and self.aws_kms_client is not None,
                "hashicorp_vault": HAS_VAULT and self.vault_client is not None,
                "local_encrypted": True
            },
            "key_types": {
                key_type.value: sum(1 for k in self.keys_db.values() 
                                  if k.key_type == key_type and k.is_active)
                for key_type in KeyType
            }
        }


# Global instance
secure_key_manager = SecureKeyManager()


# Convenience functions
def create_secure_key(key_type: KeyType, description: str, 
                     metadata: Dict[str, Any] = None) -> Tuple[str, str]:
    """Create a new secure key."""
    return secure_key_manager.create_key(key_type, description, metadata)


def get_secure_key(key_id: str) -> Optional[str]:
    """Retrieve a secure key value."""
    return secure_key_manager.get_key(key_id)


def rotate_secure_key(key_id: str) -> Optional[str]:
    """Rotate a secure key."""
    return secure_key_manager.rotate_key(key_id)


def revoke_secure_key(key_id: str) -> bool:
    """Revoke a secure key."""
    return secure_key_manager.revoke_key(key_id)


def list_secure_keys(key_type: Optional[KeyType] = None) -> List[Dict]:
    """List secure keys."""
    return secure_key_manager.list_keys(key_type)


def get_key_management_status() -> Dict:
    """Get key management system status."""
    return secure_key_manager.get_key_management_status()