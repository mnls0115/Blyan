"""
Input Validation and Injection Protection
Comprehensive security layer for all user inputs
"""

import re
import os
import hashlib
import urllib.parse
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Security validation error"""
    pass

class InputValidator:
    """Comprehensive input validation with injection protection"""
    
    # Safe patterns
    SAFE_FILENAME = re.compile(r'^[a-zA-Z0-9._-]+$')
    SAFE_NODE_ID = re.compile(r'^[a-zA-Z0-9\-_]+$')
    SAFE_HEX_HASH = re.compile(r'^[a-fA-F0-9]+$')
    SAFE_EXPERT_NAME = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    # Dangerous patterns
    PATH_TRAVERSAL = re.compile(r'\.\./')
    COMMAND_INJECTION = re.compile(r'[;&|`$(){}[\]<>]')
    SQL_INJECTION = re.compile(r"[';\"\\]|(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b)", re.IGNORECASE)
    
    @classmethod
    def validate_node_id(cls, node_id: str) -> str:
        """Validate node ID format"""
        if not node_id or len(node_id) > 50:
            raise SecurityError("Invalid node ID length")
        
        if not cls.SAFE_NODE_ID.match(node_id):
            raise SecurityError("Invalid node ID format")
        
        return node_id
    
    @classmethod
    def validate_hash(cls, hash_value: str, expected_length: int = 64) -> str:
        """Validate hash format"""
        if not hash_value or len(hash_value) != expected_length:
            raise SecurityError(f"Invalid hash length: expected {expected_length}")
        
        if not cls.SAFE_HEX_HASH.match(hash_value):
            raise SecurityError("Invalid hash format")
        
        return hash_value.lower()
    
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate filename for safety"""
        if not filename or len(filename) > 255:
            raise SecurityError("Invalid filename length")
        
        if cls.PATH_TRAVERSAL.search(filename):
            raise SecurityError("Path traversal detected in filename")
        
        if not cls.SAFE_FILENAME.match(filename):
            raise SecurityError("Invalid filename format")
        
        return filename
    
    @classmethod
    def validate_expert_name(cls, expert_name: str) -> str:
        """Validate expert name"""
        if not expert_name or len(expert_name) > 100:
            raise SecurityError("Invalid expert name length")
        
        if not cls.SAFE_EXPERT_NAME.match(expert_name):
            raise SecurityError("Invalid expert name format")
        
        return expert_name
    
    @classmethod
    def validate_file_path(cls, file_path: str, allowed_dirs: List[str] = None) -> Path:
        """Validate file path for safety"""
        path = Path(file_path).resolve()
        
        # Check for path traversal
        if '..' in str(path):
            raise SecurityError("Path traversal detected")
        
        # Check allowed directories
        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    path.relative_to(Path(allowed_dir).resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                raise SecurityError(f"Path not in allowed directories: {path}")
        
        return path
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate URL format"""
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                raise SecurityError("Invalid URL scheme")
            
            if not parsed.netloc:
                raise SecurityError("Invalid URL netloc")
            
            return url
        except Exception as e:
            raise SecurityError(f"URL validation failed: {e}")
    
    @classmethod
    def validate_json_payload(cls, payload: Dict[str, Any], max_size: int = None) -> Dict[str, Any]:
        """Validate JSON payload size and content"""
        # Use network config max block size if not specified
        if max_size is None:
            from backend.config.network_config import get_network_config
            max_size = get_network_config().get_max_block_size()
        
        # Size check
        payload_str = json.dumps(payload)
        if len(payload_str) > max_size:
            raise SecurityError(f"Payload too large: {len(payload_str)} > {max_size}")
        
        # Check for dangerous patterns in string values
        def check_strings(obj):
            if isinstance(obj, str):
                if cls.COMMAND_INJECTION.search(obj):
                    raise SecurityError("Command injection pattern detected")
                if cls.SQL_INJECTION.search(obj):
                    raise SecurityError("SQL injection pattern detected")
            elif isinstance(obj, dict):
                for v in obj.values():
                    check_strings(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_strings(item)
        
        check_strings(payload)
        return payload
    
    @classmethod
    def sanitize_log_message(cls, message: str) -> str:
        """Sanitize log message to prevent log injection"""
        # Remove control characters
        message = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', message)
        
        # Limit length
        if len(message) > 500:
            message = message[:500] + "..."
        
        return message

class SecureFileHandler:
    """Secure file operations with validation"""
    
    def __init__(self, allowed_dirs: List[str], max_file_size: int = 2_000_000_000):
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.max_file_size = max_file_size
    
    def validate_file_access(self, file_path: str) -> Path:
        """Validate file can be safely accessed"""
        path = Path(file_path).resolve()
        
        # Check if in allowed directory
        allowed = False
        for allowed_dir in self.allowed_dirs:
            try:
                path.relative_to(allowed_dir)
                allowed = True
                break
            except ValueError:
                continue
        
        if not allowed:
            raise SecurityError(f"File access denied: {path}")
        
        return path
    
    def safe_read(self, file_path: str, max_size: Optional[int] = None) -> bytes:
        """Safely read file with size limits"""
        path = self.validate_file_access(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check file size
        file_size = path.stat().st_size
        max_allowed = max_size or self.max_file_size
        
        if file_size > max_allowed:
            raise SecurityError(f"File too large: {file_size} > {max_allowed}")
        
        # Read with size limit enforcement
        with open(path, 'rb') as f:
            data = f.read(max_allowed + 1)
            if len(data) > max_allowed:
                raise SecurityError("File size exceeded during read")
            
            return data[:max_allowed]
    
    def safe_write(self, file_path: str, data: bytes, overwrite: bool = False) -> bool:
        """Safely write file with validation"""
        path = self.validate_file_access(file_path)
        
        # Check size
        if len(data) > self.max_file_size:
            raise SecurityError(f"Data too large: {len(data)} > {self.max_file_size}")
        
        # Check overwrite permission
        if path.exists() and not overwrite:
            raise SecurityError("File exists and overwrite not allowed")
        
        # Write atomically
        temp_path = path.with_suffix(path.suffix + '.tmp')
        try:
            with open(temp_path, 'wb') as f:
                f.write(data)
            
            temp_path.replace(path)
            return True
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise SecurityError(f"Write failed: {e}")

# Validation decorators for FastAPI
def validate_request_data(func):
    """Decorator to validate request data"""
    def wrapper(request_data, *args, **kwargs):
        try:
            # Validate as JSON
            if hasattr(request_data, 'dict'):
                payload = request_data.dict()
            else:
                payload = request_data
            
            InputValidator.validate_json_payload(payload)
            
            return func(request_data, *args, **kwargs)
            
        except SecurityError as e:
            logger.warning(f"Security validation failed: {e}")
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    
    return wrapper


def validate_heartbeat_params(load_factor: float, net_mbps: float = None, latency_ms: float = None, vram_gb: float = None, tflops_est: float = None) -> None:
    """Stricter validation for heartbeat inputs to prevent abuse."""
    if not (0.0 <= load_factor <= 1.0):
        raise SecurityError("load_factor out of range")
    for name, val in (('net_mbps', net_mbps), ('latency_ms', latency_ms), ('vram_gb', vram_gb), ('tflops_est', tflops_est)):
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            raise SecurityError(f"{name} must be numeric")
        # allow reasonable bounds
        if name == 'net_mbps' and not (0 <= val <= 100000):
            raise SecurityError("net_mbps out of bounds")
        if name == 'latency_ms' and not (0 <= val <= 100000):
            raise SecurityError("latency_ms out of bounds")
        if name == 'vram_gb' and not (0 <= val <= 200):
            raise SecurityError("vram_gb out of bounds")
        if name == 'tflops_est' and not (0 <= val <= 10000):
            raise SecurityError("tflops_est out of bounds")

# Global instance
file_handler = SecureFileHandler([
    "./data",
    "./models", 
    "./cache",
    "/tmp/blyan"  # If using temp directory
])

# Export commonly used functions
def validate_node_id(node_id: str) -> str:
    return InputValidator.validate_node_id(node_id)

def validate_hash(hash_value: str) -> str:
    return InputValidator.validate_hash(hash_value)

def validate_expert_name(expert_name: str) -> str:
    return InputValidator.validate_expert_name(expert_name)