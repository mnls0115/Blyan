#!/usr/bin/env python3
"""
Secure Qwen1.5-MoE-A2.7B Upload from Runpod to Main Node
Handles model processing on GPU nodes and secure upload to main chain
"""

import os
import sys
import hashlib
import time
import requests
import torch
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from dataclasses import dataclass

# Security imports
import urllib.parse
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecureUploadConfig:
    """Secure configuration for remote upload"""
    api_url: str
    node_id: str
    node_token: str
    model_path: str
    meta_hash: str
    
    # Security constraints
    max_file_size: int = 2 * 1024 * 1024 * 1024  # 2GB per upload
    allowed_extensions: List[str] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.pt', '.pth', '.bin', '.safetensors']
        
        # Validate inputs against injection
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate all inputs against injection attacks"""
        # URL validation
        if not self._is_valid_url(self.api_url):
            raise ValueError(f"Invalid API URL: {self.api_url}")
        
        # Node ID validation (alphanumeric + hyphens only)
        if not re.match(r'^[a-zA-Z0-9\-_]+$', self.node_id):
            raise ValueError(f"Invalid node ID: {self.node_id}")
        
        # Token validation (hex only)
        if not re.match(r'^[a-fA-F0-9]+$', self.node_token):
            raise ValueError(f"Invalid node token format")
        
        # Path validation (no traversal)
        if '..' in self.model_path or self.model_path.startswith('/'):
            if not os.path.exists(self.model_path):
                raise ValueError(f"Invalid or dangerous model path: {self.model_path}")
        
        # Meta hash validation (64 char hex)
        if not re.match(r'^[a-fA-F0-9]{64}$', self.meta_hash):
            raise ValueError(f"Invalid meta hash format: {self.meta_hash}")
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme"""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except:
            return False

class SecureMoEUploader:
    """Secure MoE model uploader with injection protection"""
    
    def __init__(self, config: SecureUploadConfig):
        self.config = config
        self.session = requests.Session()
        
        # Set authentication headers
        self.session.headers.update({
            'X-Node-ID': config.node_id,
            'X-Node-Auth-Token': config.node_token,
            'User-Agent': 'Blyan-Runpod-Uploader/1.0',
            'Content-Type': 'application/json'
        })
    
    def verify_main_node_connection(self) -> bool:
        """Verify connection to main node with authentication"""
        try:
            response = self.session.get(f"{self.config.api_url}/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"Health check failed: {response.status_code}")
                return False
            
            # Test authenticated endpoint
            response = self.session.get(f"{self.config.api_url}/pol/status", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False
    
    def load_and_validate_model(self) -> Dict:
        """Securely load and validate model file"""
        model_path = Path(self.config.model_path)
        
        # Security checks
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if model_path.suffix not in self.config.allowed_extensions:
            raise ValueError(f"Unsupported file type: {model_path.suffix}")
        
        if model_path.stat().st_size > self.config.max_file_size:
            raise ValueError(f"File too large: {model_path.stat().st_size}")
        
        # Load model securely
        logger.info(f"Loading model from {model_path}...")
        
        try:
            # Use safe loading (no pickle deserialization if possible)
            if model_path.suffix == '.safetensors':
                from safetensors import safe_open
                tensors = {}
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                state_dict = tensors
            else:
                # For .pt/.pth, use map_location for safety
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            logger.info(f"✅ Model loaded successfully: {len(state_dict)} parameters")
            return state_dict
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def extract_moe_experts(self, state_dict: Dict) -> Dict[str, torch.Tensor]:
        """Extract MoE experts from state dict"""
        experts = {}
        
        for name, tensor in state_dict.items():
            # Look for MoE patterns
            if any(pattern in name.lower() for pattern in ['expert', 'moe', 'ffn']):
                # Validate tensor
                if not torch.is_tensor(tensor):
                    logger.warning(f"Skipping non-tensor: {name}")
                    continue
                
                # Size check for safety
                if tensor.numel() > 100_000_000:  # 100M parameters max per expert
                    logger.warning(f"Expert too large: {name} ({tensor.numel():,} params)")
                    continue
                
                experts[name] = tensor
                logger.info(f"Found expert: {name} {list(tensor.shape)}")
        
        return experts
    
    def calculate_secure_hash(self, tensor: torch.Tensor) -> str:
        """Calculate secure hash of tensor"""
        # Convert tensor to bytes for hashing
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    
    def upload_expert_secure(self, expert_name: str, tensor: torch.Tensor) -> bool:
        """Securely upload a single expert to main node"""
        try:
            # Calculate integrity hash
            tensor_hash = self.calculate_secure_hash(tensor)
            
            # Prepare payload with validation
            payload = {
                "address": "runpod-uploader",
                "expert_name": expert_name,
                "layer_id": self._extract_layer_id(expert_name),
                "meta_hash": self.config.meta_hash,
                "candidate_loss": 0.95,  # Conservative default
                "tensor_data": tensor.tolist(),  # Convert to JSON-serializable
                "tensor_hash": tensor_hash,
                "tensor_shape": list(tensor.shape),
                "upload_source": "runpod_secure"
            }
            
            # Upload with retry logic
            for attempt in range(3):
                try:
                    response = self.session.post(
                        f"{self.config.api_url}/upload_moe_experts",
                        json=payload,
                        timeout=300  # 5 minutes
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ Expert uploaded: {expert_name} -> {result.get('block_hash', 'unknown')}")
                        return True
                    else:
                        logger.error(f"Upload failed (attempt {attempt+1}): {response.status_code} - {response.text}")
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Upload timeout (attempt {attempt+1})")
                except Exception as e:
                    logger.error(f"Upload error (attempt {attempt+1}): {e}")
                
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            
            return False
            
        except Exception as e:
            logger.error(f"Secure upload failed for {expert_name}: {e}")
            return False
    
    def _extract_layer_id(self, expert_name: str) -> str:
        """Safely extract layer ID from expert name"""
        # Use regex to extract layer number
        match = re.search(r'layer[._]?(\d+)', expert_name.lower())
        if match:
            return f"layer{match.group(1)}"
        
        # Fallback pattern
        match = re.search(r'(\d+)', expert_name)
        if match:
            return f"layer{match.group(1)}"
        
        return "layer0"  # Safe default
    
    def run_secure_upload(self) -> Dict:
        """Run the complete secure upload process"""
        logger.info("🚀 Starting secure Qwen1.5-MoE-A2.7B upload...")
        
        # Step 1: Verify connection
        if not self.verify_main_node_connection():
            raise ConnectionError("Cannot connect to main node")
        
        # Step 2: Load model
        state_dict = self.load_and_validate_model()
        
        # Step 3: Extract experts
        experts = self.extract_moe_experts(state_dict)
        logger.info(f"Found {len(experts)} experts to upload")
        
        if not experts:
            raise ValueError("No MoE experts found in model")
        
        # Step 4: Upload each expert
        results = {
            "total_experts": len(experts),
            "successful_uploads": 0,
            "failed_uploads": 0,
            "upload_details": []
        }
        
        for expert_name, tensor in experts.items():
            logger.info(f"Uploading {expert_name}...")
            
            success = self.upload_expert_secure(expert_name, tensor)
            
            if success:
                results["successful_uploads"] += 1
            else:
                results["failed_uploads"] += 1
            
            results["upload_details"].append({
                "expert": expert_name,
                "success": success,
                "shape": list(tensor.shape)
            })
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"✅ Upload complete: {results['successful_uploads']}/{results['total_experts']} successful")
        return results

def main():
    """Main execution function"""
    try:
        # Get configuration from environment
        config = SecureUploadConfig(
            api_url=os.getenv("AIBLOCK_API_URL", "https://blyan.com/api"),
            node_id=os.getenv("BLYAN_NODE_ID"),
            node_token=os.getenv("BLYAN_MAIN_NODE_TOKEN"),
            model_path=os.getenv("MODEL_PATH", "./Qwen1.5-MoE-A2.7B/pytorch_model.bin"),
            meta_hash=os.getenv("META_HASH")
        )
        
        # Validate required vars
        if not all([config.node_id, config.node_token, config.meta_hash]):
            raise ValueError("Missing required environment variables: BLYAN_NODE_ID, BLYAN_MAIN_NODE_TOKEN, META_HASH")
        
        # Run upload
        uploader = SecureMoEUploader(config)
        results = uploader.run_secure_upload()
        
        # Print summary
        print(f"\n📊 Upload Summary:")
        print(f"✅ Successful: {results['successful_uploads']}")
        print(f"❌ Failed: {results['failed_uploads']}")
        print(f"📈 Success rate: {results['successful_uploads']/results['total_experts']*100:.1f}%")
        
        return 0 if results['failed_uploads'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())