#!/usr/bin/env python3
"""
Node Bootstrap Script
Handles JOIN_CODE exchange and credential management on node startup
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("node_bootstrap")

class NodeBootstrap:
    """Handles node enrollment and credential management"""
    
    def __init__(self):
        """Initialize bootstrap configuration"""
        # Paths
        self.data_dir = Path(os.getenv("DATA_DIR", "/data"))
        self.credentials_file = self.data_dir / "credentials.json"
        self.config_file = self.data_dir / "node_config.json"
        
        # Server configuration
        self.main_server = os.getenv("MAIN_SERVER_URL", "https://blyan.com/api")
        self.join_code = os.getenv("JOIN_CODE", "").strip()
        
        # Node metadata
        self.node_meta = self._collect_node_metadata()
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _collect_node_metadata(self) -> Dict[str, Any]:
        """Collect node metadata for enrollment"""
        metadata = {
            "hostname": os.uname().nodename,
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "public_ip": os.getenv("PUBLIC_IP", "unknown"),
            "node_port": os.getenv("NODE_PORT", "8001"),
            "blockchain_only": os.getenv("BLOCKCHAIN_ONLY", "false").lower() == "true",
            "donor_mode": os.getenv("DONOR_MODE", "false").lower() == "true",
        }
        
        # GPU information if available
        try:
            import torch
            if torch.cuda.is_available():
                metadata["gpu_available"] = True
                metadata["gpu_count"] = torch.cuda.device_count()
                metadata["gpu_name"] = torch.cuda.get_device_name(0)
                metadata["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                )
            else:
                metadata["gpu_available"] = False
        except ImportError:
            metadata["gpu_available"] = False
            logger.info("PyTorch not available, skipping GPU detection")
        except Exception as e:
            metadata["gpu_available"] = False
            logger.warning(f"Error detecting GPU: {e}")
        
        # Model information
        model_path = os.getenv("MODEL_PATH", "./models")
        if Path(model_path).exists():
            metadata["available_models"] = [
                d.name for d in Path(model_path).iterdir() if d.is_dir()
            ]
        
        return metadata
    
    def has_credentials(self) -> bool:
        """Check if valid credentials exist"""
        if not self.credentials_file.exists():
            return False
        
        try:
            with open(self.credentials_file, 'r') as f:
                creds = json.load(f)
            
            # Check required fields
            required = ["node_id", "node_key", "expires_at"]
            if not all(field in creds for field in required):
                return False
            
            # Check expiry
            expires_at = creds["expires_at"]
            # Simple check - in production, parse ISO timestamp
            if "T" in expires_at:  # ISO format
                # For now, assume valid if file exists and has expiry
                # In production, properly parse and compare timestamps
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error reading credentials: {e}")
            return False
    
    def enroll_with_join_code(self) -> bool:
        """
        Enroll node using JOIN_CODE
        
        Returns:
            True if enrollment successful
        """
        if not self.join_code:
            logger.error("No JOIN_CODE provided in environment")
            return False
        
        logger.info(f"Enrolling with JOIN_CODE: {self.join_code[:4]}****")
        
        # Prepare enrollment request
        enroll_url = f"{self.main_server}/p2p/enroll"
        payload = {
            "join_code": self.join_code,
            "node_meta": self.node_meta
        }
        
        try:
            # Make enrollment request
            response = requests.post(
                enroll_url,
                json=payload,
                timeout=30,
                headers={
                    "User-Agent": "Blyan-Node-Bootstrap/1.0",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Save credentials
                credentials = {
                    "node_id": data["node_id"],
                    "node_key": data["node_key"],
                    "expires_at": data["expires_at"],
                    "enrolled_at": time.time(),
                    "main_server": self.main_server
                }
                
                with open(self.credentials_file, 'w') as f:
                    json.dump(credentials, f, indent=2)
                
                # Set restrictive permissions
                os.chmod(self.credentials_file, 0o600)
                
                logger.info(f"✓ Successfully enrolled as node: {data['node_id']}")
                logger.info(f"✓ Credentials saved to: {self.credentials_file}")
                
                # Save node configuration
                config = {
                    "node_id": data["node_id"],
                    "enrolled_at": credentials["enrolled_at"],
                    "metadata": self.node_meta
                }
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Clear JOIN_CODE from environment (security)
                if "JOIN_CODE" in os.environ:
                    del os.environ["JOIN_CODE"]
                
                return True
                
            else:
                error_msg = "Unknown error"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except:
                    error_msg = response.text
                
                logger.error(f"✗ Enrollment failed: {error_msg}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("✗ Enrollment timeout - check network connection")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to {self.main_server}")
            return False
        except Exception as e:
            logger.error(f"✗ Enrollment error: {e}")
            return False
    
    def load_credentials(self) -> Optional[Dict[str, str]]:
        """
        Load existing credentials
        
        Returns:
            Credentials dict or None
        """
        if not self.has_credentials():
            return None
        
        try:
            with open(self.credentials_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return None
    
    def test_credentials(self, credentials: Dict[str, str]) -> bool:
        """
        Test if credentials are valid by calling verify endpoint
        
        Args:
            credentials: Credentials dictionary
            
        Returns:
            True if credentials are valid
        """
        verify_url = f"{self.main_server}/p2p/node-status"
        
        try:
            response = requests.get(
                verify_url,
                headers={
                    "X-Node-ID": credentials["node_id"],
                    "X-Node-Key": credentials["node_key"],
                    "User-Agent": "Blyan-Node-Bootstrap/1.0"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✓ Credentials verified successfully")
                return True
            else:
                logger.warning("✗ Credentials verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error testing credentials: {e}")
            return False
    
    def bootstrap(self) -> Dict[str, str]:
        """
        Main bootstrap process
        
        Returns:
            Credentials dict or exits with error
        """
        logger.info("=== Node Bootstrap Starting ===")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Main server: {self.main_server}")
        
        # Check for existing credentials
        if self.has_credentials():
            logger.info("Found existing credentials")
            credentials = self.load_credentials()
            
            if credentials and self.test_credentials(credentials):
                logger.info(f"✓ Using existing credentials for node: {credentials['node_id']}")
                return credentials
            else:
                logger.warning("Existing credentials are invalid or expired")
                
                # If we have a JOIN_CODE, try to re-enroll
                if self.join_code:
                    logger.info("Attempting re-enrollment with JOIN_CODE")
                else:
                    logger.error("No JOIN_CODE provided for re-enrollment")
                    logger.error("Please provide a new JOIN_CODE and restart")
                    sys.exit(1)
        
        # No valid credentials, check for JOIN_CODE
        if not self.join_code:
            logger.error("=" * 50)
            logger.error("ERROR: No credentials found and no JOIN_CODE provided")
            logger.error("")
            logger.error("To enroll this node:")
            logger.error("1. Request a JOIN_CODE from https://blyan.com/contribute")
            logger.error("2. Run with: -e JOIN_CODE=YOUR_CODE_HERE")
            logger.error("=" * 50)
            sys.exit(1)
        
        # Attempt enrollment
        logger.info("No existing credentials, enrolling with JOIN_CODE...")
        
        if self.enroll_with_join_code():
            credentials = self.load_credentials()
            if credentials:
                logger.info("=== Bootstrap Complete ===")
                return credentials
            else:
                logger.error("Failed to load credentials after enrollment")
                sys.exit(1)
        else:
            logger.error("=== Bootstrap Failed ===")
            logger.error("Could not enroll node with provided JOIN_CODE")
            sys.exit(1)


def main():
    """Main entry point"""
    bootstrap = NodeBootstrap()
    credentials = bootstrap.bootstrap()
    
    # Export credentials as environment variables for the main process
    os.environ["NODE_ID"] = credentials["node_id"]
    os.environ["NODE_KEY"] = credentials["node_key"]
    
    logger.info(f"Node {credentials['node_id']} ready to start")
    
    # In Docker, this would typically exec into the main node process
    # For testing, just print the credentials
    print(json.dumps({
        "status": "ready",
        "node_id": credentials["node_id"],
        "has_key": bool(credentials["node_key"])
    }))


if __name__ == "__main__":
    main()