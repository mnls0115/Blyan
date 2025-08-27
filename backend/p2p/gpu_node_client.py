#!/usr/bin/env python3
"""GPU Node Client - Production Implementation
================================================================
Runs on GPU nodes (e.g., Vast.ai) to register and serve inference requests.
"""

import os
import sys
import asyncio
import logging
import time
import json
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
import httpx
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class GPUNodeClient:
    """Client for GPU nodes to register with service nodes."""
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        service_url: str = "http://165.227.221.225:8000",
        api_key: Optional[str] = None,
        gpu_port: int = 8001
    ):
        # Generate unique node ID if not provided
        self.node_id = node_id or f"gpu-{secrets.token_hex(8)}"
        self.service_url = service_url.rstrip('/')
        self.api_key = api_key or os.getenv('BLYAN_API_KEY', secrets.token_hex(32))
        self.gpu_port = gpu_port
        
        # Get node's public IP (for Vast.ai or cloud providers)
        self.public_ip = self._get_public_ip()
        self.api_url = f"http://{self.public_ip}:{self.gpu_port}"
        
        # GPU capabilities
        self.capabilities = self._detect_capabilities()
        
        # Registration state
        self.registered = False
        self.heartbeat_task = None
        self.last_heartbeat = 0
        
        logger.info(f"🚀 GPU Node Client initialized")
        logger.info(f"   Node ID: {self.node_id}")
        logger.info(f"   Service URL: {self.service_url}")
        logger.info(f"   GPU API URL: {self.api_url}")
        logger.info(f"   Capabilities: {json.dumps(self.capabilities, indent=2)}")
    
    def _get_public_ip(self) -> str:
        """Get public IP address of this node."""
        # Try environment variable first (Vast.ai sets this)
        public_ip = os.getenv('PUBLIC_IP')
        if public_ip:
            return public_ip
        
        # Try to get from external service
        try:
            import subprocess
            result = subprocess.run(
                ['curl', '-s', 'https://api.ipify.org'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback to localhost
        logger.warning("Could not detect public IP, using localhost")
        return "localhost"
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect GPU capabilities."""
        capabilities = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_memory_gb": 0,
            "compute_capability": None,
            "gpu_name": "Unknown",
            "supports_bf16": False,
            "layers": []  # Will be assigned by service node
        }
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            capabilities["gpu_name"] = torch.cuda.get_device_name(device)
            
            # Get memory in GB
            mem_bytes = torch.cuda.get_device_properties(device).total_memory
            capabilities["gpu_memory_gb"] = round(mem_bytes / (1024**3), 1)
            
            # Get compute capability
            major, minor = torch.cuda.get_device_capability(device)
            capabilities["compute_capability"] = f"{major}.{minor}"
            
            # Check BF16 support (Ampere and newer)
            capabilities["supports_bf16"] = major >= 8
        
        return capabilities
    
    async def register(self, max_retries: int = 5) -> bool:
        """Register this GPU node with the service node."""
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Attempting registration (attempt {attempt + 1}/{max_retries})...")
                
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{self.service_url}/gpu/register",
                        json={
                            "node_id": self.node_id,
                            "api_url": self.api_url,
                            "capabilities": self.capabilities
                        },
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            self.registered = True
                            layers = result.get("layers_assigned", [])
                            
                            logger.info(f"✅ Successfully registered with service node!")
                            logger.info(f"   Assigned layers: {layers}")
                            
                            # Start heartbeat
                            if not self.heartbeat_task:
                                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                            
                            return True
                        else:
                            logger.error(f"Registration failed: {result.get('message')}")
                    else:
                        logger.error(f"Registration failed with status {response.status_code}: {response.text}")
                        
            except httpx.ConnectError:
                logger.warning(f"Cannot connect to service node at {self.service_url}")
            except Exception as e:
                logger.error(f"Registration error: {e}")
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"❌ Failed to register after {max_retries} attempts")
        return False
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to service node."""
        while self.registered:
            try:
                # Send heartbeat every 15 seconds
                await asyncio.sleep(15)
                
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(
                        f"{self.service_url}/gpu/heartbeat",
                        json={"node_id": self.node_id},
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    )
                    
                    if response.status_code == 200:
                        self.last_heartbeat = time.time()
                        logger.debug(f"💓 Heartbeat sent successfully")
                    else:
                        logger.warning(f"Heartbeat failed: {response.status_code}")
                        
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                # Try to re-register if heartbeat fails multiple times
                if time.time() - self.last_heartbeat > 60:
                    logger.warning("Lost connection, attempting re-registration...")
                    self.registered = False
                    await self.register()
    
    async def start_inference_server(self):
        """Start the local inference server."""
        # This would start the actual inference server
        # For now, we'll import and run the GPU node server
        from api.server import app
        import uvicorn
        
        logger.info(f"🚀 Starting inference server on port {self.gpu_port}...")
        
        # Configure the server for GPU node mode
        os.environ['IS_GPU_NODE'] = 'true'
        os.environ['NODE_ID'] = self.node_id
        os.environ['BLYAN_API_KEY'] = self.api_key
        
        # Run server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.gpu_port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run(self):
        """Main run loop."""
        # Register with service node
        if not await self.register():
            logger.error("Failed to register, exiting...")
            return
        
        # Start inference server
        await self.start_inference_server()


async def main():
    """Main entry point."""
    # Configuration from environment
    service_url = os.getenv('SERVICE_NODE_URL', 'http://165.227.221.225:8000')
    node_id = os.getenv('NODE_ID')
    api_key = os.getenv('BLYAN_API_KEY')
    gpu_port = int(os.getenv('GPU_PORT', '8001'))
    
    # Create and run client
    client = GPUNodeClient(
        node_id=node_id,
        service_url=service_url,
        api_key=api_key,
        gpu_port=gpu_port
    )
    
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🚫 Shutting down GPU node client...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)