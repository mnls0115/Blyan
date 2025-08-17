#!/usr/bin/env python3
"""
Standalone Blyan GPU Node for RunPod
This can run independently without P2P registration
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import torch and check GPU
try:
    import torch
    HAS_TORCH = True
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_GPU = False
    logger.warning("PyTorch not installed - running in CPU mode")

# Simple HTTP server for health checks and inference
from aiohttp import web
import aiohttp

class BilyanGPUNode:
    """Standalone GPU node for Blyan network."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.node_id = f"blyan_runpod_{os.getpid()}"
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.app = web.Application()
        self.setup_routes()
        
        # GPU info
        self.gpu_info = self.get_gpu_info()
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available."""
        info = {
            "has_gpu": HAS_GPU,
            "has_torch": HAS_TORCH,
            "node_id": self.node_id
        }
        
        if HAS_GPU:
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__
            })
        
        return info
    
    def setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/info', self.node_info)
        self.app.router.add_post('/inference', self.inference)
        self.app.router.add_get('/', self.index)
    
    async def index(self, request):
        """Root endpoint."""
        return web.json_response({
            "service": "Blyan GPU Node",
            "node_id": self.node_id,
            "status": "running",
            "endpoints": [
                "/health - Health check",
                "/info - Node information",
                "/inference - Run inference (POST)"
            ]
        })
    
    async def health_check(self, request):
        """Health check endpoint."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return web.json_response({
            "status": "healthy",
            "node_id": self.node_id,
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "has_gpu": HAS_GPU
        })
    
    async def node_info(self, request):
        """Get node information."""
        return web.json_response(self.gpu_info)
    
    async def inference(self, request):
        """Handle inference requests."""
        self.request_count += 1
        
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            
            # For now, return mock response
            # In production, this would load the model and run actual inference
            response = {
                "node_id": self.node_id,
                "prompt": prompt,
                "response": f"Mock response from {self.node_id}",
                "gpu_used": HAS_GPU,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if HAS_GPU:
                # Add GPU stats
                response["gpu_stats"] = {
                    "memory_allocated": torch.cuda.memory_allocated() / 1e9,
                    "memory_reserved": torch.cuda.memory_reserved() / 1e9
                }
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def register_with_main_node(self):
        """Try to register with main node."""
        main_node_url = os.environ.get("MAIN_NODE_URL", "http://165.227.221.225:8000")
        
        try:
            # Get public IP
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.ipify.org") as resp:
                    public_ip = await resp.text()
                    logger.info(f"Public IP: {public_ip}")
                
                # Try to register
                registration_data = {
                    "node_id": self.node_id,
                    "host": public_ip,
                    "port": self.port,
                    "node_type": "gpu",
                    "gpu_info": self.gpu_info
                }
                
                # First check if P2P endpoint exists
                async with session.get(f"{main_node_url}/health") as resp:
                    if resp.status == 200:
                        logger.info(f"Main node is healthy at {main_node_url}")
                
                # Try registration (may fail if P2P not initialized)
                async with session.post(
                    f"{main_node_url}/p2p/register",
                    json=registration_data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"✅ Registered with main node: {result}")
                    else:
                        text = await resp.text()
                        logger.warning(f"Registration failed ({resp.status}): {text}")
                        
        except Exception as e:
            logger.warning(f"Could not register with main node: {e}")
            logger.info("Running in standalone mode")
    
    async def start(self):
        """Start the node."""
        logger.info("=" * 60)
        logger.info("BLYAN GPU NODE - RUNPOD STANDALONE")
        logger.info("=" * 60)
        logger.info(f"Node ID: {self.node_id}")
        logger.info(f"GPU Status: {'✅ Available' if HAS_GPU else '❌ Not Available'}")
        
        if HAS_GPU:
            logger.info(f"GPU: {self.gpu_info['gpu_name']}")
            logger.info(f"Memory: {self.gpu_info['gpu_memory_gb']:.2f} GB")
            logger.info(f"CUDA: {self.gpu_info['cuda_version']}")
        
        # Try to register with main node
        await self.register_with_main_node()
        
        # Start web server
        logger.info(f"🚀 Starting HTTP server on port {self.port}...")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info("=" * 60)
        logger.info(f"NODE READY AT http://0.0.0.0:{self.port}")
        logger.info("=" * 60)
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            logger.info(f"Status: Running | Requests: {self.request_count}")


async def main():
    """Main entry point."""
    node = BilyanGPUNode(port=8001)
    await node.start()


if __name__ == "__main__":
    try:
        # Install aiohttp if not present
        import aiohttp
    except ImportError:
        logger.info("Installing aiohttp...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp"])
        import aiohttp
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)