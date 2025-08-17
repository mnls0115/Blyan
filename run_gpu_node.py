#!/usr/bin/env python3
"""
Blyan GPU Node - Simple Working Version
Run this on GPU servers (RunPod, etc.) to serve as a compute node
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
PORT = int(os.environ.get('NODE_PORT', 8002))
MAIN_NODE_URL = os.environ.get('MAIN_NODE_URL', 'http://165.227.221.225:8000')

async def main():
    """Run GPU node server."""
    logger.info("=" * 60)
    logger.info("BLYAN GPU NODE - STARTING")
    logger.info("=" * 60)
    
    # Check GPU
    gpu_available = False
    gpu_name = "None"
    gpu_memory = 0
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Memory: {gpu_memory:.2f} GB")
            logger.info(f"CUDA: {torch.version.cuda}")
        else:
            logger.info("No GPU detected")
    except ImportError:
        logger.info("PyTorch not installed")
    
    # Try to import aiohttp, install if needed
    try:
        from aiohttp import web
    except ImportError:
        logger.info("Installing aiohttp...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp"])
        from aiohttp import web
    
    # Create web app
    app = web.Application()
    node_id = f"gpu_node_{os.getpid()}"
    
    # Health endpoint
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "node_id": node_id,
            "gpu": gpu_name,
            "gpu_available": gpu_available,
            "port": PORT
        })
    
    # Info endpoint
    async def info(request):
        return web.json_response({
            "node_id": node_id,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
            "gpu_available": gpu_available,
            "port": PORT
        })
    
    # Inference endpoint
    async def inference(request):
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            
            # For now, just echo back with GPU info
            return web.json_response({
                "node_id": node_id,
                "prompt": prompt,
                "response": f"GPU Node {node_id} received: {prompt}",
                "gpu": gpu_name,
                "gpu_used": gpu_available,
                "port": PORT
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    # Register routes
    app.router.add_get('/', health)
    app.router.add_get('/health', health)
    app.router.add_get('/info', info)
    app.router.add_post('/inference', inference)
    
    # Start server - try multiple ports if needed
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = PORT
    site = None
    for attempt in range(10):
        try:
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            logger.info(f"Server started on port {port}")
            break
        except OSError as e:
            if "address already in use" in str(e).lower():
                logger.warning(f"Port {port} in use, trying {port + 1}")
                port += 1
            else:
                raise
    else:
        logger.error("Could not find available port")
        return
    
    logger.info(f"Node ID: {node_id}")
    logger.info(f"Server running on http://0.0.0.0:{port}")
    logger.info("=" * 60)
    logger.info("Endpoints:")
    logger.info(f"  GET  http://localhost:{port}/health")
    logger.info(f"  GET  http://localhost:{port}/info")
    logger.info(f"  POST http://localhost:{port}/inference")
    logger.info("=" * 60)
    
    # Try to register with main node
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get public IP
            try:
                resp = await client.get("https://api.ipify.org")
                public_ip = resp.text.strip() if resp.status_code == 200 else "unknown"
                logger.info(f"Public IP: {public_ip}")
            except:
                public_ip = "unknown"
            
            # Try to register
            register_data = {
                "node_id": node_id,
                "host": public_ip,
                "port": port,
                "node_type": "gpu",
                "gpu_info": {
                    "name": gpu_name,
                    "memory_gb": gpu_memory,
                    "available": gpu_available
                }
            }
            
            resp = await client.post(f"{MAIN_NODE_URL}/p2p/register", json=register_data)
            if resp.status_code == 200:
                logger.info("Registered with main node")
            else:
                logger.info(f"Registration status: {resp.status_code}")
    except ImportError:
        logger.info("httpx not installed - skipping registration")
    except Exception as e:
        logger.info(f"Could not register: {e}")
    
    # Keep running
    logger.info("Node ready for requests")
    try:
        while True:
            await asyncio.sleep(60)
            logger.debug(f"Node {node_id} running on port {port}")
    except asyncio.CancelledError:
        logger.info("Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)