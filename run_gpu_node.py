#!/usr/bin/env python3
"""
Blyan GPU Node - Production Ready
Run this on GPU servers (RunPod, etc.) to serve as a compute node
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
PORT = int(os.environ.get('NODE_PORT', 8002))
MAIN_NODE_URL = os.environ.get('MAIN_NODE_URL', 'http://165.227.221.225:8000')

async def main():
    """Run GPU node server."""
    logger.info("=" * 60)
    logger.info("BLYAN GPU NODE - PRODUCTION")
    logger.info("=" * 60)
    
    # Check GPU availability
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
            logger.warning("No GPU detected - CPU mode")
    except ImportError:
        logger.warning("PyTorch not installed - CPU mode")
        import subprocess
        logger.info("Installing PyTorch...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
        import torch
    
    # Initialize model manager
    model_manager = None
    available_experts = []
    
    try:
        from backend.model.moe_infer import MoEModelManager
        model_manager = MoEModelManager(
            models_dir=Path("./models"),
            device="cuda" if gpu_available else "cpu"
        )
        available_experts = model_manager.list_available_experts()
        logger.info(f"Model manager ready. Experts: {available_experts if available_experts else 'None loaded'}")
    except Exception as e:
        logger.warning(f"Model manager not available: {e}")
    
    # Create HTTP server
    try:
        from aiohttp import web
    except ImportError:
        logger.info("Installing aiohttp...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp"])
        from aiohttp import web
    
    app = web.Application()
    node_id = f"gpu_node_{os.getpid()}"
    
    # Health check endpoint
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "node_id": node_id,
            "gpu": gpu_name,
            "gpu_available": gpu_available,
            "port": PORT,
            "model_ready": model_manager is not None,
            "experts_loaded": len(available_experts)
        })
    
    # Node information endpoint
    async def info(request):
        return web.json_response({
            "node_id": node_id,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
            "gpu_available": gpu_available,
            "cuda_version": torch.version.cuda if gpu_available else None,
            "pytorch_version": torch.__version__ if 'torch' in sys.modules else None,
            "port": PORT,
            "experts": available_experts,
            "main_node": MAIN_NODE_URL
        })
    
    # Inference endpoint
    async def inference(request):
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            max_length = data.get("max_length", 100)
            temperature = data.get("temperature", 0.7)
            
            if not prompt:
                return web.json_response(
                    {"error": "No prompt provided"},
                    status=400
                )
            
            # Use model manager if available
            if model_manager:
                try:
                    # Run inference in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    response_text = await loop.run_in_executor(
                        None,
                        lambda: model_manager.generate(
                            prompt,
                            max_length=max_length,
                            temperature=temperature
                        )
                    )
                    
                    return web.json_response({
                        "node_id": node_id,
                        "prompt": prompt,
                        "response": response_text,
                        "model": "blyan-moe",
                        "gpu_used": gpu_available,
                        "experts_used": available_experts[:2] if available_experts else [],
                        "port": PORT
                    })
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    return web.json_response({
                        "error": f"Inference failed: {str(e)}",
                        "node_id": node_id
                    }, status=500)
            
            # No model available - return informative message
            return web.json_response({
                "node_id": node_id,
                "prompt": prompt,
                "response": f"Node {node_id} received prompt but no model is loaded. GPU: {gpu_name}",
                "model": "none",
                "gpu_available": gpu_available,
                "port": PORT
            })
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            return web.json_response({
                "error": str(e)
            }, status=500)
    
    # Register routes
    app.router.add_get('/', health)
    app.router.add_get('/health', health)
    app.router.add_get('/info', info)
    app.router.add_post('/inference', inference)
    
    # Start server with automatic port finding
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = PORT
    for attempt in range(10):
        try:
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            break
        except OSError as e:
            if "address already in use" in str(e).lower():
                logger.warning(f"Port {port} in use, trying {port + 1}")
                port += 1
            else:
                raise
    else:
        logger.error("Could not find available port after 10 attempts")
        return
    
    logger.info(f"Node ID: {node_id}")
    logger.info(f"Server running on http://0.0.0.0:{port}")
    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info(f"  GET  http://localhost:{port}/health")
    logger.info(f"  GET  http://localhost:{port}/info")
    logger.info(f"  POST http://localhost:{port}/inference")
    logger.info("=" * 60)
    
    # Try to register with main node
    try:
        import httpx
    except ImportError:
        logger.info("Installing httpx for registration...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "httpx"])
        import httpx
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get public IP
            try:
                resp = await client.get("https://api.ipify.org")
                public_ip = resp.text.strip() if resp.status_code == 200 else "unknown"
            except:
                public_ip = "unknown"
            
            logger.info(f"Public IP: {public_ip}")
            
            # Register with main node
            register_data = {
                "node_id": node_id,
                "host": public_ip,
                "port": port,
                "node_type": "gpu",
                "gpu_info": {
                    "name": gpu_name,
                    "memory_gb": gpu_memory,
                    "available": gpu_available
                },
                "available_experts": available_experts
            }
            
            resp = await client.post(
                f"{MAIN_NODE_URL}/p2p/register",
                json=register_data
            )
            
            if resp.status_code == 200:
                logger.info(f"Successfully registered with main node")
            else:
                logger.info(f"Registration response: {resp.status_code}")
                
    except Exception as e:
        logger.info(f"Running standalone (no registration): {e}")
    
    # Keep running
    logger.info("Node is ready for inference requests")
    try:
        while True:
            await asyncio.sleep(60)
            logger.debug(f"Node {node_id} alive on port {port}")
    except asyncio.CancelledError:
        logger.info("Shutting down...")

if __name__ == "__main__":
    try:
        # Load .env if exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Run the node
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)