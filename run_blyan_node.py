#!/usr/bin/env python3
"""Blyan GPU Node for distributed inference - PRODUCTION.

This script runs a GPU node that connects to the Blyan main node for distributed inference.
It requires a valid API key and GPU hardware for production use.

Usage:
    python run_blyan_node.py
    
Environment Variables:
    BLYAN_API_KEY: Required API key for authentication
    MAIN_NODE_URL: URL of the main node (default: http://165.227.221.225:8000)
    RUNPOD_POD_ID: RunPod pod identifier (optional)
    RUNPOD_DC_ID: RunPod datacenter ID (optional)
    RUNPOD_PUBLIC_IP: Public IP address (optional, will auto-detect if not set)
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.p2p.distributed_inference import ExpertNode
from backend.model.moe_infer import MoEModelManager
import torch

# Production logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run GPU node for distributed inference in PRODUCTION mode."""
    
    logger.info("=" * 60)
    logger.info("BLYAN GPU Node - PRODUCTION MODE")
    logger.info("=" * 60)
    
    # PRODUCTION: Require API key
    api_key = os.environ.get("BLYAN_API_KEY")
    if not api_key:
        logger.error("FATAL: BLYAN_API_KEY not set - this is required for production")
        logger.error("Please set BLYAN_API_KEY in .env file or environment")
        logger.error("")
        logger.error("To get an API key:")
        logger.error("1. Run: ./generate_gpu_node_key.sh")
        logger.error("2. Or contact the Blyan network administrator")
        sys.exit(1)
    
    logger.info(f"API Key configured: {api_key[:20]}...")
    
    # Check GPU availability - REQUIRED for production
    if not torch.cuda.is_available():
        logger.error("FATAL: No GPU detected - GPU is required for production nodes")
        logger.error("This node must run on hardware with NVIDIA GPU support")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU detected: {gpu_name}")
    logger.info(f"GPU memory: {gpu_memory:.2f} GB")
    
    # Initialize MoE model manager
    model_manager = MoEModelManager(
        models_dir=Path("./models"),
        device="cuda"  # Always CUDA in production
    )
    
    # Get available experts
    available_experts = model_manager.list_available_experts()
    if not available_experts:
        logger.warning("No experts found locally - will download from main node as needed")
        available_experts = []
    else:
        logger.info(f"Available experts: {available_experts}")
    
    # Get node metadata
    node_suffix = os.environ.get('RUNPOD_POD_ID', f'node_{int(time.time())}')
    datacenter = os.environ.get('RUNPOD_DC_ID', 'unknown')
    
    # Initialize expert node with production config
    node = ExpertNode(
        node_id=f"blyan_gpu_{node_suffix}",
        host="0.0.0.0",
        port=8001,
        available_experts=available_experts
    )
    
    # Register with main node using API key
    main_node_url = os.environ.get("MAIN_NODE_URL", "http://165.227.221.225:8000")
    
    logger.info(f"Registering with main node: {main_node_url}")
    
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get public IP - CRITICAL for production
            public_ip = None
            
            # Try environment variable first
            public_ip = os.environ.get("RUNPOD_PUBLIC_IP") or os.environ.get("PUBLIC_IP")
            
            if not public_ip:
                # Try RunPod metadata endpoint
                try:
                    response = await client.get("http://169.254.169.254/latest/meta-data/public-ipv4", timeout=2)
                    if response.status_code == 200:
                        public_ip = response.text.strip()
                        logger.info(f"Detected public IP from metadata: {public_ip}")
                except:
                    pass
            
            if not public_ip:
                # Try external service as fallback
                try:
                    response = await client.get("https://api.ipify.org", timeout=5)
                    if response.status_code == 200:
                        public_ip = response.text.strip()
                        logger.info(f"Detected public IP from ipify: {public_ip}")
                except:
                    pass
            
            if not public_ip:
                logger.error("FATAL: Could not determine public IP address")
                logger.error("Please set RUNPOD_PUBLIC_IP or PUBLIC_IP environment variable")
                sys.exit(1)
            
            # PRODUCTION registration payload
            registration_data = {
                "node_id": node.node_id,
                "host": public_ip,
                "port": 8001,
                "available_experts": available_experts,
                "node_type": "gpu",
                "gpu_info": {
                    "name": gpu_name,
                    "memory_gb": gpu_memory,
                    "cuda_version": torch.version.cuda,
                    "pytorch_version": torch.__version__
                },
                "datacenter": datacenter,
                "production": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Registering node: {node.node_id} at {public_ip}:8001")
            
            response = await client.post(
                f"{main_node_url}/p2p/register",
                headers=headers,
                json=registration_data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Successfully registered with main node")
                logger.info(f"Registration ID: {result.get('registration_id', 'N/A')}")
                logger.info(f"Assigned experts: {result.get('assigned_experts', available_experts)}")
            else:
                logger.error(f"❌ Registration failed: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                sys.exit(1)
                
    except httpx.ConnectError as e:
        logger.error(f"❌ Cannot connect to main node at {main_node_url}")
        logger.error(f"Error: {e}")
        logger.error("Please check:")
        logger.error("1. Main node is running")
        logger.error("2. Network connectivity")
        logger.error("3. Firewall settings")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during registration: {e}")
        sys.exit(1)
    
    # Start heartbeat task for production monitoring
    async def heartbeat():
        """Send periodic heartbeat to main node."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    await asyncio.sleep(30)  # Heartbeat every 30 seconds
                    
                    # Get current GPU stats
                    gpu_util = 0
                    gpu_temp = 0
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        pass
                    
                    heartbeat_data = {
                        "node_id": node.node_id,
                        "status": "active",
                        "gpu_utilization": gpu_util,
                        "gpu_temperature": gpu_temp,
                        "available_experts": node.available_experts,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    response = await client.post(
                        f"{main_node_url}/p2p/heartbeat",
                        headers=headers,
                        json=heartbeat_data
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Heartbeat failed: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
    
    # Start heartbeat in background
    asyncio.create_task(heartbeat())
    logger.info("✅ Heartbeat monitor started")
    
    # Start serving
    logger.info(f"🚀 Starting Blyan GPU node server on port 8001...")
    logger.info("=" * 60)
    logger.info("BLYAN GPU NODE IS READY FOR PRODUCTION INFERENCE")
    logger.info("=" * 60)
    
    await node.start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Blyan GPU node...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)