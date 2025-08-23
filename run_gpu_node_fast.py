#!/usr/bin/env python3
"""
Fast GPU Node Runner - Minimal startup time
Skips validation and uses cached state for quick startup
"""

import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, List
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
PORT = int(os.environ.get('NODE_PORT', '8000'))
MAIN_NODE_URL = os.environ.get('MAIN_NODE_URL', 'https://blyan.com/api')
DATA_DIR = Path(os.environ.get('BLYAN_DATA_DIR', './data'))
PUBLIC_HOST = os.environ.get('PUBLIC_HOST', '')
PUBLIC_PORT = int(os.environ.get('PUBLIC_PORT', str(PORT)))

# Model configuration
try:
    from config.model_profile import LAYERS, MOE, MODEL_ID
    NUM_LAYERS = LAYERS["num_layers"]
    NUM_EXPERTS = MOE["num_experts"]
    TOTAL_EXPERTS = NUM_LAYERS * NUM_EXPERTS
except ImportError:
    NUM_LAYERS = 48
    NUM_EXPERTS = 128
    TOTAL_EXPERTS = 6144
    MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

class FastGPUNode:
    """Minimal GPU node for fast startup."""
    
    def __init__(self):
        self.node_id = f"gpu_node_fast_{os.getpid()}"
        self.port = PORT
        self.blockchain_ready = False
        self.experts_available = []
        
    async def quick_check_blockchain(self):
        """Quick check if blockchain exists with expected experts."""
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        chain_b_path = DATA_DIR / 'chain_B_storage.pkl'
        logger.info(f"📂 Looking for blockchain at: {chain_b_path}")
        
        if not chain_b_path.exists():
            logger.warning(f"❌ No blockchain found at {DATA_DIR}")
            # Still register with some test experts
            self.blockchain_ready = False
            self.experts_available = []
            for layer_idx in range(min(4, NUM_LAYERS)):  # Just first 4 layers
                self.experts_available.extend([f"layer{layer_idx}.expert{i}" for i in range(2)])
            logger.info(f"⚠️ Running without blockchain, using {len(self.experts_available)} test experts")
            return False
            
        # Quick check using index file
        index_path = DATA_DIR / 'chain_B_index.json'
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                    block_count = index_data.get('block_count', 0)
                    logger.info(f"✅ Blockchain has {block_count} blocks (from index)")
                    
                    if block_count >= TOTAL_EXPERTS:
                        self.blockchain_ready = True
                        # Generate representative expert list (2 per layer) for registration
                        self.experts_available = []
                        for layer_idx in range(NUM_LAYERS):
                            self.experts_available.extend([f"layer{layer_idx}.expert{i}" for i in range(2)])
                        logger.info(f"✅ All {TOTAL_EXPERTS} experts available (registering {len(self.experts_available)} representatives)")
                        return True
            except:
                pass
        
        # Fallback: Just check if file exists and has reasonable size
        file_size = chain_b_path.stat().st_size
        if file_size > 1000000:  # At least 1MB
            logger.info(f"✅ Blockchain exists ({file_size / 1024 / 1024:.1f} MB)")
            self.blockchain_ready = True
            # Generate representative expert list (2 per layer) for registration
            self.experts_available = []
            for layer_idx in range(NUM_LAYERS):
                self.experts_available.extend([f"layer{layer_idx}.expert{i}" for i in range(2)])
            return True
            
        return False
    
    async def start_server(self):
        """Start the API server."""
        from aiohttp import web
        import aiohttp_cors
        
        app = web.Application()
        
        # Add CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Health endpoint
        async def health(request):
            return web.json_response({
                "status": "healthy",
                "node_id": self.node_id,
                "blockchain_ready": self.blockchain_ready,
                "experts_available": len(self.experts_available) > 0,
                "gpu_available": True,
                "model": MODEL_ID
            })
        
        # Chat endpoint
        async def chat(request):
            if not self.blockchain_ready:
                return web.json_response({
                    "error": "Blockchain not ready. Please wait for sync."
                }, status=503)
                
            data = await request.json()
            prompt = data.get("prompt", "")
            
            # For fast node, just return a response indicating readiness
            return web.json_response({
                "response": f"GPU node ready. Blockchain has {TOTAL_EXPERTS} experts available.",
                "node_id": self.node_id,
                "experts_used": self.experts_available[:5]  # Sample
            })
        
        # Expert info endpoint
        async def expert_info(request):
            return web.json_response({
                "node_id": self.node_id,
                "experts_available": self.experts_available,
                "total_experts": TOTAL_EXPERTS if self.blockchain_ready else 0,
                "model": MODEL_ID
            })
        
        # Add routes
        app.router.add_get('/health', health)
        app.router.add_post('/chat', chat)
        app.router.add_get('/experts', expert_info)
        
        # Add CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"🌐 Server running on http://0.0.0.0:{self.port}")
        
    async def register_with_main(self):
        """Register with main node."""
        try:
            import httpx
            
            # Determine public endpoint
            if PUBLIC_HOST:
                if PUBLIC_HOST.startswith('http'):
                    endpoint = PUBLIC_HOST
                else:
                    endpoint = f"https://{PUBLIC_HOST}" if 'proxy.runpod.net' in PUBLIC_HOST else f"http://{PUBLIC_HOST}:{PUBLIC_PORT}"
            else:
                # Auto-detect
                async with httpx.AsyncClient() as client:
                    resp = await client.get("https://api.ipify.org")
                    public_ip = resp.text.strip()
                    endpoint = f"http://{public_ip}:{PUBLIC_PORT}"
            
            logger.info(f"📝 Registering with main node...")
            logger.info(f"   Endpoint: {endpoint}")
            logger.info(f"   Sending {len(self.experts_available)} representative experts (for {TOTAL_EXPERTS} total)")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                data = {
                    "node_id": self.node_id,
                    "host": endpoint,
                    "port": 443 if 'https://' in endpoint else PUBLIC_PORT,
                    "available_experts": self.experts_available,
                    "node_type": "gpu",
                    "model_ready": self.blockchain_ready,
                    "fast_mode": True
                }
                
                resp = await client.post(f"{MAIN_NODE_URL}/p2p/register", json=data)
                if resp.status_code == 200:
                    logger.info("✅ Registered with main node")
                else:
                    logger.warning(f"Registration failed: {resp.status_code}")
                    
        except Exception as e:
            logger.warning(f"Could not register: {e}")
    
    async def run(self):
        """Main run loop."""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("⚡ FAST GPU NODE STARTUP")
        logger.info("=" * 60)
        
        # Step 1: Quick blockchain check
        logger.info("📦 Checking blockchain...")
        await self.quick_check_blockchain()
        
        # Step 2: Start server
        logger.info("🚀 Starting server...")
        await self.start_server()
        
        # Step 3: Register with main node
        await self.register_with_main()
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ STARTUP COMPLETE in {elapsed:.1f} seconds")
        logger.info(f"🌐 Server: http://0.0.0.0:{self.port}")
        logger.info(f"📁 Data directory: {DATA_DIR}")
        logger.info(f"🤖 Blockchain ready: {self.blockchain_ready}")
        logger.info(f"📦 Experts registered: {len(self.experts_available)}")
        logger.info("=" * 60)
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Shutting down...")

async def main():
    """Main entry point."""
    node = FastGPUNode()
    await node.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")