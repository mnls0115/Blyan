#!/usr/bin/env python3
"""
Simple GPU Node Setup Script
Run this directly on RunPod to set up everything automatically
"""

import os
import sys
import secrets
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

def generate_api_key():
    """Generate a simple API key for this node."""
    # Generate a unique key
    api_key = f"blyan_node_{secrets.token_hex(32)}"
    
    # Save to .env file
    env_file = Path(".env")
    with open(env_file, 'w') as f:
        f.write(f"BLYAN_API_KEY={api_key}\n")
        f.write(f"MAIN_NODE_URL=http://165.227.221.225:8000\n")
        f.write(f"NODE_TYPE=gpu\n")
        f.write(f"BLOCKCHAIN_ONLY=false\n")
    
    print(f"✅ API key generated and saved to .env")
    return api_key

def setup_node():
    """Set up the GPU node with minimal configuration."""
    print("=" * 60)
    print("🚀 BLYAN GPU NODE - QUICK SETUP")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Error: Not in Blyan directory. Please run from /workspace/blyan")
        sys.exit(1)
    
    # Generate API key
    print("\n📝 Generating API key...")
    api_key = generate_api_key()
    
    # Create a simple run script
    print("\n📝 Creating run script...")
    run_script = """#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up path
sys.path.insert(0, str(Path(__file__).parent))

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 60)
    logger.info("BLYAN GPU NODE - STARTING")
    logger.info("=" * 60)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠️  No GPU detected - CPU mode")
    except:
        logger.info("⚠️  PyTorch not installed - CPU mode")
    
    # Try to import and run the node
    try:
        from backend.p2p.distributed_inference import ExpertNode
        from backend.model.moe_infer import MoEModelManager
        
        # Create simple node
        node = ExpertNode(
            node_id=f"gpu_node_{os.getpid()}",
            host="0.0.0.0",
            port=8001,
            available_experts=[]
        )
        
        logger.info(f"📡 Node ID: {node.node_id}")
        logger.info(f"🌐 Serving on port 8001")
        logger.info("=" * 60)
        
        # Start server
        await node.start_server()
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Running in standalone mode...")
        
        # Fallback to simple HTTP server
        from aiohttp import web
        
        app = web.Application()
        
        async def health(request):
            return web.json_response({"status": "healthy", "node": "gpu_node"})
        
        app.router.add_get('/health', health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8001)
        await site.start()
        
        logger.info("Running standalone HTTP server on port 8001")
        while True:
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
"""
    
    with open("run_node.py", "w") as f:
        f.write(run_script)
    os.chmod("run_node.py", 0o755)
    print("✅ Run script created: run_node.py")
    
    # Install minimal dependencies
    print("\n📦 Installing dependencies...")
    os.system("pip install -q python-dotenv aiohttp 2>/dev/null")
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nTo start the node, run:")
    print("  python run_node.py")
    print("\nOr run in background:")
    print("  nohup python run_node.py > node.log 2>&1 &")
    print("=" * 60)

if __name__ == "__main__":
    setup_node()