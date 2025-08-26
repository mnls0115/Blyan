#!/usr/bin/env python3
"""
Fix GPU inference on Vast.ai server
Enable local inference mode for the already-loaded model
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def enable_local_inference():
    """Enable local inference mode on the GPU server."""
    
    # Set environment variables for local inference
    os.environ['ENABLE_LOCAL_INFERENCE'] = 'true'
    os.environ['USE_BLOCKCHAIN'] = 'true'
    os.environ['SKIP_GPU_REGISTRATION'] = 'false'
    
    logger.info("🔧 Configuring local inference mode...")
    
    # Import after setting env vars
    from backend.model.manager import UnifiedModelManager
    from backend.core.chain import Chain
    
    # Initialize with blockchain weights already uploaded
    data_dir = Path('./data')
    
    # Create model manager that will use blockchain weights
    model_manager = UnifiedModelManager(
        root_dir=data_dir,
        model_name="Qwen/Qwen3-8B",
        device="cuda",
        use_blockchain=True,
        use_gpu_direct=True
    )
    
    # Load the model from blockchain
    logger.info("📦 Loading model from blockchain...")
    try:
        model_manager.load_model()
        logger.info("✅ Model loaded successfully from blockchain")
        
        # Test inference
        logger.info("🧪 Testing inference...")
        response = model_manager.generate(
            prompt="Hello, this is a test.",
            max_new_tokens=20
        )
        logger.info(f"✅ Test response: {response}")
        
        # Register as local GPU node
        logger.info("📝 Registering as local GPU node...")
        from backend.p2p.distributed_inference import GPUNode, NodeRegistry
        
        registry = NodeRegistry()
        node = GPUNode(
            node_id="local-gpu-vast",
            host="127.0.0.1",
            port=8000,
            available_layers=list(range(36)),  # All 36 layers
            vram_gb=24.0,  # Adjust based on your GPU
            compute_capability=8.0
        )
        registry.register_node(node)
        logger.info("✅ Registered as local GPU node")
        
        # Save state
        state_file = data_dir / "gpu_node_enabled.json"
        import json
        with open(state_file, 'w') as f:
            json.dump({
                "enabled": True,
                "node_id": "local-gpu-vast",
                "timestamp": str(Path(__file__).stat().st_mtime)
            }, f)
        
        logger.info("✅ Local inference mode enabled successfully!")
        logger.info("🚀 The server should now handle chat requests locally")
        
    except Exception as e:
        logger.error(f"❌ Failed to enable local inference: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(enable_local_inference())