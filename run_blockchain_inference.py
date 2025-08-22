#!/usr/bin/env python3
"""
Test blockchain inference directly
"""
import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_blockchain_inference():
    """Test if we can load and use blockchain experts."""
    
    logger.info("="*60)
    logger.info("Testing Blockchain Inference")
    logger.info("="*60)
    
    # 1. Check GPU
    if torch.cuda.is_available():
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️  No GPU (will be slow)")
    
    # 2. Load chains
    from backend.core.chain import Chain
    
    data_dir = Path("./data")
    logger.info(f"\nLoading chains from {data_dir}")
    
    chain_A = Chain(data_dir, "A")
    chain_B = Chain(data_dir, "B")
    
    logger.info(f"Chain A blocks: {len(chain_A._hash_index)}")
    logger.info(f"Chain B blocks: {len(chain_B._hash_index)}")
    
    # 3. Check for experts
    logger.info("\nChecking for expert blocks...")
    expert_count = 0
    expert_list = []
    
    for block_hash in list(chain_B._hash_index.keys())[:500]:  # Check first 500
        block = chain_B.get_block_by_hash(block_hash)
        if block and hasattr(block, 'metadata') and block.metadata:
            if block.metadata.get('block_type') == 'expert':
                expert_count += 1
                expert_name = block.metadata.get('expert_name', 'unknown')
                if len(expert_list) < 10:  # Collect first 10
                    expert_list.append(expert_name)
    
    logger.info(f"Found {expert_count} expert blocks")
    if expert_list:
        logger.info(f"Sample experts: {expert_list}")
    
    # 4. Try to load MoEModelManager
    try:
        from backend.model.moe_infer import MoEModelManager
        
        logger.info("\nInitializing MoEModelManager...")
        manager = MoEModelManager(chain_A, chain_B)
        
        # Check if we can generate
        if hasattr(manager, 'generate'):
            logger.info("✅ MoEModelManager has generate method")
            
            # Try a simple generation
            try:
                result = manager.generate(
                    prompt="Hello",
                    max_new_tokens=10,
                    use_moe=True
                )
                logger.info(f"✅ Generation successful: {result}")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
        else:
            logger.info("❌ MoEModelManager missing generate method")
            
    except ImportError as e:
        logger.error(f"Cannot import MoEModelManager: {e}")
    except Exception as e:
        logger.error(f"Error initializing MoEModelManager: {e}")
    
    # 5. Alternative: Check BlockchainFirstLoader
    try:
        from backend.model.blockchain_first_loader import BlockchainFirstModelLoader
        
        logger.info("\nTrying BlockchainFirstModelLoader...")
        loader = BlockchainFirstModelLoader(
            chains={'A': chain_A, 'B': chain_B},
            model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
        )
        logger.info("✅ BlockchainFirstModelLoader initialized")
        
    except ImportError:
        logger.info("BlockchainFirstModelLoader not available")
    except Exception as e:
        logger.error(f"Error with BlockchainFirstModelLoader: {e}")

if __name__ == "__main__":
    test_blockchain_inference()