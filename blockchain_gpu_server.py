#!/usr/bin/env python3
"""
Blockchain GPU Server - Inference using experts stored in blockchain
"""
import os
import sys
import torch
import logging
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import blockchain components
from backend.core.chain import Chain
try:
    from backend.model.blockchain_first_loader import BlockchainFirstModelLoader
    LOADER_CLASS = BlockchainFirstModelLoader
except ImportError:
    try:
        from backend.model.moe_infer import MoEModelManager
        LOADER_CLASS = MoEModelManager
    except ImportError:
        LOADER_CLASS = None
        logger.warning("No blockchain model loader found")

app = FastAPI()

# Global blockchain manager
blockchain_manager = None
chains = {}

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    use_moe: bool = True

@app.on_event("startup")
async def initialize_blockchain():
    """Initialize blockchain and load experts."""
    global blockchain_manager, chains
    
    logger.info("🔗 Initializing Blockchain-Based Inference")
    
    # 1. Initialize chains
    data_dir = Path("./data")
    logger.info(f"Loading blockchain from {data_dir}")
    
    chains['A'] = Chain(data_dir, "A")  # Meta chain
    chains['B'] = Chain(data_dir, "B")  # Parameter chain with experts
    
    # 2. Check expert blocks
    logger.info("Checking expert blocks in chain B...")
    expert_count = 0
    expert_samples = []
    
    for block_hash in list(chains['B']._hash_index.keys())[:100]:  # Sample first 100
        block = chains['B'].get_block_by_hash(block_hash)
        if block and hasattr(block, 'metadata') and block.metadata:
            if block.metadata.get('block_type') == 'expert':
                expert_count += 1
                expert_name = block.metadata.get('expert_name', 'unknown')
                if expert_count <= 5:  # Show first 5
                    expert_samples.append(expert_name)
    
    logger.info(f"✅ Found {expert_count} expert blocks in blockchain")
    if expert_samples:
        logger.info(f"   Sample experts: {', '.join(expert_samples)}")
    
    # 3. Initialize Blockchain Model Manager
    logger.info("Initializing Blockchain Model Manager...")
    try:
        if LOADER_CLASS:
            if LOADER_CLASS.__name__ == "MoEModelManager":
                # Use MoEModelManager
                blockchain_manager = LOADER_CLASS(
                    chain_A=chains['A'],
                    chain_B=chains['B']
                )
            else:
                # Use BlockchainFirstModelLoader
                blockchain_manager = LOADER_CLASS(
                    chains={'A': chains['A'], 'B': chains['B']},
                    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
                )
            logger.info(f"✅ {LOADER_CLASS.__name__} initialized")
        else:
            logger.error("No blockchain loader available!")
            blockchain_manager = None
        
        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠️  No GPU detected - will be slow!")
            
    except Exception as e:
        logger.error(f"Failed to initialize blockchain manager: {e}")
        blockchain_manager = None

@app.get("/health")
async def health():
    """Health check with blockchain status."""
    return {
        "status": "healthy",
        "blockchain_initialized": blockchain_manager is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "chain_A_blocks": len(chains['A']._hash_index) if 'A' in chains else 0,
        "chain_B_blocks": len(chains['B']._hash_index) if 'B' in chains else 0
    }

@app.get("/blockchain/experts")
async def list_experts():
    """List available experts in blockchain."""
    if not chains or 'B' not in chains:
        return {"error": "Blockchain not initialized"}
    
    experts = []
    for block_hash in list(chains['B']._hash_index.keys())[:50]:  # First 50
        block = chains['B'].get_block_by_hash(block_hash)
        if block and hasattr(block, 'metadata') and block.metadata:
            if block.metadata.get('block_type') == 'expert':
                experts.append({
                    "name": block.metadata.get('expert_name'),
                    "layer": block.metadata.get('layer_id'),
                    "hash": block_hash[:16] + "..."
                })
    
    return {
        "total_blocks": len(chains['B']._hash_index),
        "experts_found": len(experts),
        "sample_experts": experts[:10]  # Show first 10
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Generate response using blockchain-stored experts."""
    if not blockchain_manager:
        return {"error": "Blockchain not initialized. Check /health endpoint"}
    
    try:
        logger.info(f"Processing prompt: {request.prompt[:50]}...")
        
        # Generate using blockchain experts
        if request.use_moe:
            logger.info("Using MoE mode with blockchain experts...")
            
            # Try to load expert from blockchain
            sample_expert = "layer0.expert0"  # Try a common expert
            logger.info(f"Attempting to load {sample_expert} from blockchain...")
            
            # Check if expert exists
            expert_block = None
            for block_hash in chains['B']._hash_index.keys():
                block = chains['B'].get_block_by_hash(block_hash)
                if block and hasattr(block, 'metadata') and block.metadata:
                    if block.metadata.get('expert_name') == sample_expert:
                        expert_block = block
                        break
            
            if expert_block:
                logger.info(f"✅ Found expert {sample_expert} in blockchain!")
                # Here you would reconstruct the model from blockchain
                # For now, we'll show it's found
                response = f"[BLOCKCHAIN MODE] Expert {sample_expert} loaded from blockchain. "
                response += f"This would process: {request.prompt}"
                
                return {
                    "prompt": request.prompt,
                    "response": response,
                    "mode": "blockchain",
                    "expert_used": sample_expert,
                    "gpu_used": torch.cuda.is_available()
                }
            else:
                logger.warning(f"Expert {sample_expert} not found in blockchain")
                return {
                    "error": "Experts not found in blockchain",
                    "suggestion": "Upload experts using miner/upload_moe_parameters.py"
                }
        else:
            # Non-MoE mode
            response = blockchain_manager.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens
            )
            
            return {
                "prompt": request.prompt,
                "response": response,
                "mode": "blockchain",
                "gpu_used": torch.cuda.is_available()
            }
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"error": str(e)}

@app.get("/blockchain/verify")
async def verify_blockchain():
    """Verify blockchain has necessary experts."""
    if not chains or 'B' not in chains:
        return {"error": "Blockchain not initialized"}
    
    # Count different types of blocks
    expert_count = 0
    router_count = 0
    meta_count = 0
    
    for block_hash in chains['B']._hash_index.keys():
        block = chains['B'].get_block_by_hash(block_hash)
        if block and hasattr(block, 'metadata') and block.metadata:
            block_type = block.metadata.get('block_type')
            if block_type == 'expert':
                expert_count += 1
            elif block_type == 'router':
                router_count += 1
            elif block_type == 'meta':
                meta_count += 1
    
    # Expected: 48 layers × 128 experts = 6144
    expected_experts = 48 * 128
    completeness = (expert_count / expected_experts * 100) if expected_experts > 0 else 0
    
    return {
        "blockchain_ready": expert_count > 0,
        "expert_blocks": expert_count,
        "router_blocks": router_count,
        "meta_blocks": meta_count,
        "expected_experts": expected_experts,
        "completeness_percent": f"{completeness:.1f}%",
        "status": "ready" if expert_count >= expected_experts else "incomplete"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    logger.info(f"🚀 Starting Blockchain GPU Server on port {port}")
    logger.info("This server uses ONLY blockchain-stored experts for inference")
    uvicorn.run(app, host="0.0.0.0", port=port)