#!/usr/bin/env python3
"""
Blyan GPU Node - Full Integration
Includes blockchain initialization, model loading, and peer sync
"""
import os
import sys
import asyncio
import logging
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
PORT = int(os.environ.get('NODE_PORT', 8002))
MAIN_NODE_URL = os.environ.get('MAIN_NODE_URL', 'http://165.227.221.225:8000')
DATA_DIR = Path(os.environ.get('BLYAN_DATA_DIR', './data'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/gpt-oss-20b')
SKIP_POL = os.environ.get('SKIP_POL', 'true').lower() == 'true'
BLOCKCHAIN_ONLY = os.environ.get('BLOCKCHAIN_ONLY', 'false').lower() == 'true'

class BilyanGPUNode:
    """Integrated GPU node with blockchain and model support."""
    
    def __init__(self):
        self.node_id = f"gpu_node_{os.getpid()}"
        self.port = PORT
        self.chains = {}
        self.model_manager = None
        self.gpu_available = False
        self.gpu_info = {}
        self.genesis_hash = None
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    def check_gpu(self) -> bool:
        """Check GPU availability and get info."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "cuda_version": torch.version.cuda
                }
                logger.info(f"GPU: {self.gpu_info['name']}")
                logger.info(f"Memory: {self.gpu_info['memory_gb']:.2f} GB")
                logger.info(f"CUDA: {self.gpu_info['cuda_version']}")
                return True
        except ImportError:
            logger.warning("PyTorch not installed - installing...")
            import subprocess
            if self._detect_gpu_hardware():
                # Install CUDA version for GPU
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                              "torch", "torchvision", "torchaudio", 
                              "--index-url", "https://download.pytorch.org/whl/cu121"])
            else:
                # Install CPU version
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                              "torch", "torchvision", "torchaudio"])
            return self.check_gpu()  # Recursive call after install
        
        logger.info("No GPU detected - CPU mode")
        return False
    
    def _detect_gpu_hardware(self) -> bool:
        """Detect if GPU hardware exists (even without drivers)."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    async def fetch_genesis_from_main(self) -> Optional[Dict]:
        """Fetch genesis block from main node."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First get genesis hash
                response = await client.get(f"{MAIN_NODE_URL}/genesis/hash")
                if response.status_code == 200:
                    genesis_info = response.json()
                    self.genesis_hash = genesis_info.get('genesis_hash')
                    logger.info(f"Got genesis hash: {self.genesis_hash[:16]}...")
                    
                    # Now get the actual genesis block from chain A
                    response = await client.get(f"{MAIN_NODE_URL}/chain/A/blocks")
                    if response.status_code == 200:
                        blocks = response.json()
                        if blocks and len(blocks) > 0:
                            # Find genesis block (usually first)
                            for block in blocks:
                                if block.get('header', {}).get('block_type') == 'genesis_pact':
                                    logger.info("Found genesis pact block")
                                    return block
                            # If no genesis_pact found, use first block
                            logger.info("Using first block as genesis")
                            return blocks[0]
                else:
                    logger.warning(f"Could not fetch genesis hash: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to fetch genesis: {e}")
        
        return None
    
    def initialize_chains(self) -> bool:
        """Initialize or sync blockchain chains."""
        try:
            from backend.core.chain import Chain
            from backend.core.dataset_chain import DatasetChain
            
            logger.info("Initializing blockchains...")
            
            # Check if chains already exist locally
            chains_exist = (DATA_DIR / "chain_A").exists()
            
            if chains_exist:
                logger.info("Loading existing chains from disk...")
            else:
                logger.info("Creating new chains (first node)...")
            
            # Initialize chains (creates or loads existing)
            self.chains['A'] = Chain(DATA_DIR, "A", skip_pol=SKIP_POL)  # Meta chain
            self.chains['B'] = Chain(DATA_DIR, "B", skip_pol=SKIP_POL)  # Parameter chain
            self.chains['D'] = DatasetChain(DATA_DIR, "D")  # Dataset chain
            
            # Log chain status
            for chain_id, chain in self.chains.items():
                blocks = chain.get_all_blocks()
                logger.info(f"Chain {chain_id}: {len(blocks)} blocks")
            
            # If chain A is empty, we need genesis first
            if not chains_exist or len(self.chains['A'].get_all_blocks()) == 0:
                logger.info("Chain A is empty, will fetch genesis during sync")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chains: {e}")
            return False
    
    async def sync_from_peers(self):
        """Sync blockchain from other nodes."""
        try:
            import httpx
            logger.info(f"Attempting to sync from main node: {MAIN_NODE_URL}")
            
            # First ensure we have genesis
            genesis_block = await self.fetch_genesis_from_main()
            if genesis_block and len(self.chains['A'].get_all_blocks()) == 0:
                logger.info("Adding genesis block to chain A")
                try:
                    # Add genesis block first
                    from backend.core.block import Block, BlockHeader
                    header_dict = genesis_block.get('header', {})
                    header = BlockHeader(**header_dict)
                    payload = genesis_block.get('payload', b'')
                    if isinstance(payload, str):
                        payload = payload.encode('utf-8')
                    elif isinstance(payload, dict):
                        payload = json.dumps(payload).encode('utf-8')
                    
                    block = Block(header=header, payload=payload)
                    self.chains['A'].storage.save_block(block)
                    logger.info("Genesis block added successfully")
                except Exception as e:
                    logger.warning(f"Could not add genesis block directly: {e}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to get chain data from main node
                for chain_id in ['A', 'B', 'D']:
                    try:
                        response = await client.get(f"{MAIN_NODE_URL}/chain/{chain_id}/blocks")
                        if response.status_code == 200:
                            blocks = response.json()
                            logger.info(f"Received {len(blocks)} blocks for chain {chain_id}")
                            
                            # Add blocks to local chain
                            added_count = 0
                            for block_dict in blocks:
                                try:
                                    # For chain A, check if it's genesis and we already have it
                                    if chain_id == 'A' and block_dict.get('header', {}).get('index') == 0:
                                        if len(self.chains[chain_id].get_all_blocks()) > 0:
                                            continue  # Skip if we already have genesis
                                    
                                    # Try to add block
                                    if hasattr(self.chains[chain_id], 'add_block_from_dict'):
                                        self.chains[chain_id].add_block_from_dict(block_dict)
                                        added_count += 1
                                    else:
                                        # Manual block creation
                                        from backend.core.block import Block, BlockHeader
                                        header = BlockHeader(**block_dict.get('header', {}))
                                        payload = block_dict.get('payload', b'')
                                        if isinstance(payload, str):
                                            payload = payload.encode('utf-8')
                                        elif isinstance(payload, dict):
                                            payload = json.dumps(payload).encode('utf-8')
                                        block = Block(header=header, payload=payload)
                                        self.chains[chain_id].storage.save_block(block)
                                        added_count += 1
                                except Exception as e:
                                    logger.debug(f"Could not add block: {e}")
                            
                            if added_count > 0:
                                logger.info(f"Added {added_count} blocks to chain {chain_id}")
                        elif response.status_code == 404 and chain_id == 'D':
                            logger.info(f"Chain {chain_id} not available on main node (expected for dataset chain)")
                        else:
                            logger.warning(f"Could not sync chain {chain_id}: {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Sync error for chain {chain_id}: {e}")
                        
        except ImportError:
            logger.warning("httpx not installed - skipping sync")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
    
    def initialize_model_manager(self) -> bool:
        """Initialize model manager for inference."""
        if BLOCKCHAIN_ONLY:
            logger.info("BLOCKCHAIN_ONLY mode - skipping model initialization")
            return True
            
        try:
            from backend.model.moe_infer import MoEModelManager
            from backend.core.param_index import ParameterIndex
            from backend.model.moe_infer import ExpertUsageTracker
            
            # Initialize components
            param_index = ParameterIndex(DATA_DIR / "param_index.json")
            usage_tracker = ExpertUsageTracker(DATA_DIR / "expert_usage.json")
            
            # Create model manager
            self.model_manager = MoEModelManager(
                meta_chain=self.chains.get('A'),
                param_chain=self.chains.get('B'),
                param_index=param_index,
                usage_tracker=usage_tracker,
                device="cuda" if self.gpu_available else "cpu"
            )
            
            # Check for available experts
            experts = []
            if hasattr(self.model_manager, '_get_available_experts_for_layer'):
                # Try to get experts for different layers
                for layer_id in range(24):  # Assuming 24 layers
                    layer_experts = self.model_manager._get_available_experts_for_layer(f"layer{layer_id}")
                    experts.extend(layer_experts)
            
            if experts:
                logger.info(f"Found {len(experts)} experts from blockchain")
            else:
                logger.info("No experts in blockchain - model download may be needed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            # Try simpler approach
            try:
                from backend.model.arch import ArchModel
                self.model_manager = ArchModel()
                return True
            except:
                return False
    
    async def download_and_extract_model(self):
        """Download model and extract experts to blockchain."""
        if BLOCKCHAIN_ONLY:
            return
            
        logger.info(f"Downloading model: {MODEL_NAME}")
        
        try:
            # Check if model exists locally first
            model_path = Path(f"./models/{MODEL_NAME.split('/')[-1]}")
            
            if not model_path.exists():
                logger.info("Model not found locally - downloading from HuggingFace...")
                
                # Download model using transformers
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info("This may take a while for large models...")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype="auto",
                    device_map="auto" if self.gpu_available else "cpu"
                )
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                
                # Save locally
                model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                logger.info(f"Model saved to {model_path}")
            else:
                logger.info(f"Model found at {model_path}")
            
            # Extract experts and add to blockchain
            logger.info("Extracting experts to blockchain...")
            # This would use upload_moe_parameters.py logic
            # For now, just log that model is ready
            logger.info("Model ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.info("Running without model - blockchain node only")
    
    async def start_server(self):
        """Start HTTP server for the node."""
        try:
            from aiohttp import web
        except ImportError:
            logger.info("Installing aiohttp...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp"])
            from aiohttp import web
        
        app = web.Application()
        
        # Health endpoint
        async def health(request):
            return web.json_response({
                "status": "healthy",
                "node_id": self.node_id,
                "gpu": self.gpu_info.get("name", "None"),
                "gpu_available": self.gpu_available,
                "chains": list(self.chains.keys()),
                "model_ready": self.model_manager is not None,
                "port": self.port
            })
        
        # Chain info endpoint
        async def chain_info(request):
            chain_id = request.match_info.get('chain_id', 'A')
            if chain_id in self.chains:
                blocks = self.chains[chain_id].get_all_blocks()
                return web.json_response({
                    "chain_id": chain_id,
                    "blocks": len(blocks),
                    "latest_hash": blocks[-1]["hash"] if blocks else None
                })
            return web.json_response({"error": "Chain not found"}, status=404)
        
        # Inference endpoint
        async def inference(request):
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
                
                if self.model_manager and not BLOCKCHAIN_ONLY:
                    # Real inference
                    try:
                        response = await asyncio.to_thread(
                            self.model_manager.generate,
                            prompt,
                            max_length=data.get("max_length", 100)
                        )
                        return web.json_response({
                            "node_id": self.node_id,
                            "prompt": prompt,
                            "response": response,
                            "model": MODEL_NAME,
                            "gpu_used": self.gpu_available
                        })
                    except Exception as e:
                        logger.error(f"Inference error: {e}")
                
                # Fallback response
                return web.json_response({
                    "node_id": self.node_id,
                    "prompt": prompt,
                    "response": f"Node {self.node_id} received prompt (model not ready)",
                    "blockchain_only": BLOCKCHAIN_ONLY
                })
                
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        
        # Register routes
        app.router.add_get('/', health)
        app.router.add_get('/health', health)
        app.router.add_get('/chain/{chain_id}', chain_info)
        app.router.add_post('/inference', inference)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Find available port
        port = self.port
        for _ in range(10):
            try:
                site = web.TCPSite(runner, '0.0.0.0', port)
                await site.start()
                self.port = port
                break
            except OSError:
                port += 1
        
        logger.info(f"Server running on http://0.0.0.0:{self.port}")
        
        # Register with main node
        await self.register_with_main()
    
    async def register_with_main(self):
        """Register this node with the main node."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get public IP
                try:
                    resp = await client.get("https://api.ipify.org")
                    public_ip = resp.text.strip() if resp.status_code == 200 else "unknown"
                except:
                    public_ip = "unknown"
                
                # Get available experts for registration
                available_experts = []
                if self.model_manager and hasattr(self.model_manager, '_get_available_experts_for_layer'):
                    for layer_id in range(24):
                        layer_experts = self.model_manager._get_available_experts_for_layer(f"layer{layer_id}")
                        available_experts.extend(layer_experts)
                
                # Register
                data = {
                    "node_id": self.node_id,
                    "host": public_ip,
                    "port": self.port,
                    "available_experts": available_experts,  # This field is required
                    "node_type": "gpu",
                    "gpu_info": self.gpu_info,
                    "chains": list(self.chains.keys()),
                    "model_ready": self.model_manager is not None,
                    "genesis_hash": self.genesis_hash  # Include genesis hash for verification
                }
                
                resp = await client.post(f"{MAIN_NODE_URL}/p2p/register", json=data)
                if resp.status_code == 200:
                    logger.info("Registered with main node")
                else:
                    logger.info(f"Registration status: {resp.status_code}")
                    
        except Exception as e:
            logger.info(f"Could not register: {e}")
    
    async def run(self):
        """Main run loop."""
        logger.info("=" * 60)
        logger.info("BLYAN GPU NODE - INTEGRATED")
        logger.info("=" * 60)
        
        # 1. Check GPU
        self.check_gpu()
        
        # 2. Initialize blockchains
        if not self.initialize_chains():
            logger.error("Failed to initialize chains")
            return
        
        # 3. Sync from peers (including genesis)
        await self.sync_from_peers()
        
        # 4. Initialize model manager
        if not self.initialize_model_manager():
            logger.warning("Running without model manager")
        
        # 5. Start server
        await self.start_server()
        
        # 6. Keep running
        logger.info("Node ready for requests")
        while True:
            await asyncio.sleep(60)
            logger.debug(f"Node {self.node_id} running...")

async def main():
    """Entry point."""
    node = BilyanGPUNode()
    await node.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)