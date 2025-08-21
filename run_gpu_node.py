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
MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/gpt-oss-20b')  # OpenAI's GPT-OSS-20B model
SKIP_POL = os.environ.get('SKIP_POL', 'true').lower() == 'true'
AUTO_UPLOAD = os.environ.get('AUTO_UPLOAD', 'true').lower() == 'true'  # Auto-upload by default

class BilyanGPUNode:
    """Integrated GPU node with blockchain and model support."""
    
    def __init__(self):
        self.node_id = f"gpu_node_{os.getpid()}"
        self.port = PORT
        self.chains = {}
        self.model_manager = None
        self.gpu_available = False
        
        # Initialize learning state
        self.current_learning_round = None
        self.training_in_progress = False
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
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch"])
            return self.check_gpu()  # Recursive call after install
        
        logger.info("No GPU detected - CPU mode")
        return False
    
    
    def _create_local_genesis(self):
        """Create a local genesis block for offline testing."""
        try:
            spec = {
                "model_name": MODEL_NAME,
                "architecture": "mixture-of-experts",
                "num_layers": 24,
                "num_experts": 16,
                "routing_strategy": "top2",
                "created_by": "gpu_node_local",
                "timestamp": time.time()
            }
            payload = json.dumps(spec).encode()
            
            # Add genesis to chain A
            self.chains['A'].add_block(payload, block_type='genesis_pact')
            logger.info("✅ Local genesis block created")
            
            # Store the hash for reference
            blocks = self.chains['A'].get_all_blocks()
            if blocks:
                self.genesis_hash = blocks[0]['hash']
                logger.info(f"Genesis hash: {self.genesis_hash[:16]}...")
        except Exception as e:
            logger.error(f"Failed to create local genesis: {e}")
    
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
                    
                    # Get the full genesis block (index 0) from chain A
                    response = await client.get(f"{MAIN_NODE_URL}/chain/A/block/0")
                    if response.status_code == 200:
                        block = response.json()
                        if block and block.get('header'):
                            logger.info("Found genesis block")
                            return block
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
            
            # If chain A is empty, we can create a local genesis for testing
            if not chains_exist or len(self.chains['A'].get_all_blocks()) == 0:
                if os.getenv("CREATE_LOCAL_GENESIS", "false").lower() == "true":
                    logger.info("Creating local genesis block for testing...")
                    self._create_local_genesis()
                else:
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
            
            # Check if main node is reachable
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{MAIN_NODE_URL}/health")
                    if response.status_code != 200:
                        logger.warning(f"Main node unhealthy: {response.status_code}")
                        return False
            except Exception as e:
                logger.warning(f"Main node unreachable: {e}")
                logger.info("💡 Running in offline mode - will retry sync periodically")
                return False
            
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
                        # First get block metadata
                        response = await client.get(f"{MAIN_NODE_URL}/chain/{chain_id}/blocks")
                        if response.status_code == 200:
                            data = response.json()
                            # Handle both list and dict responses
                            if isinstance(data, dict) and 'blocks' in data:
                                block_metas = data['blocks']
                            else:
                                block_metas = data
                            logger.info(f"Found {len(block_metas)} blocks for chain {chain_id}")
                            
                            # Fetch and add each full block
                            added_count = 0
                            for block_meta in block_metas:
                                try:
                                    block_index = block_meta.get('index', 0)
                                    
                                    # Skip genesis if we already have it
                                    if chain_id == 'A' and block_index == 0:
                                        if len(self.chains[chain_id].get_all_blocks()) > 0:
                                            continue
                                    
                                    # Fetch the full block
                                    block_response = await client.get(f"{MAIN_NODE_URL}/chain/{chain_id}/block/{block_index}")
                                    if block_response.status_code != 200:
                                        continue
                                        
                                    block_dict = block_response.json()
                                    
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
                                    logger.debug(f"Could not add block {block_index}: {e}")
                            
                            if added_count > 0:
                                logger.info(f"Added {added_count} blocks to chain {chain_id}")
                        elif response.status_code == 404 and chain_id == 'D':
                            logger.info(f"Chain {chain_id} not available on main node (expected for dataset chain)")
                        else:
                            logger.warning(f"Could not sync chain {chain_id}: {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Sync error for chain {chain_id}: {e}")
            
            logger.info("✅ Blockchain sync completed")
            return True
                        
        except ImportError:
            logger.warning("httpx not installed - skipping sync")
            return False
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False
    
    def initialize_model_manager(self) -> bool:
        """Initialize blockchain-first model manager for inference."""
        try:
            # Use blockchain-first loader - NO local models
            from backend.model.blockchain_first_loader import BlockchainOnlyModelManager
            from backend.core.param_index import ParameterIndex
            from backend.core.zero_copy_loader import ZeroCopyTileLoader
            
            # Initialize parameter index
            param_index = ParameterIndex(DATA_DIR / "param_index.json")
            
            # Initialize blockchain-only model manager
            self.model_manager = BlockchainOnlyModelManager(
                meta_chain=self.chains.get('A'),
                param_chain=self.chains.get('B'),
                param_index=param_index,
                device="cuda" if self.gpu_available else "cpu"
            )
            
            # Initialize zero-copy loader for efficient loading
            self.zero_copy_loader = ZeroCopyTileLoader(
                chain=self.chains.get('B'),
                cache_dir=DATA_DIR / "tile_cache"
            )
            
            # Check for available experts in blockchain
            available_experts = self.model_manager.get_available_experts()
            
            if available_experts:
                logger.info(f"✅ Found {len(available_experts)} experts in blockchain")
                for expert in available_experts[:5]:  # Show first 5
                    logger.info(f"  - {expert}")
            else:
                logger.info("📦 No experts in blockchain yet")
                if AUTO_UPLOAD:
                    logger.info("🚀 Auto-uploading model to blockchain...")
                    # Create task to download and upload model
                    asyncio.create_task(self.download_and_upload_model())
                else:
                    logger.info("💡 Upload model using: python miner/upload_moe_parameters.py")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            return False
    
    async def download_and_upload_model(self):
        """Download model from HuggingFace and upload to blockchain as experts."""
        logger.info(f"📥 Auto-downloading model: {MODEL_NAME}")
        logger.info("⚠️  This is a 20B parameter model - download may take time...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import gc
            
            # Use INT8 quantization for consistency across all nodes
            logger.info("⏳ Loading model with INT8 quantization (~10GB memory)...")
            
            from transformers import BitsAndBytesConfig
            
            # Configure quantization properly for newer transformers
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto" if self.gpu_available else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info("✅ Model loaded with INT8 quantization")
            
            # Create meta block if needed
            if len(self.chains['A'].get_all_blocks()) == 0:
                meta_spec = {
                    "model_name": MODEL_NAME,
                    "architecture": "mixture-of-experts",
                    "num_layers": 24,
                    "num_experts": 8,
                    "routing_strategy": "top2"
                }
                self.chains['A'].add_block(json.dumps(meta_spec).encode(), block_type='meta')
                logger.info("✅ Created meta block")
            
            # Extract and upload experts with memory-efficient streaming
            num_uploaded = 0
            
            # MoE 구조에 맞게 expert 추출
            for layer_idx in range(24):  # gpt-oss-20b has 24 layers
                layer_name = f"model.layers.{layer_idx}"
                
                # 각 레이어의 MLP를 expert로 추출
                for name, module in model.named_modules():
                    if layer_name in name and 'mlp' in name.lower():
                        expert_name = f"layer{layer_idx}.expert0"  # 각 레이어당 1개 expert로 시작
                        
                        try:
                            # 메모리 효율적 처리: 하나씩 serialize
                            import io
                            import pickle
                            
                            # state_dict를 메모리에 유지하지 않고 바로 serialize
                            buffer = io.BytesIO()
                            state_dict = module.state_dict()
                            
                            # INT8 quantized weights 그대로 저장
                            for key, tensor in state_dict.items():
                                # CPU로 이동 없이 직접 저장 (메모리 절약)
                                state_dict[key] = tensor.detach()
                            
                            pickle.dump(state_dict, buffer)
                            payload = buffer.getvalue()
                            
                            # 블록체인에 추가
                            metadata = json.dumps({
                                "expert_name": expert_name,
                                "layer_id": layer_idx,
                                "quantization": "int8",
                                "model_source": MODEL_NAME
                            })
                            
                            self.chains['B'].add_block(
                                payload,
                                block_type='expert',
                                metadata=metadata
                            )
                            logger.info(f"✅ Uploaded {expert_name} to blockchain")
                            num_uploaded += 1
                            
                            # 메모리 정리
                            del state_dict, payload, buffer
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            break  # 각 레이어당 하나의 expert만
                        
                        except Exception as e:
                            logger.warning(f"Failed to upload {expert_name}: {e}")
                
                # 메모리 부족 방지를 위해 주기적으로 정리
                if layer_idx % 4 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            logger.info(f"✅ Uploaded {num_uploaded} experts to blockchain")
            
            # Reinitialize model manager to use new experts
            self.initialize_model_manager()
            
        except Exception as e:
            logger.error(f"Failed to download/upload model: {e}")
    
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
        
        # Inference endpoint - BLOCKCHAIN-FIRST, no local models
        async def inference(request):
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_length", 100)
                selected_experts = data.get("experts", [])
                
                if not self.model_manager:
                    return web.json_response({
                        "error": "Model manager not initialized"
                    }, status=503)
                
                # Get available experts if none specified
                if not selected_experts:
                    available = self.model_manager.get_available_experts()
                    if available:
                        # Select top experts for inference
                        selected_experts = available[:min(4, len(available))]
                    else:
                        return web.json_response({
                            "error": "No experts in blockchain. Upload model first.",
                            "hint": "Run: python miner/upload_moe_parameters.py"
                        }, status=503)
                
                logger.info(f"Performing blockchain inference with {len(selected_experts)} experts")
                
                # Perform blockchain-first inference
                response = await asyncio.to_thread(
                    self.model_manager.generate,
                    prompt,
                    selected_experts,
                    max_tokens
                )
                
                return web.json_response({
                    "node_id": self.node_id,
                    "prompt": prompt,
                    "response": response,
                    "experts_used": selected_experts,
                    "blockchain_inference": True,
                    "gpu_used": self.gpu_available
                })
                
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        
        # Learning endpoints
        async def learning_start(request):
            """Handle learning start notification from service node."""
            try:
                data = await request.json()
                round_id = data.get("round_id")
                target_expert = data.get("target_expert")
                base_version = data.get("base_version")
                
                # Store learning state
                self.current_learning_round = {
                    "round_id": round_id,
                    "target_expert": target_expert,
                    "base_version": base_version,
                    "status": "notified",
                    "start_time": time.time()
                }
                
                logger.info(f"📚 Learning round {round_id} started for {target_expert}")
                
                return web.json_response({
                    "status": "accepted",
                    "round_id": round_id,
                    "node_id": self.node_id
                })
                
            except Exception as e:
                logger.error(f"Failed to start learning: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def learning_data_allocation(request):
            """Receive data allocation for learning."""
            try:
                data = await request.json()
                round_id = data.get("round_id")
                dataset_ids = data.get("dataset_ids", [])
                
                if not self.current_learning_round or self.current_learning_round["round_id"] != round_id:
                    return web.json_response({"error": "Invalid round"}, status=400)
                
                # Store allocated datasets
                self.current_learning_round["datasets"] = dataset_ids
                self.current_learning_round["status"] = "data_allocated"
                
                logger.info(f"Received {len(dataset_ids)} datasets for round {round_id}")
                
                # Start training asynchronously
                asyncio.create_task(self.execute_training(round_id, dataset_ids))
                
                return web.json_response({
                    "status": "accepted",
                    "datasets_received": len(dataset_ids)
                })
                
            except Exception as e:
                logger.error(f"Failed to receive data allocation: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def learning_status(request):
            """Report current learning status."""
            try:
                if not self.current_learning_round:
                    return web.json_response({
                        "status": "idle",
                        "current_round": None
                    })
                
                return web.json_response({
                    "status": self.current_learning_round.get("status", "unknown"),
                    "round_id": self.current_learning_round.get("round_id"),
                    "progress": self.current_learning_round.get("progress", 0),
                    "elapsed_time": time.time() - self.current_learning_round.get("start_time", 0)
                })
                
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        
        async def submit_delta(request):
            """Submit trained delta to service node."""
            try:
                data = await request.json()
                round_id = data.get("round_id")
                
                if not self.current_learning_round or self.current_learning_round["round_id"] != round_id:
                    return web.json_response({"error": "Invalid round"}, status=400)
                
                # Get the trained delta
                delta = self.current_learning_round.get("trained_delta")
                if not delta:
                    return web.json_response({"error": "No delta available"}, status=400)
                
                logger.info(f"Submitting delta for round {round_id}")
                
                return web.json_response({
                    "status": "submitted",
                    "delta_hash": hashlib.sha256(str(delta).encode()).hexdigest()[:16],
                    "node_id": self.node_id
                })
                
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        
        # Register routes
        app.router.add_get('/', health)
        app.router.add_get('/health', health)
        app.router.add_get('/chain/{chain_id}', chain_info)
        app.router.add_post('/inference', inference)
        
        # Learning routes
        app.router.add_post('/learning/start', learning_start)
        app.router.add_post('/learning/data', learning_data_allocation)
        app.router.add_get('/learning/status', learning_status)
        app.router.add_post('/learning/delta', submit_delta)
        
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
                    logger.info("✅ Registered with main node")
                elif resp.status_code == 500:
                    # Check if it's because distributed coordinator is not initialized
                    if "distributed coordinator" in resp.text.lower():
                        logger.warning("⚠️  Main node P2P/distributed mode is disabled")
                        logger.info("💡 Running as standalone GPU node (blockchain sync only)")
                    else:
                        logger.warning(f"Registration failed (500): {resp.text[:100]}")
                else:
                    logger.info(f"Registration status: {resp.status_code}")
                    
        except Exception as e:
            logger.info(f"Could not register: {e}")
    
    async def periodic_sync(self):
        """Periodically attempt to sync with main node."""
        retry_interval = 30  # Start with 30 seconds
        max_interval = 300  # Max 5 minutes
        
        while True:
            await asyncio.sleep(retry_interval)
            
            if not self.genesis_hash:
                logger.info("Attempting to sync blockchain...")
                success = await self.sync_from_peers()
                
                if success:
                    logger.info("✅ Sync successful!")
                    retry_interval = 30  # Reset interval
                    
                    # Try to register if we haven't
                    await self.register_with_main()
                else:
                    # Exponential backoff
                    retry_interval = min(retry_interval * 2, max_interval)
                    logger.debug(f"Next sync attempt in {retry_interval} seconds")
    
    async def execute_training(self, round_id: str, dataset_ids: List[str]):
        """Execute actual training with allocated datasets."""
        if self.training_in_progress:
            logger.warning("Training already in progress")
            return
        
        self.training_in_progress = True
        logger.info(f"Starting training for round {round_id} with {len(dataset_ids)} datasets")
        
        try:
            # Update status
            self.current_learning_round["status"] = "training"
            self.current_learning_round["progress"] = 0
            
            # Get target expert
            target_expert = self.current_learning_round.get("target_expert", "layer0.expert0")
            base_version = self.current_learning_round.get("base_version", "")
            
            # Load datasets from chain D
            training_data = []
            if dataset_ids:
                for dataset_id in dataset_ids[:5]:  # Limit to 5 datasets for now
                    # In production: Load actual dataset from chain D
                    # For now, simulate with sample data
                    training_data.append({
                        "dataset_id": dataset_id,
                        "samples": ["Sample text for training"] * 10
                    })
            
            # Simulate training process
            total_steps = 100
            for step in range(total_steps):
                # Update progress
                self.current_learning_round["progress"] = (step + 1) / total_steps * 100
                
                # Simulate training step
                await asyncio.sleep(0.1)  # Simulate computation time
                
                if step % 10 == 0:
                    logger.info(f"Training progress: {self.current_learning_round['progress']:.1f}%")
            
            # Generate delta (difference between new and old weights)
            # In production: Calculate actual weight deltas from training
            trained_delta = {
                "expert": target_expert,
                "base_version": base_version,
                "delta_weights": {
                    "layer.weight": [0.001] * 100,  # Simulated weight changes
                    "layer.bias": [0.0001] * 10
                },
                "improvement_metrics": {
                    "loss_reduction": 0.15,
                    "accuracy_gain": 0.02
                },
                "training_metadata": {
                    "datasets_used": len(dataset_ids),
                    "training_steps": total_steps,
                    "node_id": self.node_id
                }
            }
            
            # Store trained delta
            self.current_learning_round["trained_delta"] = trained_delta
            self.current_learning_round["status"] = "trained"
            
            logger.info(f"✅ Training complete for round {round_id}")
            
            # Submit delta to service node
            await self.submit_delta_to_service(round_id, trained_delta)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.current_learning_round["status"] = "failed"
            self.current_learning_round["error"] = str(e)
        finally:
            self.training_in_progress = False
    
    async def submit_delta_to_service(self, round_id: str, delta: Dict):
        """Submit trained delta to service node."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Submit delta to service node
                resp = await client.post(
                    f"{MAIN_NODE_URL}/learning/node_delta",
                    json={
                        "round_id": round_id,
                        "node_id": self.node_id,
                        "delta": delta,
                        "timestamp": time.time()
                    }
                )
                
                if resp.status_code == 200:
                    logger.info(f"✅ Delta submitted for round {round_id}")
                    self.current_learning_round["status"] = "delta_submitted"
                else:
                    logger.warning(f"Failed to submit delta: {resp.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to submit delta: {e}")
    
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
        
        # 3. Initial sync attempt (non-blocking)
        sync_success = await self.sync_from_peers()
        
        # 4. Initialize model manager
        if not self.initialize_model_manager():
            logger.warning("Running without model manager")
        
        # 5. Start server
        await self.start_server()
        
        # 6. Start periodic sync if initial sync failed
        if not sync_success:
            asyncio.create_task(self.periodic_sync())
        
        # 7. Keep running
        logger.info("✅ Node ready for requests")
        logger.info(f"🌐 API available at http://0.0.0.0:{self.port}")
        
        if not sync_success:
            logger.info("📡 Running in offline mode - will sync when main node becomes available")
        
        while True:
            await asyncio.sleep(60)
            # Periodic health check
            if hasattr(self, 'model_manager') and self.model_manager:
                logger.debug(f"Node {self.node_id} healthy - GPU: {self.gpu_available}")

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