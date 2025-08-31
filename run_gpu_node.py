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
import atexit
import signal as sig
from pathlib import Path
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Version check for debugging
try:
    import transformers
    import tokenizers
    logger.info(f"Library versions - transformers: {transformers.__version__}, tokenizers: {tokenizers.__version__}")
except Exception as e:
    logger.warning(f"Could not check library versions: {e}")

# Import compatibility layer
try:
    from backend.common.compat import (
        setup_hf_cache,
        load_tokenizer,
        load_model_any_precision,
        detect_capabilities,
        print_capabilities
    )
    COMPAT_AVAILABLE = True
except ImportError:
    logger.warning("Compatibility module not available, using fallback loading")
    COMPAT_AVAILABLE = False

# Import model configuration
try:
    from config.model_profile import (
        MODEL_ID, MODEL_NAME, ARCHITECTURE, LAYERS,
        CONTEXT, PRECISION, COMPUTE, BLOCKCHAIN,
        get_model_config
    )
    DEFAULT_MODEL_NAME = MODEL_ID
    PROFILE_AVAILABLE = True
except ImportError:
    logger.warning("Model profile not available, using fallback")
    DEFAULT_MODEL_NAME = 'Qwen/Qwen3-8B-FP8'
    PROFILE_AVAILABLE = False
    def get_model_config(name):
        return {}

# Configuration
# Accept NODE_PORT or PORT (alias). Support 'auto' or '0' for ephemeral.
_raw_port = os.environ.get('NODE_PORT') or os.environ.get('PORT')
if _raw_port is None:
    PORT = 8001
else:
    try:
        PORT = 0 if str(_raw_port).lower() in ('auto', '0') else int(_raw_port)
    except Exception:
        PORT = 8001

# Accept API_URL as alias for MAIN_NODE_URL
MAIN_NODE_URL = os.environ.get('MAIN_NODE_URL') or os.environ.get('API_URL') or 'http://165.227.221.225:8000'
DATA_DIR = Path(os.environ.get('BLYAN_DATA_DIR', './data'))
MODEL_NAME = os.environ.get('MODEL_NAME', DEFAULT_MODEL_NAME)

# 🔒 Production Safety Settings (always production)
# Default: Proof-of-Learning ON, no HF bootstrap unless explicitly allowed
SKIP_POL = os.environ.get('SKIP_POL', 'false').lower() == 'true'
AUTO_UPLOAD = os.environ.get('AUTO_UPLOAD', 'false').lower() == 'true'
ALLOW_HF_UPLOAD = os.environ.get('ALLOW_HF_UPLOAD', 'false').lower() == 'true'
if SKIP_POL:
    logger.warning("⚠️ PoL disabled via SKIP_POL=true — NOT RECOMMENDED in production")
logger.info("🔒 Running with production defaults (PoL on by default)")

# Auto-apply safe optimizations
def apply_production_optimizations():
    """Apply safe production optimizations automatically."""
    if os.environ.get('OPTIMIZATIONS_APPLIED'):
        return
    
    logger.info("🚀 Auto-applying production optimizations...")
    
    # GPU Memory optimizations
    if not os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True,max_split_size_mb:128"
        logger.info("  ✅ Set PYTORCH_CUDA_ALLOC_CONF for better memory management")
    
    # Reduce memory fragmentation
    if not os.environ.get('GPU_DIRECT_CHUNK_SIZE'):
        os.environ['GPU_DIRECT_CHUNK_SIZE'] = "1"
        logger.info("  ✅ Set GPU_DIRECT_CHUNK_SIZE=1 for tight VRAM")
    
    if not os.environ.get('GPU_LOAD_WORKERS'):
        os.environ['GPU_LOAD_WORKERS'] = "1"
        logger.info("  ✅ Set GPU_LOAD_WORKERS=1 to prevent memory spikes")
    
    # CPU optimizations to reduce memory pressure
    if not os.environ.get('OMP_NUM_THREADS'):
        os.environ['OMP_NUM_THREADS'] = "1"
    if not os.environ.get('MKL_NUM_THREADS'):
        os.environ['MKL_NUM_THREADS'] = "1"
    logger.info("  ✅ Limited CPU threads to reduce memory usage")
    
    # Disable tokenizer parallelism warnings
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    
    # Mark optimizations as applied
    os.environ['OPTIMIZATIONS_APPLIED'] = "true"
    
    # Always safe optimizations
    os.environ.setdefault('ENABLE_FUSED_SNAPSHOT', 'true')
    os.environ.setdefault('BLOCK_FETCH_MAX_WORKERS', '4')
    os.environ.setdefault('SNAPSHOT_MAX_AGE_HOURS', '12')
    
    # Check if we have full model before going offline
    param_index_path = DATA_DIR / "param_index.json"
    if param_index_path.exists():
        try:
            import json
            with open(param_index_path) as f:
                index = json.load(f)
            
            # Need at least 38 layers for full model
            if len(index) >= 38:
                os.environ['TRANSFORMERS_OFFLINE'] = 'true'
                os.environ['SKIP_UPLOAD_IF_PARAM_INDEX_MATCHES'] = 'true'
                logger.info("   ✅ Full model in blockchain - offline mode enabled")
            else:
                logger.info(f"   ⚠️ Only {len(index)} layers - online mode required")
        except Exception as e:
            logger.debug(f"   Could not check param_index: {e}")
    
    # Verify on first run, then trust cache
    cache_verified = (DATA_DIR / ".verification_complete").exists()
    if cache_verified:
        os.environ.setdefault('VERIFY_ON_START', 'false')
        logger.info("   ✅ Previous verification cached - fast startup")
    else:
        os.environ.setdefault('VERIFY_ON_START', 'true')
        logger.info("   📋 First run - will verify chain integrity")
    
    os.environ['OPTIMIZATIONS_APPLIED'] = 'true'
    logger.info("   ✅ Optimizations applied")

# Apply optimizations early (always production)
apply_production_optimizations()

# Optional: Use custom public IP/hostname if provided, otherwise auto-detect
PUBLIC_HOST = os.environ.get('PUBLIC_HOST', '')  # Can be IP, domain, or empty for auto-detect
# If PUBLIC_PORT is not set and PORT is ephemeral (0), default public port to 8001 instead of 0
_raw_public_port = os.environ.get('PUBLIC_PORT')
if _raw_public_port is not None:
    try:
        PUBLIC_PORT = int(_raw_public_port)
    except Exception:
        PUBLIC_PORT = 80
else:
    PUBLIC_PORT = PORT if isinstance(PORT, int) and PORT != 0 else 8001

# Model precision is now auto-detected from model config
# FP8 for Qwen3-30B, FP16 for others
REQUIRE_INT8_SUPPORT = False  # INT8 not required

class BlyanGPUNode:
    """Integrated GPU node with blockchain and model support."""
    
    def __init__(self):
        # Generate and validate node ID
        import re
        import socket
        
        # Detect RunPod environment and extract pod ID
        runpod_pod_id = os.environ.get('RUNPOD_POD_ID')
        public_host = os.environ.get('PUBLIC_HOST', '')
        
        # Try to extract pod ID from PUBLIC_HOST if RUNPOD_POD_ID not set
        if not runpod_pod_id and 'runpod.net' in public_host:
            # Extract from format like: txpnn40k57a1ye-8000.proxy.runpod.net
            try:
                pod_part = public_host.split('.')[0]  # txpnn40k57a1ye-8000
                if '-' in pod_part:
                    runpod_pod_id = pod_part.split('-')[0]  # txpnn40k57a1ye
                    logger.info(f"Extracted RunPod ID from PUBLIC_HOST: {runpod_pod_id}")
            except:
                pass
        
        # Generate stable node ID
        if os.environ.get('NODE_ID'):
            # Explicitly set NODE_ID takes priority
            self.node_id = os.environ.get('NODE_ID')
            logger.info(f"Using explicit NODE_ID from environment")
        elif runpod_pod_id:
            # RunPod environment - use pod ID for stability
            self.node_id = f"gpu_{runpod_pod_id}"
            logger.info(f"Using RunPod pod ID for node identification")
        else:
            # Fallback to hostname-based ID for stability across restarts
            hostname = socket.gethostname().replace('.', '_')
            self.node_id = f"gpu_{hostname}"
            logger.info(f"Using hostname-based node ID")
        
        # Validate node ID format
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', self.node_id):
            logger.error(f"Invalid node_id format: '{self.node_id}'")
            logger.error("Node ID must be alphanumeric with _ or -, max 64 chars")
            # Auto-fix by replacing invalid characters
            self.node_id = re.sub(r'[^a-zA-Z0-9_-]', '_', self.node_id)[:64]
            logger.info(f"Auto-corrected node_id to: '{self.node_id}'")
        
        # Always log the node ID prominently
        logger.info("=" * 60)
        logger.info(f"🔑 NODE ID: {self.node_id}")
        logger.info("=" * 60)
        
        if runpod_pod_id:
            logger.info(f"📍 RunPod Pod ID detected: {runpod_pod_id}")
        if public_host:
            logger.info(f"🌐 Public host: {public_host}")
        
        self.port = PORT
        self.server_started = False  # One-time server-start guard
        self.chains = {}
        self.model_manager = None
        self.gpu_available = False
        self.supports_int8 = False  # Track INT8 support
        
        # Initialize learning state
        self.current_learning_round = None
        self.training_in_progress = False
        self.gpu_info = {}
        self.genesis_hash = None
        # Precision is now auto-detected from model config
        model_config = get_model_config()
        self.precision = model_config.get('precision', 'auto')
        # Warmup tracking
        self._warmup_status = 'idle'
        # VRAM health tracking
        self.vram_healthy = True
        self.vram_oom_count = 0
        self.degraded_mode = False
        # Background auditor
        self.background_auditor = None
        
        # Queue management
        self.queue_manager = None
        # Support both JOB_CAPACITY and MAX_CONCURRENT_JOBS (JOB_CAPACITY takes precedence)
        self.max_concurrent_jobs = int(os.environ.get('JOB_CAPACITY', os.environ.get('MAX_CONCURRENT_JOBS', '1')))
        # Alias used in some responses
        self.job_capacity = self.max_concurrent_jobs
        self.active_jobs = 0
        self._job_lock = asyncio.Lock()
        self._admission_task = None
        self._sse_clients = {}  # ticket_id -> SSE response
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _blockchain_model_readiness(self) -> tuple:
        """Return (ready, have_count, need_count) for on-chain model components.
        Ready means param_index contains at least num_hidden_layers + 3 components
        (embedding, all transformer layers, model_norm, lm_head)."""
        try:
            from backend.core.param_index import ParameterIndex
            try:
                from config.model_profile import LAYERS as _LAYERS
                need = int(_LAYERS.get('num_hidden_layers', 36)) + 3
            except Exception:
                need = 39  # Conservative default
            p = ParameterIndex(DATA_DIR / "param_index.json")
            have = len(p.get_all_layers())
            return (have >= need, have, need)
        except Exception:
            return (False, 0, 39)
        
    def check_gpu(self) -> bool:
        """Check GPU availability and detect all GPUs."""
        # Use compatibility layer if available
        if COMPAT_AVAILABLE:
            caps = detect_capabilities()
            self.gpu_available = caps['cuda']
            self.supports_int8 = False  # INT8 not required anymore
            
            if caps['cuda']:
                import torch
                num_gpus = torch.cuda.device_count()
                
                # Store info for all GPUs
                self.gpu_info = {
                    "num_gpus": num_gpus,
                    "gpus": [],
                    "cuda_version": caps['cuda_version'],
                    "supports_bf16": caps['supports_bf16'],
                    "bnb_cuda": caps.get('bnb_cuda', False),
                    # For compatibility, keep single GPU info
                    "name": caps['gpu_name'],
                    "memory_gb": caps['gpu_memory_gb']
                }
                
                # Get info for each GPU
                for i in range(num_gpus):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_gb": gpu_props.total_memory / 1e9
                    }
                    self.gpu_info["gpus"].append(gpu_info)
                    logger.info(f"GPU {i}: {gpu_info['name']} - Memory: {gpu_info['memory_gb']:.2f} GB")
                
                logger.info(f"Total GPUs detected: {num_gpus}")
                logger.info(f"CUDA: {self.gpu_info['cuda_version']}")
                logger.info(f"✅ FP16 support enabled on all GPUs")
                return True
            else:
                logger.info("No GPU detected - CPU mode with FP16")
                return False
        else:
            # Fallback to original method
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
                "architecture": "dense",
                "num_layers": 36,
                "hidden_size": 3584,
                "created_by": "gpu_node_local",
                "timestamp": time.time()
            }
            payload = json.dumps(spec).encode()
            
            # Add genesis to chain A
            self.chains['A'].add_block(payload, block_type='genesis_pact')
            logger.info("✅ Local genesis block created")
            
            # Store the hash for reference (only load first block)
            if hasattr(self.chains['A'], 'storage'):
                if hasattr(self.chains['A'], 'get_block_by_index'):
                    first_block = self.chains['A'].get_block_by_index(0)
                elif hasattr(self.chains['A'], 'storage') and self.chains['A'].storage:
                    first_block = self.chains['A'].storage.get_block_by_index(0)
                else:
                    first_block = None
                if first_block:
                    self.genesis_hash = first_block.compute_hash()
                logger.info(f"Genesis hash: {self.genesis_hash[:16]}...")
        except Exception as e:
            logger.error(f"Failed to create local genesis: {e}")
    
    def initialize_chains(self) -> bool:
        """Initialize or sync blockchain chains."""
        try:
            # Choose chain implementation (optimized by default, no env required)
            try:
                from backend.core.chain_optimized import OptimizedChain as ChainCls
                logger.info("🔗 Using optimized chain loading (default)")
            except Exception as e:
                from backend.core.chain import Chain as ChainCls
                logger.warning(f"Optimized chain unavailable, using standard: {e}")
            logger.info(f"   Data dir: {DATA_DIR}")
            logger.info(f"   Skip PoL: {SKIP_POL}")
            logger.info(f"   Verify on start: {os.getenv('VERIFY_ON_START', 'false')}")
            
            from backend.core.dataset_chain import DatasetChain
            
            logger.info("📂 Initializing blockchains...")
            
            # Check if chains already exist locally (correct directories: A/B/D)
            chains_exist = (DATA_DIR / "A").exists() or (DATA_DIR / "B").exists() or (DATA_DIR / "D").exists()
            
            if chains_exist:
                logger.info("   ✅ Loading existing chains from disk...")
                # Log chain sizes using correct paths
                import glob
                for chain_id in ['A', 'B', 'D']:
                    chain_dir = DATA_DIR / chain_id
                    if chain_dir.exists():
                        block_count = len(list(chain_dir.glob("*.json")))
                        logger.info(f"      Chain {chain_id}: {block_count} blocks on disk")
            else:
                logger.info("   📝 Creating new chains (first node)...")
            
            # Initialize chains (creates or loads existing)
            self.chains['A'] = ChainCls(DATA_DIR, "A", skip_pol=SKIP_POL)  # Meta chain
            self.chains['B'] = ChainCls(DATA_DIR, "B", skip_pol=SKIP_POL)  # Parameter chain
            self.chains['D'] = DatasetChain(DATA_DIR, "D", skip_pol=SKIP_POL)  # Dataset chain
            
            # Log chain status (optimized to avoid loading all blocks)
            for chain_id, chain in self.chains.items():
                # Use cached block count instead of loading all blocks
                if hasattr(chain, '_hash_index'):
                    # OptimizedChain has a hash index we can use
                    block_count = len(chain._hash_index)
                    logger.info(f"Chain {chain_id}: {block_count} blocks (indexed)")
                else:
                    # Fallback for chains without index
                    # Use cached index count (O(1) instead of O(n))
                    block_count = len(chain._hash_index) if hasattr(chain, '_hash_index') else 0
                    logger.info(f"Chain {chain_id}: {block_count} blocks (from index)")
            
            # If chain A is empty, we can create a local genesis for testing
            chain_a_empty = len(self.chains['A']._hash_index) == 0 if hasattr(self.chains['A'], '_hash_index') else 0
            if not chains_exist or chain_a_empty:
                if os.getenv("CREATE_LOCAL_GENESIS", "false").lower() == "true":
                    logger.info("Creating local genesis block for testing...")
                    self._create_local_genesis()
                else:
                    logger.info("Chain A is empty, will fetch genesis during sync")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chains: {e}")
            return False
    
    async def sync_from_peers(self, force_full: bool = False):
        """Sync blockchain from other GPU nodes (NOT from main node).
        
        Note: Main node (DigitalOcean) is just a service coordinator.
        Only GPU nodes store blockchain data.
        
        Args:
            force_full: If True, reset local chains before syncing
        """
        # Currently, the main node does not host the GPU blockchain.
        # Each GPU node maintains its own local chains; future versions may add peer sync.
        logger.info("📝 GPU nodes maintain local blockchains (peer sync not yet implemented).")
        
        # For now, skip sync since main node doesn't store blockchain
        # TODO: Implement GPU-to-GPU peer sync in the future
        return True  # Return success to continue initialization
    
    def check_block_progress(self) -> dict:
        """Check blockchain progress (counts only; integrity handled separately)."""
        progress = {
            "meta_blocks": 0,
            "layer_blocks": 0,
            "expected_layers": 36,  # Dense model has 36 layers
            "missing_layers": [],
            "integrity_valid": True,
            "progress_percentage": 0.0
        }
        
        try:
            # Check meta chain (use index count for O(1) performance)
            meta_blocks_count = len(self.chains['A']._hash_index) if hasattr(self.chains['A'], '_hash_index') else 0
            progress["meta_blocks"] = meta_blocks_count
            
            # Dense model expects 36 layers
            progress["expected_layers"] = 36
            
            # Prefer ParameterIndex to avoid scanning blocks (O(1))
            from backend.core.param_index import ParameterIndex
            param_index_path = DATA_DIR / "param_index.json"
            layer_blocks = []
            if param_index_path.exists():
                try:
                    param_index = ParameterIndex(param_index_path)
                    existing_layers = set(param_index.get_all_layers())
                    progress["layer_blocks"] = len(existing_layers)
                    # Determine missing layers from param index
                    expected = [f"layer_{i}" for i in range(36)]
                    for layer_name in expected:
                        if layer_name not in existing_layers:
                            progress["missing_layers"].append(layer_name)
                            if len(progress["missing_layers"]) >= 10:
                                progress["missing_layers"].append("... and more")
                                break
                except Exception as e:
                    logger.debug(f"Param index not available: {e}; falling back to scan")
            
            if progress["layer_blocks"] == 0:
                # Fallback: scan only if param index absent (small chains)
                logger.info("📊 Counting blocks in chain B (fallback scan)...")
                legacy_blocks = self.chains['B'].get_blocks_by_type('layer')
                dense_blocks = self.chains['B'].get_blocks_by_type('dense_layer')
                layer_blocks = legacy_blocks + dense_blocks
                progress["layer_blocks"] = len(layer_blocks)
            
            # Find missing layers (skip if we have many blocks)
            if layer_blocks:
                existing_layers = set()
                for block in layer_blocks:
                    if getattr(block.header, 'layer_name', None):
                        existing_layers.add(block.header.layer_name)
                    elif getattr(block.header, 'layer_id', None) is not None:
                        existing_layers.add(f"layer_{block.header.layer_id}")
                expected = [f"layer_{i}" for i in range(36)]
                for layer_name in expected:
                    if layer_name not in existing_layers and layer_name not in progress["missing_layers"]:
                        progress["missing_layers"].append(layer_name)
                        if len(progress["missing_layers"]) >= 10:
                            progress["missing_layers"].append("... and more")
                            break
            
            # Calculate progress percentage
            if progress["expected_layers"] > 0:
                progress["progress_percentage"] = (progress["layer_blocks"] / progress["expected_layers"]) * 100
            
            # Integrity is verified separately in startup Step 4
            progress["integrity_valid"] = True
            
            # Log progress summary
            logger.info(f"📊 Block Progress Report:")
            logger.info(f"  Meta blocks: {progress['meta_blocks']}")
            logger.info(f"  Layer blocks: {progress['layer_blocks']}/{progress['expected_layers']} ({progress['progress_percentage']:.1f}%)")
            logger.info(f"  Integrity: {'✅ Valid' if progress['integrity_valid'] else '❌ Invalid'}")
            
            if progress["missing_layers"] and len(progress["missing_layers"]) < 10:
                logger.info(f"  Missing layers: {progress['missing_layers'][:5]}")
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to check block progress: {e}")
            progress["integrity_valid"] = False
            return progress
    
    def verify_block_integrity(self) -> bool:
        """Verify blockchain integrity using header index and tail verification."""
        try:
            if not self.chains:
                logger.error("No chains initialized")
                return False
            
            # Import header index module
            from backend.core.header_index import HeaderIndex
            
            # Verify each chain
            for chain_id in ['A', 'B']:
                chain = self.chains.get(chain_id)
                if not chain:
                    continue
                
                # Get chain directory
                chain_dir = DATA_DIR / chain_id
                if not chain_dir.exists():
                    logger.debug(f"Chain {chain_id} directory doesn't exist (OK for new chain)")
                    continue
                
                # Initialize header index
                header_index = HeaderIndex(chain_dir)
                
                # Build index if missing (one-time migration)
                if not (chain_dir / "headers.idx.jsonl").exists():
                    logger.info(f"Building header index for chain {chain_id}...")
                    header_index.build_from_chain(chain)
                
                # Load header records
                records = header_index.load()
                if len(records) == 0:
                    logger.debug(f"Chain {chain_id} empty (OK)")
                    continue
                
                # Load or create finality anchor (per-chain)
                anchor_file = DATA_DIR / f"finality_anchor_{chain_id}.json"
                anchor_data = None
                
                if anchor_file.exists():
                    try:
                        with open(anchor_file, 'r') as f:
                            anchor_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load finality anchor: {e}")
                
                # Create bootstrap anchor if missing
                if not anchor_data:
                    FINALITY_DEPTH = int(os.getenv('FINALITY_DEPTH', '512'))
                    anchor_height = max(0, len(records) - FINALITY_DEPTH)
                    if anchor_height >= 0 and anchor_height < len(records):
                        anchor_record = records[anchor_height]
                        anchor_data = {
                            'height': anchor_height,
                            'cum_digest': anchor_record.cum_digest,
                            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                        }
                        # Save bootstrap anchor
                        try:
                            with open(anchor_file, 'w') as f:
                                json.dump(anchor_data, f, indent=2)
                            logger.info(f"Created bootstrap finality anchor at height {anchor_height}")
                        except Exception as e:
                            logger.warning(f"Could not save bootstrap anchor: {e}")
                
                # Verify header chain to anchor
                if anchor_data:
                    anchor_height = anchor_data.get('height', 0)
                    anchor_digest = anchor_data.get('cum_digest', '')
                    
                    if anchor_height < len(records):
                        # Verify up to anchor height (inclusive)
                        if not header_index.verify_to_height(anchor_height, anchor_digest):
                            logger.error(f"Chain {chain_id}: Header verification failed to anchor")
                            return False
                        logger.info(f"Chain {chain_id}: Headers verified to anchor at height {anchor_height}")
                
                # Verify tail bodies
                TAIL_VERIFY_DEPTH = int(os.getenv('TAIL_VERIFY_DEPTH', '128'))
                start_idx = max(0, len(records) - TAIL_VERIFY_DEPTH)
                end_idx = len(records)
                
                if end_idx > start_idx:
                    success, error_msg = header_index.verify_tail_bodies(chain, start_idx, end_idx)
                    if not success:
                        logger.error(f"Chain {chain_id}: Tail verification failed - {error_msg}")
                        return False
                    logger.info(f"Chain {chain_id}: Verified {end_idx - start_idx} tail blocks")
            
            logger.info("✅ Blockchain integrity verified")
            return True
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False
    
    def _verify_last_block_integrity_legacy(self, chain, chain_id: str, block_count: int) -> bool:
        """Verify chain integrity using cached hash index - O(1) operation.
        
        OPTIMIZED: Uses in-memory hash index instead of loading blocks from disk.
        Falls back to disk verification only if hash index is unavailable.
        """
        try:
            import hashlib
            start_time = time.time()
            
            # OPTIMIZATION: Use cached hash index if available
            if hasattr(chain, '_hash_index') and chain._hash_index:
                # Fast path: just verify we have the expected number of blocks
                index_count = len(chain._hash_index)
                if index_count == block_count:
                    # Get any hash from the index to show we have data
                    sample_hash = list(chain._hash_index.keys())[0] if chain._hash_index else "empty"
                    elapsed = time.time() - start_time
                    logger.info(f"Chain {chain_id}: {block_count} blocks verified via hash index (sample: {sample_hash[:16]}...) ({elapsed:.3f}s)")
                    return True
                else:
                    logger.error(f"Chain {chain_id}: Index mismatch - expected {block_count}, found {index_count}")
                    return False
            
            # Fallback: verify a small tail window for stronger assurance
            # Default to a safer tail depth by default (no env required)
            tail = max(1, int(os.getenv('FAST_RECHECK_DEPTH', '128')))
            start_idx = max(0, block_count - tail)
            loader = None
            if hasattr(chain, 'get_block_by_index'):
                loader = chain.get_block_by_index
            elif hasattr(chain, 'storage') and chain.storage and hasattr(chain.storage, 'get_block_by_index'):
                loader = chain.storage.get_block_by_index
            else:
                logger.info(f"Chain {chain_id}: Empty chain, nothing to verify")
                return True

            prev_hash = None
            verified = 0
            for idx in range(start_idx, block_count):
                blk = loader(idx)
                if not blk:
                    logger.error(f"Chain {chain_id}: Missing block at index {idx}")
                    return False
                # Verify payload hash matches header
                if hashlib.sha256(blk.payload).hexdigest() != blk.header.payload_hash:
                    logger.error(f"Chain {chain_id}: Payload hash mismatch at index {idx}")
                    return False
                # Verify prev linkage (skip genesis)
                if idx > 0:
                    if prev_hash is None:
                        prev = loader(idx - 1)
                        if not prev:
                            logger.error(f"Chain {chain_id}: Missing prev block at index {idx-1}")
                            return False
                        prev_hash = prev.compute_hash()
                    if blk.header.prev_hash != prev_hash:
                        logger.error(f"Chain {chain_id}: Broken prev_hash at index {idx}")
                        return False
                    prev_hash = blk.compute_hash()
                else:
                    prev_hash = blk.compute_hash()
                verified += 1

            elapsed = time.time() - start_time
            logger.info(f"Chain {chain_id}: verified last {verified} blocks link+payload in {elapsed:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying chain {chain_id}: {e}")
            return False
    
    async def acquire_job_slot(self) -> bool:
        """Try to acquire a job slot."""
        async with self._job_lock:
            if self.active_jobs < self.max_concurrent_jobs:
                self.active_jobs += 1
                logger.debug(f"Acquired job slot ({self.active_jobs}/{self.max_concurrent_jobs})")
                return True
            return False
    
    async def release_job_slot(self) -> None:
        """Release a job slot."""
        async with self._job_lock:
            if self.active_jobs > 0:
                self.active_jobs -= 1
                logger.debug(f"Released job slot ({self.active_jobs}/{self.max_concurrent_jobs})")
                
                # Trigger admission check when slot freed
                if self.queue_manager:
                    asyncio.create_task(self._try_admit())
    
    async def _try_admit(self) -> None:
        """Try to admit queued requests."""
        if not self.queue_manager:
            return
        
        async with self._job_lock:
            available_slots = self.max_concurrent_jobs - self.active_jobs
            if available_slots > 0:
                admitted = await self.queue_manager.admit_next(available_slots)
                for ticket in admitted:
                    # Notify via SSE if client is listening
                    await self._notify_ticket_update(ticket.ticket_id)
    
    async def _notify_ticket_update(self, ticket_id: str) -> None:
        """Notify SSE clients about ticket updates."""
        if ticket_id in self._sse_clients:
            # Will be implemented with SSE endpoints
            pass
    
    def _handle_vram_oom(self, context: str) -> None:
        """Handle VRAM out-of-memory errors gracefully."""
        self.vram_oom_count += 1
        self.vram_healthy = False
        
        logger.error(f"🚨 VRAM OOM during {context} (count: {self.vram_oom_count})")
        
        # Enter degraded mode after 2 OOMs
        if self.vram_oom_count >= 2 and not self.degraded_mode:
            self.degraded_mode = True
            logger.warning("⚠️ Entering degraded mode - reducing service capacity")
            
            # Try to free memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Log current VRAM usage
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(f"VRAM after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                logger.error(f"Failed to clear VRAM: {e}")
    
    def reset_blockchain(self, chain_id: str = None):
        """Reset blockchain to clean state."""
        try:
            if chain_id:
                chains_to_reset = [chain_id]
            else:
                chains_to_reset = ['A', 'B', 'D']
            
            for cid in chains_to_reset:
                if cid in self.chains:
                    logger.warning(f"🔄 Resetting chain {cid}")
                    # Remove all block files
                    chain_dir = DATA_DIR / cid
                    if chain_dir.exists():
                        import shutil
                        shutil.rmtree(chain_dir)
                        chain_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Reinitialize chain (optimized by default)
                    try:
                        from backend.core.chain_optimized import OptimizedChain as ChainCls
                        logger.info("Using optimized chain loading")
                    except Exception:
                        from backend.core.chain import Chain as ChainCls
                        logger.info("Using standard chain loading")
                    
                    self.chains[cid] = ChainCls(DATA_DIR, cid, skip_pol=True)
                    
                    # Re-add genesis for meta chain
                    if cid == 'A' and self.genesis_hash:
                        try:
                            # Create proper genesis block
                            genesis_data = {
                                "version": "1.0.0",
                                "timestamp": time.time(),
                                "network": "blyan",
                                "genesis": True,
                                "hash": self.genesis_hash
                            }
                            self.chains['A'].add_block(
                                json.dumps(genesis_data).encode(),
                                block_type='genesis_pact'
                            )
                            logger.info(f"✅ Re-created genesis block for chain A")
                        except Exception as e:
                            logger.warning(f"Could not recreate genesis: {e}")
            
            logger.info(f"✅ Blockchain reset complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset blockchain: {e}")
            return False
    
    async def handle_integrity_failure(self):
        """Handle blockchain integrity failures."""
        logger.warning("⚠️ Blockchain integrity check failed - attempting recovery")
        
        # Reset and rebuild (main node has no blockchain to sync from)
        logger.warning("Resetting blockchain to restore integrity")
        
        # Save any uploaded layers before reset
        layer_blocks = self.chains['B'].get_blocks_by_type('layer')
        saved_layers = []
        if layer_blocks:
            logger.info(f"Saving {len(layer_blocks)} layer blocks before reset...")
            for block in layer_blocks:
                saved_layers.append({
                    'payload': block.payload,
                    'layer_id': block.header.layer_id
                })
        
        # Reset chains
        self.reset_blockchain()
        
        # Re-initialize chains
        if not self.initialize_chains():
            logger.error("Failed to reinitialize chains after reset")
            return False
        
        # Restore saved layers if any
        if saved_layers:
            logger.info(f"Restoring {len(saved_layers)} layer blocks...")
            restored = 0
            for layer_data in saved_layers:
                try:
                    self.chains['B'].add_block(
                        layer_data['payload'],
                        block_type='layer',
                        layer_id=layer_data['layer_id']
                    )
                    restored += 1
                except Exception as e:
                    logger.warning(f"Could not restore layer {layer_data['layer_id']}: {e}")
            logger.info(f"✅ Restored {restored}/{len(saved_layers)} layer blocks")
        
        return True
    
    def continue_block_building(self, progress: dict):
        """Continue building missing blocks."""
        if progress["progress_percentage"] >= 100:
            logger.info("✅ All blocks already present")
            return
        
        if not progress["integrity_valid"]:
            logger.warning("⚠️ Integrity issue detected, but continuing...")
            # Don't block upload for GPU nodes - they maintain independent chains
        
        # Calculate what's needed
        total_needed = progress["expected_layers"]
        current_count = progress["layer_blocks"]
        
        if current_count == 0:
            logger.info(f"📦 Starting fresh - need to upload {total_needed} layers")
        else:
            remaining = total_needed - current_count
            logger.info(f"📦 Resuming - need {remaining} more layers ({progress['progress_percentage']:.1f}% complete)")
        
        # Check if we should auto-upload
        if AUTO_UPLOAD and ALLOW_HF_UPLOAD:
            # Check if upload was already completed (verify blocks exist)
            upload_state_file = DATA_DIR / "upload_completed.json"
            actual_blocks = progress.get('layer_blocks', 0)
            if upload_state_file.exists() and actual_blocks > 0:
                logger.info("✅ Upload already completed (found state file)")
            elif upload_state_file.exists() and actual_blocks == 0:
                logger.warning("⚠️ State file exists but no blocks found - removing invalid state")
                upload_state_file.unlink()
            elif progress["progress_percentage"] >= 99.0:
                # If we're 99%+ complete, consider it done (may have minor discrepancies)
                logger.info(f"✅ Model upload essentially complete ({progress['progress_percentage']:.1f}%)")
                # Mark as complete to prevent re-upload
                upload_state = {
                    "model": MODEL_NAME,
                    "num_layers": current_count,
                    "timestamp": time.time(),
                    "completed": True,
                    "version": "dense-v1"
                }
                with open(upload_state_file, 'w') as f:
                    json.dump(upload_state, f)
            elif current_count < total_needed:
                logger.info("🚀 Starting auto-upload of model to blockchain...")
                # Mark that upload was triggered from block building
                self._upload_triggered_from_block_building = True
                asyncio.create_task(self.download_and_upload_model())
            else:
                logger.info("✅ Model fully uploaded")
        else:
            if AUTO_UPLOAD and not ALLOW_HF_UPLOAD:
                logger.info("ℹ️  AUTO_UPLOAD requested but ALLOW_HF_UPLOAD is false in production. Skipping HF download.")
            else:
                logger.info(f"ℹ️  Manual upload required (set AUTO_UPLOAD=true and ALLOW_HF_UPLOAD=true to enable bootstrap)")
    
    def initialize_model_manager(self) -> bool:
        """Initialize blockchain-first model manager for inference."""
        try:
            logger.info("📋 Starting model manager initialization...")
            
            # Use blockchain-first loader - NO local models
            logger.info("  1/5: Importing unified model manager...")
            try:
                from backend.model.manager import get_model_manager
                logger.info("    ✓ Model manager accessor imported")
            except ImportError as e:
                logger.error(f"    ✗ Failed to import model manager accessor: {e}")
                return False
            
            try:
                from backend.core.param_index import ParameterIndex
                logger.info("    ✓ ParameterIndex imported")
            except ImportError as e:
                logger.error(f"    ✗ Failed to import ParameterIndex: {e}")
                return False
            
            try:
                from backend.core.zero_copy_loader import ZeroCopyTileLoader
                logger.info("    ✓ ZeroCopyTileLoader imported")
            except ImportError as e:
                logger.error(f"    ✗ Failed to import ZeroCopyTileLoader: {e}")
                return False
            
            # Initialize parameter index
            logger.info("  2/5: Loading parameter index...")
            param_index = ParameterIndex(DATA_DIR / "param_index.json")
            # Auto-rebuild if checksum mismatch detected
            try:
                rebuilt = param_index.rebuild_if_corrupted(self.chains.get('B'))
                if rebuilt:
                    logger.warning("    ⚠️ Param index was corrupted and has been rebuilt")
            except Exception as e:
                logger.warning(f"    ⚠️ Could not verify/rebuild param index: {e}")
            
            # Check blockchain compatibility
            logger.info("  3/5: Checking blockchain compatibility...")
            try:
                from backend.core.migration_helper import check_blockchain_compatibility, print_blockchain_status
                if check_blockchain_compatibility(DATA_DIR):
                    logger.info("    ✓ Blockchain compatible with dense model")
                else:
                    logger.warning("    ⚠️ Blockchain may have compatibility issues")
                    print_blockchain_status(DATA_DIR)
            except Exception as e:
                logger.warning(f"    Could not check compatibility: {e}")
            
            # Validate critical components if STRICT_MODEL_LOAD is enabled
            if os.getenv("STRICT_MODEL_LOAD", "false").lower() == "true":
                logger.info("  3.5/5: Validating critical model components...")
                critical_components = ["embedding", "lm_head", "model_norm"]
                missing_critical = []
                
                for component in critical_components:
                    if component not in param_index.get_all_layers():
                        missing_critical.append(component)
                
                if missing_critical:
                    logger.error(f"    ✗ CRITICAL: Missing essential components: {missing_critical}")
                    logger.error(f"    Cannot serve model without: embedding, lm_head, and model_norm")
                    logger.error(f"    Please ensure these components are uploaded to blockchain")
                    if os.getenv("STRICT_MODEL_LOAD", "false").lower() == "true":
                        return False  # Refuse to start
                else:
                    logger.info(f"    ✓ All critical components present in param_index")
            
            # Initialize unified model manager
            logger.info("  4/5: Creating unified model manager...")
            
            # Log what we have available (reuse same index)
            param_layers = param_index.get_all_layers()
            logger.info(f"     📊 Parameter index: {len(param_layers)} layers")
            
            if len(param_layers) > 0:
                logger.info(f"     ✅ Layers available: {', '.join(param_layers[:3])}...")
                if len(param_layers) >= 38:
                    logger.info(f"     🎯 Full model in blockchain - no HF needed")
            else:
                logger.info(f"     ⚠️ No layers in param_index - will need upload")
            
            # Check environment settings
            logger.info(f"     🔧 Environment:")
            logger.info(f"        TRANSFORMERS_OFFLINE={os.getenv('TRANSFORMERS_OFFLINE', 'false')}")
            logger.info(f"        ENABLE_FUSED_SNAPSHOT={os.getenv('ENABLE_FUSED_SNAPSHOT', 'true')}")
            logger.info(f"        BLOCK_FETCH_MAX_WORKERS={os.getenv('BLOCK_FETCH_MAX_WORKERS', '4')}")
            
            # Enforce singleton model manager to prevent multiple loads
            self.model_manager = get_model_manager(
                DATA_DIR,
                force_new=False,
                model_name=MODEL_NAME,
                device="cuda" if self.gpu_available else "cpu",
                use_blockchain=True,
                use_gpu_direct=os.getenv("USE_GPU_DIRECT", "true").lower() == "true"
            )
            
            # Initialize zero-copy loader for efficient loading
            logger.info("  5/5: Setting up zero-copy loader...")
            self.zero_copy_loader = ZeroCopyTileLoader(
                chain=self.chains.get('B'),
                cache_dir=DATA_DIR / "tile_cache"
            )
            
            # Check for available experts in blockchain
            logger.info("  5/5: Checking available experts...")
            available_experts = self.model_manager.get_available_experts()
            
            # Use cached block count instead of loading all blocks
            actual_block_count = 0
            try:
                chain_b = self.chains.get('B')
                if chain_b and hasattr(chain_b, '_hash_index'):
                    actual_block_count = len(chain_b._hash_index)
                    logger.info(f"    Block count from index: {actual_block_count}")
                else:
                    logger.warning("    Chain B has no index, skipping block count")
            except Exception as e:
                logger.warning(f"    Error getting block count: {e}")
            
            if available_experts or actual_block_count > 100:  # If we have experts OR many blocks
                logger.info(f"✅ {len(available_experts)} experts loaded from blockchain (total blocks: {actual_block_count})")
                if len(available_experts) > 5:
                    logger.info(f"  Showing first 5 experts:")
                    for expert in available_experts[:5]:
                        logger.info(f"    - {expert}")
                else:
                    for expert in available_experts:
                        logger.info(f"    - {expert}")
            else:
                logger.info(f"📦 No experts in blockchain yet (blocks: {actual_block_count})")
                if AUTO_UPLOAD and ALLOW_HF_UPLOAD:
                    # Check if upload was already triggered from block building
                    if hasattr(self, '_upload_triggered_from_block_building') and self._upload_triggered_from_block_building:
                        logger.info("📋 Upload already triggered from block building phase")
                    elif hasattr(self, '_upload_in_progress') and self._upload_in_progress:
                        logger.info("📋 Upload already in progress")
                    else:
                        # Check if upload was already done (verify blocks exist)
                        upload_state_file = DATA_DIR / "upload_completed.json" 
                        if upload_state_file.exists() and actual_block_count > 0:
                            logger.info("✅ Upload already completed (found state file)")
                        elif upload_state_file.exists() and actual_block_count == 0:
                            logger.warning("⚠️ State file exists but no blocks - will upload")
                            upload_state_file.unlink()
                            logger.info("🚀 Auto-uploading model to blockchain...")
                            # Create task to download and upload model
                            asyncio.create_task(self.download_and_upload_model())
                        else:
                            logger.info("🚀 Auto-uploading model to blockchain...")
                            # Create task to download and upload model
                            asyncio.create_task(self.download_and_upload_model())
                else:
                    if AUTO_UPLOAD and not ALLOW_HF_UPLOAD:
                        logger.info("ℹ️  AUTO_UPLOAD requested but ALLOW_HF_UPLOAD=false. Skipping HF bootstrap in production.")
                    logger.info("💡 Upload model using: python miner/upload_moe_parameters.py or pre-seed chain data")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            return False

    async def _warmup_gpu(self, timeout_seconds: int = 180) -> None:
        """Warm up model on GPU without blocking server startup.
        Loads tokenizer, allocates model, and runs a minimal 1-token generation.
        """
        try:
            import asyncio
            from backend.model.manager import get_model_manager
            import os
            import time as _time
            import torch
            # Skip if blockchain is not ready for full model load
            ready, have, need = self._blockchain_model_readiness()
            if not ready:
                self._warmup_status = "skipped"
                logger.warning(f"GPU warmup skipped: blockchain not ready ({have}/{need} components)")
                return
            if not self.model_manager:
                # Ensure global manager exists
                self.model_manager = get_model_manager(DATA_DIR)
            self._warmup_status = "starting"
            logger.info("🔥 Eager warmup starting...")

            async def _do_warmup():
                try:
                    # Minimal deterministic prompt to materialize weights and KV
                    prompt = os.getenv("WARMUP_PROMPT", "Warmup")
                    tokens = int(os.getenv("WARMUP_TOKENS", "1"))
                    # Track VRAM
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    t0 = _time.time()
                    await self.model_manager.generate_async(prompt=prompt, max_new_tokens=tokens, temperature=0.0)
                    dt = _time.time() - t0
                    peak_gb = None
                    cur_gb = None
                    if torch.cuda.is_available():
                        cur_gb = torch.cuda.memory_allocated()/(1024**3)
                        peak_gb = torch.cuda.max_memory_allocated()/(1024**3)
                    logger.info(f"✅ Eager warmup complete in {dt:.2f}s; VRAM current={cur_gb:.2f}GB peak={peak_gb:.2f}GB")
                    # Log approximate breakdown for awareness (no hard budget)
                    try:
                        p0 = next(self.model_manager.model.parameters())
                        dtype_size = p0.dtype.itemsize
                        total_elems = sum(p.numel() for p in self.model_manager.model.parameters())
                        weights_gb = (total_elems * dtype_size)/(1024**3)
                    except Exception:
                        weights_gb = None
                    try:
                        from config.model_profile import calculate_kv_cache_size
                        kv_est = calculate_kv_cache_size(batch_size=1, seq_len=2048, num_layers=36)
                    except Exception:
                        kv_est = None
                    if peak_gb is not None and (weights_gb is not None or kv_est is not None):
                        temps_gb = None
                        if weights_gb is not None and kv_est is not None:
                            temps_gb = max(0.0, (peak_gb - weights_gb - kv_est))
                        logger.info(f"   VRAM breakdown est: weights={weights_gb if weights_gb is not None else 'n/a'}GB, kv={kv_est if kv_est is not None else 'n/a'}GB, temp/frag={temps_gb if temps_gb is not None else 'n/a'}GB")
                    self._warmup_status = "complete"
                except Exception as we:
                    # Detect CUDA OOM specifically (avoid shadowing torch in local scope)
                    try:
                        if 'torch' in globals():
                            # torch is imported in the outer scope of _warmup_gpu
                            if isinstance(we, torch.cuda.OutOfMemoryError):
                                self._handle_vram_oom("warmup")
                    except Exception:
                        pass
                    self._warmup_status = "skipped"
                    logger.warning(f"GPU warmup skipped: {we}")

            await asyncio.wait_for(_do_warmup(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self._warmup_status = "timeout"
            logger.warning(f"GPU warmup timed out after {timeout_seconds}s - continuing")
        except Exception as e:
            self._warmup_status = "failed"
            logger.warning(f"GPU warmup failed: {e}")
    
    async def download_and_upload_model(self):
        """Download and upload model to blockchain."""
        if hasattr(self, '_upload_in_progress') and self._upload_in_progress:
            return
        
        self._upload_in_progress = True
        try:
            await self._do_download_and_upload()
        finally:
            self._upload_in_progress = False
    
    async def _do_download_and_upload(self):
        """Download dense model and upload layers to blockchain - OPTIMIZED."""
        # Respect production policy: do not touch HF unless explicitly allowed
        if not ALLOW_HF_UPLOAD:
            logger.info("⛔ HF bootstrap disabled (ALLOW_HF_UPLOAD=false). Skipping download/upload.")
            return
        
        # OPTIMIZATION: Check parameter index first (fast)
        from backend.core.param_index import ParameterIndex
        param_index = ParameterIndex(DATA_DIR / "param_index.json")
        existing_layers = param_index.get_all_layers()
        
        # Check if we already have the model uploaded (including ALL required components)
        # Note: other_weights is optional - only required if model has extra tensors
        expected_layers = ["embedding"] + [f"layer_{i}" for i in range(36)] + ["lm_head", "model_norm"]
        
        if os.getenv("SKIP_UPLOAD_IF_PARAM_INDEX_MATCHES", "true").lower() == "true":
            if set(existing_layers) >= set(expected_layers):
                logger.info(f"✅ Model already uploaded - {len(existing_layers)} layers in param_index")
                logger.info("💡 Set SKIP_UPLOAD_IF_PARAM_INDEX_MATCHES=false to force upload")
                return
        
        # Check upload state file for additional verification
        upload_state_file = DATA_DIR / "upload_completed.json"
        if upload_state_file.exists():
            try:
                with open(upload_state_file, 'r') as f:
                    upload_state = json.load(f)
                
                if upload_state.get("completed") and upload_state.get("model") == MODEL_NAME:
                    if upload_state.get("num_layers", 0) == 38:  # 36 layers + embedding + lm_head
                        logger.info(f"✅ Model {MODEL_NAME} already uploaded per state file")
                        return
            except Exception as e:
                logger.warning(f"Failed to read upload state: {e}")
        
        logger.info(f"📥 Auto-downloading model: {MODEL_NAME}")
        model_config = get_model_config()
        logger.info(f"⚙️  Model: {model_config.get('total_params', 'unknown')} total params, {model_config.get('active_params', 'unknown')} active params per token")
        
        try:
            import torch
            import gc
            
            # Use compatibility layer if available
            if COMPAT_AVAILABLE:
                # Setup HF cache
                setup_hf_cache((DATA_DIR / ".hf").resolve())
                
                # Detect capabilities
                caps = detect_capabilities()
                
                # Check if we can use quantization
                if not caps['cuda']:
                    logger.error("CUDA not available on this node. Skipping auto-upload.")
                    return
                
                # No quantization check needed - BF16 only
                if not caps['supports_bf16']:
                    logger.error("GPU does not support BF16. Cannot proceed.")
                    logger.error("Minimum compute capability 8.0 required (Ampere or newer).")
                    return
                
                # Load tokenizer with compatibility handling
                logger.info("Loading tokenizer...")
                tokenizer = load_tokenizer(MODEL_NAME)
                
                # Load with model-specific configuration
                model_config = get_model_config()
                logger.info(f"Loading model with {model_config.get('precision', 'auto')} precision...")
                
                # Force BF16 loading ONLY - no fallbacks
                from transformers import AutoModelForCausalLM
                import torch
                
                # Verify BF16 support
                if not self.gpu_available:
                    logger.error("CUDA not available. BF16 requires GPU with compute capability 8.0+")
                    return
                
                if torch.cuda.get_device_capability()[0] < 8:
                    logger.error(f"GPU compute capability {torch.cuda.get_device_capability()} does not support BF16.")
                    logger.error("Minimum compute capability 8.0 required (Ampere or newer).")
                    return
                
                # Multi-GPU support with device_map="auto"
                num_gpus = self.gpu_info.get("num_gpus", 1)
                logger.info(f"Using {num_gpus} GPU(s) for model loading")
                if num_gpus > 1:
                    logger.info("Model will be distributed across all available GPUs")
                
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,  # ALWAYS BF16 - no fallbacks
                    device_map="auto",  # Auto distributes across all GPUs
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                logger.info(f"✅ Model loaded with BF16 precision (REQUIRED - no fallbacks)")
                
            else:
                # Fallback to original loading method
                if not torch.cuda.is_available() or not self.gpu_available:
                    logger.error("CUDA not available on this node. Skipping auto-upload.")
                    return
                
                # Setup environment
                os.environ.setdefault("HF_HOME", str((DATA_DIR / ".hf").resolve()))
                os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
                
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # BF16 loading ONLY - no fallbacks
                logger.info("⏳ Loading model in BF16...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                
                # Verify BF16 support
                if not self.gpu_available or not torch.cuda.is_available():
                    logger.error("CUDA not available. BF16 requires GPU with compute capability 8.0+")
                    return
                
                if torch.cuda.get_device_capability()[0] < 8:
                    logger.error(f"GPU compute capability {torch.cuda.get_device_capability()} does not support BF16.")
                    logger.error("Minimum compute capability 8.0 required (Ampere or newer).")
                    return
                
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.bfloat16,  # ALWAYS BF16 - no fallbacks
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                    
                logger.info(f"✅ Model loaded with BF16 precision (REQUIRED - no fallbacks)")
            
            # Create meta block if needed
            meta_count = len(self.chains['A']._hash_index) if hasattr(self.chains['A'], '_hash_index') else 0
            if meta_count == 0:
                if PROFILE_AVAILABLE:
                    # Compute total params from config (supports *_b fields)
                    total_params = ARCHITECTURE.get("total_params")
                    if total_params is None:
                        try:
                            total_params_b = float(ARCHITECTURE.get("total_params_b", 0))
                            total_params = int(total_params_b * 1e9)
                        except Exception:
                            total_params = 0
                    meta_spec = {
                        "model_name": MODEL_NAME,
                        "architecture": "dense",
                        "num_layers": 36,
                        "hidden_size": LAYERS["hidden_size"],
                        "num_attention_heads": LAYERS["num_attention_heads"],
                        "num_kv_heads": LAYERS["num_kv_heads"],
                        "context_length": CONTEXT["max_length"],
                        "total_params": total_params
                    }
                self.chains['A'].add_block(json.dumps(meta_spec).encode(), block_type='meta')
                logger.info("✅ Created meta block")
            
            # Extract and upload dense model layers with zero-copy streaming
            import io, gc, torch, hashlib
            from backend.core.param_index import ParameterIndex

            num_uploaded = 0
            
            # Get model state dict - keep on GPU for zero-copy
            state_dict = model.state_dict()
            
            # Expected number of layers (36 layers + embedding + lm_head + model_norm)
            num_layers = 36  # Dense model has 36 layers
            expected_names = ["embedding"] + [f"layer_{i}" for i in range(num_layers)] + ["lm_head", "model_norm"]
            # Note: other_weights will be added dynamically if there are remaining keys
            
            logger.info(f"📦 Uploading dense model: {num_layers} layers + embedding + lm_head")
            
            # Initialize parameter index
            param_index = ParameterIndex(DATA_DIR / "param_index.json")
            uploaded_names = []
            
            # 1. Upload embedding layer with metadata
            embedding_keys = [k for k in state_dict.keys() if 'embed_tokens' in k]
            if embedding_keys:
                try:
                    # Stream tensors directly - no CPU copy
                    buffer = io.BytesIO()
                    tensors_to_save = {}
                    for key in embedding_keys:
                        # ENSURE BFLOAT16 before saving to blockchain
                        tensor = state_dict[key]
                        if tensor.dtype != torch.bfloat16:
                            tensor = tensor.to(torch.bfloat16)
                        tensors_to_save[key] = tensor
                    
                    torch.save(tensors_to_save, buffer)
                    payload = buffer.getvalue()
                    
                    # Calculate content hash for integrity
                    content_hash = hashlib.sha256(payload).hexdigest()[:16]
                    
                    # Add block to chain
                    block = self.chains['B'].add_block(
                        payload,
                        block_type='dense_layer',  # Standardized block type
                        layer_name="embedding"  # Use layer_name instead of expert_name
                    )
                    
                    # Update parameter index with block index
                    param_index.set("embedding", block.header.index)
                    uploaded_names.append("embedding")
                    
                    logger.info(f"✅ Uploaded embedding ({len(payload) / 1024 / 1024:.1f} MB, hash: {content_hash})")
                    num_uploaded += 1
                    
                    # Cleanup
                    del tensors_to_save, buffer, payload
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to upload embedding: {e}")
            
            # 2. Upload each transformer layer with streaming
            for layer_idx in range(num_layers):
                layer_keys = [k for k in state_dict.keys() if f'layers.{layer_idx}.' in k]
                
                if not layer_keys:
                    logger.warning(f"Layer {layer_idx} has no weights - skipping")
                    continue
                
                try:
                    # Stream layer weights directly from GPU
                    buffer = io.BytesIO()
                    tensors_to_save = {}
                    for key in layer_keys:
                        # ENSURE BFLOAT16 before saving to blockchain
                        tensor = state_dict[key]
                        if tensor.dtype != torch.bfloat16:
                            tensor = tensor.to(torch.bfloat16)
                        tensors_to_save[key] = tensor
                    
                    torch.save(tensors_to_save, buffer)
                    payload = buffer.getvalue()
                    
                    # Calculate content hash
                    content_hash = hashlib.sha256(payload).hexdigest()[:16]
                    
                    # Upload to blockchain
                    layer_name = f"layer_{layer_idx}"
                    block = self.chains['B'].add_block(
                        payload,
                        block_type='dense_layer',  # Standardized block type
                        layer_name=layer_name  # Use layer_name instead of expert_name
                    )
                    
                    # Update parameter index with block index
                    param_index.set(layer_name, block.header.index)
                    uploaded_names.append(layer_name)
                    
                    logger.info(f"✅ Uploaded layer {layer_idx} ({len(payload) / 1024 / 1024:.1f} MB, hash: {content_hash})")
                    num_uploaded += 1
                    
                    # Cleanup
                    del tensors_to_save, buffer, payload
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to upload layer {layer_idx}: {e}")
                    continue
            
            # 3. Upload LM head with metadata
            lm_head_keys = [k for k in state_dict.keys() if 'lm_head' in k]
            if lm_head_keys:
                try:
                    # Stream directly from GPU
                    buffer = io.BytesIO()
                    tensors_to_save = {}
                    for key in lm_head_keys:
                        # ENSURE BFLOAT16 before saving to blockchain
                        tensor = state_dict[key]
                        if tensor.dtype != torch.bfloat16:
                            tensor = tensor.to(torch.bfloat16)
                        tensors_to_save[key] = tensor
                    
                    torch.save(tensors_to_save, buffer)
                    payload = buffer.getvalue()
                    
                    # Calculate content hash
                    content_hash = hashlib.sha256(payload).hexdigest()[:16]
                    
                    
                    block = self.chains['B'].add_block(
                        payload,
                        block_type='dense_layer',  # Standardized block type
                        layer_name="lm_head"  # Use layer_name instead of expert_name
                    )
                    
                    # Update parameter index with block index
                    param_index.set("lm_head", block.header.index)
                    uploaded_names.append("lm_head")
                    
                    logger.info(f"✅ Uploaded LM head ({len(payload) / 1024 / 1024:.1f} MB, hash: {content_hash})")
                    num_uploaded += 1
                    
                    # Cleanup
                    del tensors_to_save, buffer, payload
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to upload LM head: {e}")
            
            # 4. Upload model.norm (final layer normalization) - CRITICAL FOR OUTPUT
            norm_keys = [k for k in state_dict.keys() if 'norm' in k and 'layers.' not in k]
            if norm_keys:
                logger.info(f"📦 Found norm keys to upload: {norm_keys}")
                try:
                    buffer = io.BytesIO()
                    tensors_to_save = {}
                    for key in norm_keys:
                        # ENSURE BFLOAT16 before saving to blockchain
                        tensor = state_dict[key]
                        if tensor.dtype != torch.bfloat16:
                            tensor = tensor.to(torch.bfloat16)
                        tensors_to_save[key] = tensor
                    
                    torch.save(tensors_to_save, buffer)
                    payload = buffer.getvalue()
                    
                    # Calculate content hash
                    content_hash = hashlib.sha256(payload).hexdigest()[:16]
                    
                    block = self.chains['B'].add_block(
                        payload,
                        block_type='dense_layer',
                        layer_name="model_norm"
                    )
                    
                    # Update parameter index with block index
                    param_index.set("model_norm", block.header.index)
                    uploaded_names.append("model_norm")
                    
                    logger.info(f"✅ Uploaded model norm ({len(payload) / 1024 / 1024:.1f} MB, hash: {content_hash})")
                    num_uploaded += 1
                    
                    # Cleanup
                    del tensors_to_save, buffer, payload
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to upload model norm: {e}")
            else:
                logger.warning("⚠️ No model.norm keys found - model will produce garbled output!")
            
            # 5. Upload any remaining keys (rotary embeddings, etc.) as "other_weights"
            # These are essential for proper model operation
            uploaded_keys = set()
            for name in uploaded_names:
                if name == "embedding":
                    uploaded_keys.update([k for k in state_dict.keys() if 'embed_tokens' in k])
                elif name == "lm_head":
                    uploaded_keys.update([k for k in state_dict.keys() if 'lm_head' in k])
                elif name == "model_norm":
                    uploaded_keys.update([k for k in state_dict.keys() if 'norm' in k and 'layers.' not in k])
                elif name.startswith("layer_"):
                    layer_idx = int(name.split("_")[1])
                    uploaded_keys.update([k for k in state_dict.keys() if f'layers.{layer_idx}.' in k])
            
            remaining_keys = set(state_dict.keys()) - uploaded_keys
            if remaining_keys:
                logger.warning(f"⚠️ Found {len(remaining_keys)} unhandled keys: {list(remaining_keys)[:5]}...")
                # Only add other_weights to expected if there are actually remaining keys
                expected_names.append("other_weights")
                try:
                    buffer = io.BytesIO()
                    tensors_to_save = {}
                    for key in remaining_keys:
                        # ENSURE BFLOAT16 before saving to blockchain
                        tensor = state_dict[key]
                        if tensor.dtype != torch.bfloat16:
                            tensor = tensor.to(torch.bfloat16)
                        tensors_to_save[key] = tensor
                    
                    torch.save(tensors_to_save, buffer)
                    payload = buffer.getvalue()
                    
                    # Calculate content hash
                    content_hash = hashlib.sha256(payload).hexdigest()[:16]
                    
                    block = self.chains['B'].add_block(
                        payload,
                        block_type='dense_layer',
                        layer_name="other_weights"
                    )
                    
                    # Update parameter index with block index
                    param_index.set("other_weights", block.header.index)
                    uploaded_names.append("other_weights")
                    
                    logger.info(f"✅ Uploaded other weights ({len(payload) / 1024 / 1024:.1f} MB, hash: {content_hash})")
                    logger.info(f"   Keys included: {list(remaining_keys)}")
                    num_uploaded += 1
                    
                    # Cleanup
                    del tensors_to_save, buffer, payload
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to upload other weights: {e}")
            
            # Parameter index auto-saves on each set() call
            
            # Verify completeness by names, not percentages
            missing_components = set(expected_names) - set(uploaded_names)
            if missing_components:
                logger.error(f"❌ Missing components: {missing_components}")
                logger.error("Upload incomplete - some components failed")
            else:
                logger.info(f"✅ All expected components uploaded: {len(uploaded_names)}/{len(expected_names)}")
            
            
            # Final verification - already done above
            logger.info(f"✅ Upload complete: {num_uploaded} blocks uploaded")
            logger.info(f"📊 Uploaded components: {uploaded_names}")
            
            # Save upload state
            upload_complete = len(missing_components) == 0
            
            if upload_complete:
                # Save completion state
                upload_state = {
                    "completed": True,
                    "model": MODEL_NAME,
                    "num_layers": len(uploaded_names),
                    "timestamp": time.time(),
                    "components": uploaded_names
                }
                with open(upload_state_file, 'w') as f:
                    json.dump(upload_state, f, indent=2)
                logger.info("✅ Upload state saved")
            
            if not upload_complete:
                logger.error(f"⚠️ Upload incomplete - missing: {missing_components}")
            
            # Save upload state with detailed component list
            upload_state_file = DATA_DIR / "upload_completed.json"
            upload_state = {
                "model": MODEL_NAME,
                "num_layers": num_layers,
                "uploaded_components": uploaded_names,
                "expected_components": expected_names,
                "timestamp": time.time(),
                "completed": upload_complete,
                "version": "dense-v1"
            }
            with open(upload_state_file, 'w') as f:
                json.dump(upload_state, f)
            logger.info(f"💾 Saved upload state to {upload_state_file}")
            
            # Clean up model from memory after upload
            logger.info("🧹 Cleaning up model from memory...")
            del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # DO NOT reinitialize model manager here - it causes a loop!
            # Set a flag to indicate upload is complete but model manager needs init
            self.upload_completed = True
            logger.info("✅ Upload complete. Model manager will reinitialize automatically.")
            
        except Exception as e:
            logger.error(f"Failed to download/upload model: {e}")
            # Clean up on error too
            if 'model' in locals():
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _cleanup_port(self, port: int, force: bool = False) -> bool:
        """
        Clean up a specific port by killing ONLY the process using it.
        Essential for Vast.ai where we must bind to exact port.
        
        Args:
            port: Port number to cleanup
            force: If True, cleanup even if not strictly required
            
        Returns:
            True if port is free (either was free or successfully cleaned)
        """
        import subprocess
        import signal
        import shutil
        
        # Check if we have the tools available (in order of preference)
        has_lsof = shutil.which('lsof') is not None
        has_ss = shutil.which('ss') is not None  # Most common in minimal containers
        has_fuser = shutil.which('fuser') is not None
        has_netstat = shutil.which('netstat') is not None
        
        if not any([has_lsof, has_ss, has_fuser, has_netstat]):
            logger.warning(f"⚠️ No port inspection tools available")
            logger.warning(f"   Cannot automatically free port {port}")
            logger.warning(f"   Install one of:")
            logger.warning(f"   - apt-get install iproute2    (for ss - recommended)")
            logger.warning(f"   - apt-get install lsof         (for lsof)")
            logger.warning(f"   - apt-get install psmisc       (for fuser)")
            logger.warning(f"   - apt-get install net-tools    (for netstat)")
            
            # If we absolutely need this port, fail early
            if force:
                raise RuntimeError(f"Port cleanup tools not available, cannot ensure port {port} is free")
            return False
        
        try:
            # Collect all PIDs using the port (handles IPv4/IPv6)
            pids_to_kill = set()
            current_pid = os.getpid()
            
            # Try tools in order of preference and availability
            if has_lsof:
                logger.info(f"🔍 Checking port {port} with lsof...")
                # Check both IPv4 and IPv6
                for protocol in ['4', '6']:
                    result = subprocess.run(
                        ['lsof', '-nP', f'-i{protocol}TCP:{port}', '-sTCP:LISTEN'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.stdout:
                        lines = result.stdout.strip().split('\n')
                        for line in lines[1:]:  # Skip header
                            parts = line.split()
                            if len(parts) > 1 and parts[1].isdigit():
                                pid = int(parts[1])
                                if pid != current_pid:
                                    pids_to_kill.add((pid, parts[0]))  # (pid, command)
            
            elif has_ss:
                logger.info(f"🔍 Checking port {port} with ss (most reliable in containers)...")
                # ss is more reliable in containers than netstat
                result = subprocess.run(
                    ['ss', '-tlnp'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if f':{port} ' in line and 'LISTEN' in line:
                            # Parse ss output: users:(("process",pid=12345,fd=3))
                            import re
                            pid_match = re.search(r'pid=(\d+)', line)
                            proc_match = re.search(r'\("([^"]+)"', line)
                            if pid_match:
                                pid = int(pid_match.group(1))
                                proc = proc_match.group(1) if proc_match else 'unknown'
                                if pid != current_pid:
                                    pids_to_kill.add((pid, proc))
            
            elif has_fuser:
                logger.info(f"🔍 Checking port {port} with fuser...")
                result = subprocess.run(
                    ['fuser', f'{port}/tcp'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.stdout:
                    for pid_str in result.stdout.strip().split():
                        if pid_str.isdigit():
                            pid = int(pid_str)
                            if pid != current_pid:
                                pids_to_kill.add((pid, 'unknown'))
            
            if not pids_to_kill:
                logger.info(f"✅ Port {port} is free")
                return True
            else:
                logger.warning(f"⚠️ Port {port} is in use by {len(pids_to_kill)} process(es)")
                for pid, cmd in pids_to_kill:
                    logger.info(f"   • {cmd} (PID {pid})")

                # Kill each PID
                killed_count = 0
                for pid, command in pids_to_kill:
                    logger.info(f"   → Terminating {command} (PID {pid})...")

                    try:
                        # Try graceful termination first
                        logger.info(f"   → Sending SIGTERM to PID {pid}...")
                        os.kill(pid, signal.SIGTERM)

                        # Wait up to 3 seconds for graceful shutdown
                        for _ in range(30):  # 30 * 100ms = 3 seconds
                            await asyncio.sleep(0.1)
                            try:
                                os.kill(pid, 0)  # Check if still alive
                            except ProcessLookupError:
                                logger.info(f"   ✅ PID {pid} terminated gracefully")
                                break
                        else:
                            # Process still running after 3 seconds
                            logger.warning(f"   → Process still running, sending SIGKILL to PID {pid}...")
                            os.kill(pid, signal.SIGKILL)

                            # Wait briefly for force kill
                            for _ in range(10):  # 10 * 100ms = 1 second
                                await asyncio.sleep(0.1)
                                try:
                                    os.kill(pid, 0)
                                except ProcessLookupError:
                                    logger.info(f"   ✅ PID {pid} force killed")
                                    break
                            else:
                                logger.error(f"   ❌ Failed to kill PID {pid} - process may be protected")
                                return False

                    except PermissionError:
                        logger.error(f"❌ Permission denied to kill PID {pid}")
                        logger.error(f"   In container: This shouldn't happen - check container privileges")
                        logger.error(f"   Manual cleanup: kill -9 {pid}")
                        return False
                    except ProcessLookupError:
                        logger.info(f"   Process {pid} already terminated")

                    killed_count += 1

                # Wait for OS to release the port
                await asyncio.sleep(0.5)

                # Double-check port is now free (use fastest available tool)
                port_free = True
                if has_ss:
                    # ss is fastest
                    result = subprocess.run(
                        ['ss', '-tln'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.stdout and f':{port} ' in result.stdout:
                        port_free = False
                elif has_lsof:
                    for protocol in ['4', '6']:
                        result = subprocess.run(
                            ['lsof', '-nP', f'-i{protocol}TCP:{port}', '-sTCP:LISTEN'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.stdout:
                            port_free = False
                            break
                elif has_fuser:
                    result = subprocess.run(
                        ['fuser', f'{port}/tcp'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.stdout.strip():
                        port_free = False

                if port_free:
                    logger.info(f"✅ Port {port} is now free (cleaned {killed_count}/{len(pids_to_kill)} processes)")
                    return True
                else:
                    logger.error(f"❌ Port {port} is still in use after cleanup attempt")
                    logger.error(f"   Attempted to kill {killed_count} process(es)")
                    logger.error("   Manual intervention required")
                    return False
                
        except Exception as e:
            logger.error(f"⚠️ Port cleanup error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    async def start_server(self):
        """Start HTTP server for the node."""
        # We may update module-level PUBLIC_PORT when auto-selecting a free port
        global PUBLIC_PORT
        # One-time guard to avoid double-bind within the same process
        if getattr(self, 'server_started', False):
            logger.info("🟢 Server already started; skipping start_server()")
            return
        
        # Initialize runner/site references for clean shutdown
        self._runner = None
        self._site = None
        try:
            from aiohttp import web
        except ImportError:
            logger.info("Installing aiohttp...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp"])
            from aiohttp import web
        
        # Add CORS middleware
        from aiohttp import web
        try:
            import aiohttp_cors
            CORS_AVAILABLE = True
        except ImportError:
            logger.warning("aiohttp-cors not installed. Installing...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp-cors"])
            try:
                import aiohttp_cors
                CORS_AVAILABLE = True
            except ImportError:
                logger.warning("CORS support not available")
                CORS_AVAILABLE = False
        
        # Enforce request size limit to protect memory
        try:
            max_req_mb = float(os.getenv('MAX_REQUEST_SIZE_MB', '2'))
        except Exception:
            max_req_mb = 2.0
        client_max_size = int(max(0.5, min(max_req_mb, 16.0)) * 1024 * 1024)  # clamp 0.5MB..16MB
        app = web.Application(client_max_size=client_max_size)
        
        # Initialize queued prompts storage (shared reference for cleanup)
        app['queued_prompts'] = {}
        self._queued_prompts = app['queued_prompts']

        # Unified Retry-After configuration (seconds)
        try:
            retry_chat_cap = int(os.getenv('RETRY_AFTER_CHAT_CAPACITY_SECONDS', '5'))
        except Exception:
            retry_chat_cap = 5
        try:
            retry_queue_full = int(os.getenv('RETRY_AFTER_QUEUE_FULL_SECONDS', '30'))
        except Exception:
            retry_queue_full = 30
        try:
            retry_rate_limit = int(os.getenv('RETRY_AFTER_RATE_LIMIT_SECONDS', '10'))
        except Exception:
            retry_rate_limit = 10
        try:
            retry_start_cap = int(os.getenv('RETRY_AFTER_START_CAPACITY_SECONDS', '3'))
        except Exception:
            retry_start_cap = 3
        
        # Import queue exceptions
        from backend.runtime.queue_manager import QueueFull, LimitExceeded
        
        # Add request logging middleware
        @web.middleware
        async def log_middleware(request, handler):
            start_time = time.time()
            client_ip = request.remote
            method = request.method
            path = request.path
            
            # Log incoming request (skip health checks to reduce noise)
            if path != '/health':
                logger.info(f"→ {method} {path} from {client_ip}")
                
                # Log headers for debugging
                if path == '/chat':
                    origin = request.headers.get('Origin', 'none')
                    auth = 'yes' if request.headers.get('Authorization') else 'no'
                    logger.debug(f"   Headers: Origin={origin}, Auth={auth}")
            
            try:
                response = await handler(request)
                elapsed = time.time() - start_time
                
                # Log response
                if path != '/health':
                    if response.status < 400:
                        logger.info(f"← ✅ {response.status} for {path} ({elapsed:.2f}s)")
                    elif response.status == 429:
                        logger.warning(f"← ⚠️ 429 Rate Limited for {path} ({elapsed:.2f}s)")
                    elif response.status < 500:
                        logger.warning(f"← ⚠️ {response.status} for {path} ({elapsed:.2f}s)")
                    else:
                        logger.error(f"← ❌ {response.status} for {path} ({elapsed:.2f}s)")
                
                return response
            except Exception as ex:
                elapsed = time.time() - start_time
                # Handle too-large payloads explicitly
                try:
                    from aiohttp.web_exceptions import HTTPRequestEntityTooLarge
                    if isinstance(ex, HTTPRequestEntityTooLarge):
                        logger.warning(f"← ⚠️ 413 Payload Too Large for {path} ({elapsed:.2f}s)")
                        return web.json_response({
                            "error": "Payload too large",
                            "max_request_size_mb": round(client_max_size/(1024*1024), 2)
                        }, status=413)
                except Exception:
                    pass
                logger.error(f"← ❌ Exception for {path} ({elapsed:.2f}s): {ex}")
                raise
        
        app.middlewares.append(log_middleware)
        
        # Configure CORS if available
        if CORS_AVAILABLE:
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
        else:
            cors = None
        
        # Health endpoint
        async def health(request):
            client_ip = request.remote
            logger.debug(f"🩺 Health check from {client_ip}")
            # Collect detailed health
            finality_anchor_height = 0
            try:
                anchor_file = DATA_DIR / "finality_anchor_B.json"
                if anchor_file.exists():
                    with open(anchor_file, 'r') as f:
                        anchor_data = json.load(f)
                        finality_anchor_height = anchor_data.get('height', 0)
            except Exception:
                pass
            header_index_ok = (DATA_DIR / "B" / "headers.idx.jsonl").exists()
            tail_verified = int(os.getenv('TAIL_VERIFY_DEPTH', '128'))
            # Param index checksum quick check
            param_index_ok = True
            try:
                import hashlib
                idx = DATA_DIR / "param_index.json"
                sha = DATA_DIR / "param_index.sha256"
                if idx.exists() and sha.exists():
                    with open(idx, 'rb') as f:
                        h = hashlib.sha256()
                        for chunk in iter(lambda: f.read(4096), b''):
                            h.update(chunk)
                        actual = h.hexdigest()
                    expected = sha.read_text().strip()
                    param_index_ok = (actual == expected)
            except Exception:
                pass
            # Background auditor stats
            auditor_stats = {}
            if getattr(self, 'background_auditor', None):
                try:
                    auditor_stats = self.background_auditor.get_stats()
                except Exception:
                    pass
            degraded = (not self.vram_healthy) or self.degraded_mode or (self.model_manager is None)
            # Get queue metrics
            queue_info = {}
            try:
                if hasattr(self, 'queue_manager') and self.queue_manager:
                    queue_metrics = await self.queue_manager.metrics()
                    queue_info = {
                        "queue_length": queue_metrics.get('queue_length', 0),
                        "active_jobs": queue_metrics.get('active_jobs', 0),
                        "capacity": {
                            "total": self.job_capacity,
                            "used": self.active_jobs,
                            "available": self.job_capacity - self.active_jobs
                        }
                    }
            except Exception:
                pass
            
            response = {
                "status": "healthy" if not degraded else "degraded",
                "node_id": self.node_id,
                "gpu": self.gpu_info.get("name", "None"),
                "gpu_available": self.gpu_available,
                "chains": list(self.chains.keys()),
                "model_ready": self.model_manager is not None,
                "port": self.port,
                "degraded": degraded,
                "vram_ok": getattr(self, 'vram_healthy', True),
                "vram_oom_count": getattr(self, 'vram_oom_count', 0),
                "finality_anchor_height": finality_anchor_height,
                "header_index_ok": header_index_ok,
                "tail_verified": tail_verified,
                "param_index_ok": param_index_ok,
                "auditor": auditor_stats,
                "queue": queue_info,
                "max_request_size_mb": round(client_max_size/(1024*1024), 2)
            }
            
            # Log if model not ready (common issue)
            if not self.model_manager:
                logger.warning(f"⚠️ Health check: Model not ready for {client_ip}")
            
            return web.json_response(response)

        # Metrics endpoint: VRAM telemetry and warmup status
        async def metrics(request):
            info = {
                "node_id": self.node_id,
                "model_ready": bool(self.model_manager),
                "warmup_status": getattr(self, "_warmup_status", "idle"),
            }
            try:
                import torch
                if torch.cuda.is_available():
                    current = torch.cuda.memory_allocated() / (1024**3)
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    info.update({
                        "vram_current_gb": round(current, 3),
                        "vram_peak_gb": round(peak, 3),
                        "vram_reserved_gb": round(reserved, 3),
                    })
            except Exception:
                pass
            return web.json_response(info)
        
        # Chain info endpoint
        async def chain_info(request):
            chain_id = request.match_info.get('chain_id', 'A')
            if chain_id in self.chains:
                # For API endpoint, we do need actual blocks
                # But we can limit to recent blocks for performance
                chain = self.chains[chain_id]
                block_count = len(chain._hash_index) if hasattr(chain, '_hash_index') else 0
                
                # Only return last 100 blocks for performance
                blocks = []
                start_idx = max(0, block_count - 100)
                for i in range(start_idx, block_count):
                    if hasattr(chain, 'get_block_by_index'):
                        block = chain.get_block_by_index(i)
                    elif hasattr(chain, 'storage') and chain.storage:
                        block = chain.storage.get_block_by_index(i)
                    else:
                        block = None
                    if block:
                        blocks.append(block)
                
                return web.json_response({
                    "chain_id": chain_id,
                    "total_blocks": block_count,
                    "returned_blocks": len(blocks),
                    "blocks": len(blocks),
                    "latest_hash": blocks[-1]["hash"] if blocks else None
                })
            return web.json_response({"error": "Chain not found"}, status=404)
        
        # Debug endpoint for MoE status
        async def debug_moe_status(request):
            """Debug endpoint to check MoE system status"""
            return web.json_response({
                "moe_model_manager_initialized": self.moe_model_manager is not None if hasattr(self, 'moe_model_manager') else False,
                "distributed_coordinator_initialized": hasattr(self, 'distributed_coordinator') and self.distributed_coordinator is not None,
                "has_distributed_nodes": bool(self.distributed_coordinator.registry.nodes) if hasattr(self, 'distributed_coordinator') and self.distributed_coordinator else False,
                "available_experts_count": len(self.distributed_coordinator.registry.expert_to_nodes) if hasattr(self, 'distributed_coordinator') and self.distributed_coordinator else 0,
                "registered_nodes_count": len(self.distributed_coordinator.registry.nodes) if hasattr(self, 'distributed_coordinator') and self.distributed_coordinator else 0,
                "model_manager_initialized": self.model_manager is not None if hasattr(self, 'model_manager') else False,
                "usage_tracker_initialized": hasattr(self, 'usage_tracker') and self.usage_tracker is not None
            })

        # Chat endpoint with MoE support
        async def chat(request):
            """Chat endpoint with dense model inference"""
            try:
                import time
                start_time = time.time()
                
                # Log incoming request details
                client_ip = request.remote
                headers = dict(request.headers)
                logger.info(f"📥 Chat request from {client_ip}")
                logger.info(f"   Headers: Origin={headers.get('Origin', 'none')}, Auth={bool(headers.get('Authorization'))}")

                # Robust JSON parse
                try:
                    data = await request.json()
                except Exception:
                    return web.json_response({"error": "Invalid JSON body"}, status=400)
                prompt = data.get("prompt", "")
                # Basic input sanitation
                if not isinstance(prompt, str):
                    return web.json_response({"error": "prompt must be a string"}, status=400)
                max_new_tokens = data.get("max_new_tokens", 64)
                try:
                    max_new_tokens = int(max_new_tokens)
                except Exception:
                    return web.json_response({"error": "max_new_tokens must be an integer"}, status=400)
                # Clamp tokens to safe bounds
                if max_new_tokens < 1:
                    max_new_tokens = 1
                if max_new_tokens > 1024:
                    max_new_tokens = 1024
                
                # Parse sampling parameters with validation
                temperature = float(data.get("temperature", 0.7))
                temperature = max(0.0, min(2.0, temperature))  # Clamp to [0.0, 2.0]
                
                top_p = float(data.get("top_p", 0.9))
                top_p = max(0.0, min(1.0, top_p))  # Clamp to [0.0, 1.0]
                
                # Parse optional top_k
                top_k = data.get("top_k")
                if top_k is not None:
                    top_k = int(top_k)
                    top_k = max(1, top_k)  # Must be at least 1
                
                # Parse repetition control parameters
                repetition_penalty = float(data.get("repetition_penalty", 1.0))
                repetition_penalty = max(0.1, min(2.0, repetition_penalty))  # Clamp to reasonable range
                
                no_repeat_ngram_size = int(data.get("no_repeat_ngram_size", 0))
                no_repeat_ngram_size = max(0, min(10, no_repeat_ngram_size))  # Clamp to [0, 10]
                
                log_params = f"tokens: {max_new_tokens}, temp: {temperature}, top_p: {top_p}"
                if top_k is not None:
                    log_params += f", top_k: {top_k}"
                if repetition_penalty != 1.0:
                    log_params += f", rep_penalty: {repetition_penalty}"
                if no_repeat_ngram_size > 0:
                    log_params += f", no_repeat_ngram: {no_repeat_ngram_size}"
                logger.info(f"   Prompt: '{prompt[:50]}...' ({log_params})")

                # Check rate limiting (if implemented)
                if hasattr(request.app, 'rate_limiter'):
                    logger.info(f"   Rate limit check for {client_ip}")

                # Use dense model
                if not hasattr(self, 'model_manager') or self.model_manager is None:
                    logger.error(f"❌ Model not ready for {client_ip}")
                    return web.json_response({"error": "Model manager not initialized"}, status=503)
                
                # Check if we're at capacity and should queue
                if hasattr(self, 'job_capacity') and self.active_jobs >= self.job_capacity:
                    logger.info(f"⏳ At capacity ({self.active_jobs}/{self.job_capacity}), redirecting to queue")
                    response = web.json_response({
                        "error": "At capacity, please use /chat/admit to queue your request",
                        "queue_available": True,
                        "capacity": {
                            "total": self.job_capacity,
                            "used": self.active_jobs,
                            "available": 0
                        }
                    }, status=503)
                    response.headers['Retry-After'] = str(retry_chat_cap)
                    return response
                
                # Try to acquire a job slot
                if not await self.acquire_job_slot():
                    response = web.json_response({
                        "error": "Unable to acquire job slot, please use /chat/admit to queue",
                        "queue_available": True
                    }, status=503)
                    response.headers['Retry-After'] = str(retry_chat_cap)
                    return response

                # Generate response using dense model
                gen_log = f"   Generating response with temperature={temperature}, top_p={top_p}"
                if top_k is not None:
                    gen_log += f", top_k={top_k}"
                if repetition_penalty != 1.0:
                    gen_log += f", repetition_penalty={repetition_penalty}"
                if no_repeat_ngram_size > 0:
                    gen_log += f", no_repeat_ngram_size={no_repeat_ngram_size}"
                logger.info(gen_log + "...")
                
                try:
                    answer = self.model_manager.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size
                    )
                except Exception as ge:
                    # Release job slot on error
                    await self.release_job_slot()
                    
                    # Handle VRAM OOM gracefully
                    try:
                        import torch
                        if isinstance(ge, torch.cuda.OutOfMemoryError):
                            self._handle_vram_oom("chat")
                            return web.json_response({
                                "error": "GPU out of memory",
                                "degraded": True
                            }, status=503)
                        # Handle common invalid inputs gracefully
                        if isinstance(ge, (ValueError, AssertionError)):
                            return web.json_response({"error": str(ge)}, status=400)
                    except Exception:
                        pass
                    raise
                inference_time = time.time() - start_time
                
                # Release job slot after successful completion
                await self.release_job_slot()
                
                logger.info(f"✅ Chat success for {client_ip} - {inference_time:.2f}s")

                return web.json_response({
                    "response": answer,
                    "inference_time": inference_time,
                    "model": MODEL_NAME,
                    "mode": "dense"
                })

            except Exception as exc:
                # Release job slot on any error
                if hasattr(self, 'release_job_slot'):
                    await self.release_job_slot()
                
                logger.error(f"❌ Chat error for {request.remote}: {exc}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                return web.json_response({"error": str(exc)}, status=500)

        # Inference endpoint - BLOCKCHAIN-FIRST, no local models
        async def inference(request):
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
                if not isinstance(prompt, str):
                    return web.json_response({"error": "prompt must be a string"}, status=400)
                max_tokens = data.get("max_new_tokens", data.get("max_length", 100))
                try:
                    max_tokens = int(max_tokens)
                except Exception:
                    return web.json_response({"error": "max_new_tokens must be an integer"}, status=400)
                if max_tokens < 1:
                    max_tokens = 1
                if max_tokens > 1024:
                    max_tokens = 1024
                # Dense model doesn't need layer selection - uses all layers
                required_precision = data.get("precision", self.precision)
                
                # Validate precision requirement
                if required_precision != self.precision:
                    return web.json_response({
                        "error": f"Precision mismatch: requested {required_precision}, node enforces {self.precision}"
                    }, status=400)
                
                if not self.model_manager:
                    return web.json_response({
                        "error": "Model manager not initialized"
                    }, status=503)
                
                # For dense model, we don't need to select specific components
                # The model manager will handle the full model inference
                logger.info(f"Performing dense model inference")
                
                # Perform blockchain-first inference with dense model
                try:
                    response = await asyncio.to_thread(
                        self.model_manager.generate,
                        prompt,
                        max_tokens
                    )
                except Exception as ge:
                    try:
                        import torch
                        if isinstance(ge, torch.cuda.OutOfMemoryError):
                            self._handle_vram_oom("inference")
                            return web.json_response({
                                "error": "GPU out of memory",
                                "degraded": True
                            }, status=503)
                        if isinstance(ge, (ValueError, AssertionError)):
                            return web.json_response({"error": str(ge)}, status=400)
                    except Exception:
                        pass
                    raise
                
                # Track which layers were used (for dense model, all layers are used)
                from backend.model.dynamic_config import get_model_config
                model_config = get_model_config()
                layers_used = list(range(model_config.num_layers))
                
                # Generate blockchain proof for verifiable inference
                blockchain_proof = None
                if self.blockchain_mode and self.param_chain:
                    try:
                        from backend.inference.blockchain_proof import create_proof_for_inference
                        blockchain_proof = create_proof_for_inference(
                            chain=self.param_chain,
                            prompt=prompt,
                            response=response,
                            model_manager=self.model_manager
                        )
                        logger.info("✅ Generated blockchain proof for verifiable AI")
                    except Exception as e:
                        logger.warning(f"Failed to generate proof: {e}")
                
                return web.json_response({
                    "node_id": self.node_id,
                    "prompt": prompt,
                    "response": response,
                    "layers_used": [f"layer_{i}" for i in layers_used],
                    "blockchain_inference": True,
                    "blockchain_proof": blockchain_proof,  # Include proof
                    "gpu_used": self.gpu_available,
                    "precision": self.precision  # Report precision used
                })
                
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        
        # Pipeline stage endpoint for distributed inference
        async def inference_stage(request):
            """Handle pipeline stage inference for distributed processing."""
            try:
                data = await request.json()
                stage = data.get("stage")  # Which layers to process
                hidden_states = data.get("hidden_states")  # Input tensor
                if not self.model_manager:
                    return web.json_response({
                        "error": "Model manager not initialized"
                    }, status=503)
                
                # Process specific layers for pipeline parallelism
                if stage is None:
                    return web.json_response({
                        "error": "Stage information required"
                    }, status=400)
                
                logger.info(f"Processing pipeline stage: {stage}")
                
                # Convert hidden states to tensor - handle both JSON and binary
                import torch
                import base64
                import io
                
                serialization_type = data.get("serialization", "json")
                
                if serialization_type == "binary" and isinstance(hidden_states, str):
                    # Decode binary serialized tensor
                    buffer = io.BytesIO(base64.b64decode(hidden_states))
                    loaded = torch.load(buffer)
                    hidden_states = loaded['tensor']
                elif isinstance(hidden_states, list):
                    hidden_states = torch.tensor(hidden_states)
                elif isinstance(hidden_states, dict):
                    # Reconstruct tensor from serialized format
                    hidden_states = torch.tensor(hidden_states.get("data", []))
                
                # Process through specific layers
                try:
                    output = await asyncio.to_thread(
                        self._process_pipeline_stage,
                        stage,
                        hidden_states
                    )
                except Exception as ge:
                    try:
                        import torch
                        if isinstance(ge, torch.cuda.OutOfMemoryError):
                            self._handle_vram_oom("pipeline_stage")
                            return web.json_response({
                                "error": "GPU out of memory",
                                "degraded": True
                            }, status=503)
                    except Exception:
                        pass
                    raise
                
                # Serialize output for transport using binary format
                import base64
                import io
                
                if isinstance(output, torch.Tensor):
                    # Binary serialization for efficiency
                    buffer = io.BytesIO()
                    torch.save({
                        'tensor': output.cpu(),
                        'dtype': str(output.dtype),
                        'shape': list(output.shape)
                    }, buffer)
                    output_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    serialization_type = 'binary'
                else:
                    output_data = output
                    serialization_type = 'json'
                
                return web.json_response({
                    "node_id": self.node_id,
                    "stage": stage,
                    "output": output_data,
                    "serialization": serialization_type,
                    "shape": list(output.shape) if hasattr(output, 'shape') else None
                })
                
            except Exception as e:
                logger.error(f"Pipeline stage error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        def _process_pipeline_stage(self, stage, hidden_states):
            """Process a specific pipeline stage - REAL IMPLEMENTATION."""
            import torch
            
            if not hasattr(self.model_manager, 'model') or self.model_manager.model is None:
                raise RuntimeError("Model not loaded")
            
            model = self.model_manager.model
            
            # Get dynamic model configuration
            from backend.model.dynamic_config import get_model_config
            model_config = get_model_config()
            
            # Parse stage information
            if isinstance(stage, dict):
                layer_range = stage.get('layer_range', [0, model_config.num_layers])
                has_embedding = stage.get('has_embedding', False)
                has_lm_head = stage.get('has_lm_head', False)
            else:
                # Default to all layers if stage info not provided
                layer_range = [0, model_config.num_layers]
                has_embedding = False
                has_lm_head = False
            
            with torch.no_grad():
                # Ensure hidden_states is on the right device
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states = torch.tensor(hidden_states)
                
                device = torch.device('cuda' if self.gpu_available else 'cpu')
                if hidden_states.device != device:
                    hidden_states = hidden_states.to(device)
                
                # Process embedding layer if needed
                if has_embedding and hasattr(model, 'embed_tokens'):
                    # If we have token IDs, embed them
                    if hidden_states.dtype in [torch.int32, torch.int64]:
                        hidden_states = model.embed_tokens(hidden_states)
                
                # Process through transformer layers
                if hasattr(model, 'layers'):
                    start_layer = layer_range[0] if layer_range else 0
                    end_layer = layer_range[1] if len(layer_range) > 1 else len(model.layers)
                    
                    # Apply each layer in the range
                    for layer_idx in range(start_layer, min(end_layer, len(model.layers))):
                        layer = model.layers[layer_idx]
                        # Qwen3 layer forward pass
                        layer_outputs = layer(hidden_states)
                        # Handle different return formats
                        if isinstance(layer_outputs, tuple):
                            hidden_states = layer_outputs[0]
                        else:
                            hidden_states = layer_outputs
                
                # Apply final LM head if needed
                if has_lm_head and hasattr(model, 'lm_head'):
                    # Apply final layer norm first if it exists
                    if hasattr(model, 'norm'):
                        hidden_states = model.norm(hidden_states)
                    # Get logits from LM head
                    hidden_states = model.lm_head(hidden_states)
                
            return hidden_states
        
        # Learning endpoints
        async def learning_start(request):
            """Handle learning start notification from service node."""
            try:
                data = await request.json()
                round_id = data.get("round_id")
                target_expert = data.get("target_expert")
                base_version = data.get("base_version")
                required_precision = data.get("precision", self.precision)
                
                # Validate precision requirement
                if required_precision != self.precision:
                    logger.error(f"Precision mismatch: coordinator requires {required_precision}, node enforces {self.precision}")
                    return web.json_response({
                        "status": "rejected",
                        "reason": f"Precision mismatch: node only supports {self.precision}"
                    }, status=400)
                
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
        
        # Queue management endpoints
        async def chat_admit(request):
            """Enqueue a chat request and get a ticket."""
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
                if not isinstance(prompt, str):
                    return web.json_response({"error": "prompt must be a string"}, status=400)
                
                # Enforce prompt length cap (default 8192 chars)
                max_prompt_length = int(os.getenv('MAX_PROMPT_LENGTH', '8192'))
                if len(prompt) > max_prompt_length:
                    return web.json_response({
                        "error": f"Prompt too long ({len(prompt)} chars, max: {max_prompt_length})"
                    }, status=400)
                
                # Get user key (IP or auth ID)
                user_key = request.headers.get('X-User-ID', request.remote)
                
                # Create prompt metadata for scheduling
                prompt_meta = {
                    "length": len(prompt),
                    "max_tokens": data.get("max_new_tokens", 64)
                }
                
                # Enqueue request
                try:
                    ticket = await self.queue_manager.enqueue(user_key, prompt_meta)
                    
                    # Store prompt for later execution
                    request.app['queued_prompts'][ticket.ticket_id] = {
                        "prompt": prompt,
                        "max_new_tokens": data.get("max_new_tokens", 64),
                        "user_key": user_key
                    }
                    
                    return web.json_response({
                        "ticket_id": ticket.ticket_id,
                        "position": ticket.position,
                        "eta_seconds": ticket.eta_seconds,
                        "state": ticket.state
                    })
                    
                except QueueFull:
                    # Suggest retry later when queue is full
                    response = web.json_response({"error": "Queue is full, please try again later"}, status=503)
                    response.headers['Retry-After'] = str(retry_queue_full)
                    return response
                except LimitExceeded as e:
                    # Suggest retry for rate limiting per user
                    response = web.json_response({"error": str(e)}, status=429)
                    response.headers['Retry-After'] = str(retry_rate_limit)
                    return response
                    
            except Exception as e:
                logger.error(f"Admit error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def queue_status(request):
            """Get status of a queued ticket."""
            try:
                ticket_id = request.match_info.get('ticket_id')
                if not ticket_id:
                    return web.json_response({"error": "ticket_id required"}, status=400)
                
                # Validate ticket_id format (32 hex chars for UUID)
                import re
                if not re.match(r'^[a-f0-9]{32}$', ticket_id):
                    return web.json_response({"error": "Invalid ticket_id format"}, status=400)
                
                ticket = await self.queue_manager.get(ticket_id)
                if not ticket:
                    return web.json_response({"error": "Ticket not found"}, status=404)
                
                # Update position and ETA
                position = await self.queue_manager.position(ticket_id)
                eta = await self.queue_manager.eta(ticket_id)
                
                return web.json_response({
                    "ticket_id": ticket.ticket_id,
                    "state": ticket.state,
                    "position": position,
                    "eta_seconds": eta,
                    "created_at": ticket.created_at,
                    "assigned_at": ticket.assigned_at,
                    "started_at": ticket.started_at,
                    "completed_at": ticket.completed_at
                })
                
            except Exception as e:
                logger.error(f"Status error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def queue_stream(request):
            """SSE stream for live queue updates."""
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'
            await response.prepare(request)
            
            try:
                # Get ticket ID from query params
                ticket_id = request.query.get('ticket_id')
                if not ticket_id:
                    await response.write(b'data: {"error": "ticket_id required"}\n\n')
                    return response
                
                # Validate ticket_id format
                import re
                if not re.match(r'^[a-f0-9]{32}$', ticket_id):
                    await response.write(b'data: {"error": "Invalid ticket_id format"}\n\n')
                    return response
                
                # Send updates every second
                while True:
                    ticket = await self.queue_manager.get(ticket_id)
                    if not ticket:
                        await response.write(b'data: {"error": "ticket not found"}\n\n')
                        break
                    
                    # Get current position and ETA
                    position = await self.queue_manager.position(ticket_id)
                    eta = await self.queue_manager.eta(ticket_id)
                    
                    # Send SSE event
                    event_data = {
                        "ticket_id": ticket.ticket_id,
                        "state": ticket.state,
                        "position": position,
                        "eta_seconds": eta
                    }
                    
                    await response.write(f'data: {json.dumps(event_data)}\n\n'.encode())
                    
                    # Stop streaming if ticket is done
                    if ticket.state in {"done", "failed", "canceled", "expired"}:
                        break
                    
                    # Stop streaming if started (client should switch to chat endpoint)
                    if ticket.state == "started":
                        await response.write(b'data: {"event": "ready_to_start"}\n\n')
                        break
                    
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                pass
            finally:
                await response.write_eof()
            
            return response
        
        async def queue_cancel(request):
            """Cancel a queued ticket."""
            try:
                ticket_id = request.match_info.get('ticket_id')
                if not ticket_id:
                    return web.json_response({"error": "ticket_id required"}, status=400)
                
                # Validate ticket_id format
                import re
                if not re.match(r'^[a-f0-9]{32}$', ticket_id):
                    return web.json_response({"error": "Invalid ticket_id format"}, status=400)
                
                # Get user key for authorization
                user_key = request.headers.get('X-User-ID', request.remote)
                
                success = await self.queue_manager.cancel(ticket_id, user_key)
                if success:
                    # Clean up stored prompt
                    if 'queued_prompts' in request.app:
                        request.app['queued_prompts'].pop(ticket_id, None)
                    
                    return web.json_response({"status": "canceled"})
                else:
                    return web.json_response({"error": "Cannot cancel ticket"}, status=403)
                    
            except Exception as e:
                logger.error(f"Cancel error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def chat_start(request):
            """Start processing an admitted ticket."""
            try:
                data = await request.json()
                ticket_id = data.get("ticket_id")
                if not ticket_id:
                    return web.json_response({"error": "ticket_id required"}, status=400)
                
                # Validate ticket_id format
                import re
                if not re.match(r'^[a-f0-9]{32}$', ticket_id):
                    return web.json_response({"error": "Invalid ticket_id format"}, status=400)
                
                # Get ticket
                ticket = await self.queue_manager.get(ticket_id)
                if not ticket:
                    return web.json_response({"error": "Ticket not found"}, status=404)
                
                # Check if assigned
                if ticket.state != "assigned":
                    return web.json_response({
                        "error": f"Ticket not ready (state: {ticket.state})"
                    }, status=400)
                
                # Get stored prompt
                prompt_data = request.app['queued_prompts'].get(ticket_id)
                if not prompt_data:
                    return web.json_response({"error": "Prompt data not found"}, status=404)
                
                # Acquire a job slot before starting
                if not await self.acquire_job_slot():
                    resp = web.json_response({
                        "error": "At capacity, please wait and retry start",
                        "capacity": {
                            "total": self.job_capacity,
                            "used": self.active_jobs,
                            "available": max(0, self.job_capacity - self.active_jobs)
                        }
                    }, status=503)
                    # Use unified start capacity retry hint
                    resp.headers['Retry-After'] = str(retry_start_cap)
                    return resp

                # Mark as started
                await self.queue_manager.start(ticket_id)
                
                try:
                    # Perform inference
                    if not self.model_manager:
                        raise RuntimeError("Model not ready")
                    
                    import time
                    start_time = time.time()
                    
                    response = self.model_manager.generate(
                        prompt_data["prompt"],
                        max_new_tokens=prompt_data["max_new_tokens"]
                    )
                    
                    inference_time = time.time() - start_time
                    
                    # Mark as completed
                    await self.queue_manager.complete(ticket_id, success=True)
                    
                    # Clean up stored prompt
                    request.app['queued_prompts'].pop(ticket_id, None)
                    
                    # Release job slot
                    await self.release_job_slot()
                    
                    return web.json_response({
                        "response": response,
                        "inference_time": inference_time,
                        "ticket_id": ticket_id
                    })
                    
                except Exception as e:
                    # Mark as failed
                    await self.queue_manager.complete(ticket_id, success=False)
                    
                    # Clean up and release slot
                    request.app['queued_prompts'].pop(ticket_id, None)
                    await self.release_job_slot()
                    
                    logger.error(f"Inference failed for ticket {ticket_id}: {e}")
                    return web.json_response({"error": str(e)}, status=500)
                    
            except Exception as e:
                logger.error(f"Start error: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def queue_metrics(request):
            """Get queue metrics."""
            try:
                metrics = await self.queue_manager.metrics()
                metrics['capacity'] = {
                    'total': self.job_capacity,
                    'used': self.active_jobs,
                    'available': self.job_capacity - self.active_jobs
                }
                # Add Retry-After configuration for visibility
                metrics['retry_after_config'] = {
                    'chat_capacity': retry_chat_cap,
                    'queue_full': retry_queue_full,
                    'rate_limit': retry_rate_limit,
                    'start_capacity': retry_start_cap
                }
                return web.json_response(metrics)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
        
        # Register routes
        enable_learning = os.getenv('ENABLE_LEARNING_ENDPOINTS', 'false').lower() == 'true'
        if cors:
            # Add routes with CORS
            cors.add(app.router.add_get('/', health))
            cors.add(app.router.add_get('/health', health))
            cors.add(app.router.add_get('/metrics', metrics))
            cors.add(app.router.add_get('/debug/moe-status', debug_moe_status))
            cors.add(app.router.add_post('/chat', chat))
            cors.add(app.router.add_get('/chain/{chain_id}', chain_info))
            cors.add(app.router.add_post('/inference', inference))
            cors.add(app.router.add_post('/inference/stage', inference_stage))  # Add pipeline stage endpoint
            cors.add(app.router.add_get('/pol/status', lambda r: web.json_response({"status": "ok"})))
            
            # Queue management routes with CORS
            cors.add(app.router.add_post('/chat/admit', chat_admit))
            cors.add(app.router.add_get('/queue/status/{ticket_id}', queue_status))
            cors.add(app.router.add_get('/queue/stream', queue_stream))
            cors.add(app.router.add_post('/queue/cancel/{ticket_id}', queue_cancel))
            cors.add(app.router.add_post('/chat/start', chat_start))
            cors.add(app.router.add_get('/queue/metrics', queue_metrics))
            
            # Learning routes with CORS (optional)
            if enable_learning:
                cors.add(app.router.add_post('/learning/start', learning_start))
                cors.add(app.router.add_post('/learning/data', learning_data_allocation))
                cors.add(app.router.add_get('/learning/status', learning_status))
                cors.add(app.router.add_post('/learning/delta', submit_delta))
        else:
            # Add routes without CORS
            app.router.add_get('/', health)
            app.router.add_get('/health', health)
            app.router.add_get('/metrics', metrics)
            app.router.add_get('/debug/moe-status', debug_moe_status)
            app.router.add_post('/chat', chat)
            app.router.add_get('/chain/{chain_id}', chain_info)
            app.router.add_post('/inference', inference)
            app.router.add_get('/pol/status', lambda r: web.json_response({"status": "ok"}))
            
            # Queue management routes
            app.router.add_post('/chat/admit', chat_admit)
            app.router.add_get('/queue/status/{ticket_id}', queue_status)
            app.router.add_get('/queue/stream', queue_stream)
            app.router.add_post('/queue/cancel/{ticket_id}', queue_cancel)
            app.router.add_post('/chat/start', chat_start)
            app.router.add_get('/queue/metrics', queue_metrics)
            
            if enable_learning:
                app.router.add_post('/learning/start', learning_start)
                app.router.add_post('/learning/data', learning_data_allocation)
                app.router.add_get('/learning/status', learning_status)
                app.router.add_post('/learning/delta', submit_delta)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner  # Store for clean shutdown
        
        # Start server on configured port
        port = self.port
        logger.info(f"🚀 Preparing to start server on port {port}")
        
        # Check if we're in RunPod environment FIRST (before any cleanup)
        if os.path.exists('/runpod') or os.path.exists('/workspace'):
            logger.info("📍 Detected RunPod/cloud environment")
            # RunPod HTTP Service uses 8001, we need a different port
            if port == 8001:
                logger.warning("⚠️ Port 8001 is reserved for RunPod HTTP Service proxy")
                logger.info("💡 Using alternative port 8000 for internal server...")
                port = 8000
                self.port = 8000
        
        # Determine if we need fixed port (Vast.ai constraint)
        disable_increment = os.environ.get('DISABLE_PORT_INCREMENT', '').lower() in ['true', '1', 'yes']
        requires_exact_port = disable_increment or (PUBLIC_PORT and port == PUBLIC_PORT)
        
        # Log WHY we need fixed port (or not)
        if requires_exact_port:
            if disable_increment and PUBLIC_PORT and port == PUBLIC_PORT:
                logger.info(f"📌 Fixed port {port} required: DISABLE_PORT_INCREMENT=true AND PUBLIC_PORT={PUBLIC_PORT}")
            elif disable_increment:
                logger.info(f"📌 Fixed port {port} required: DISABLE_PORT_INCREMENT=true")
            elif PUBLIC_PORT and port == PUBLIC_PORT:
                logger.info(f"📌 Fixed port {port} required: PUBLIC_PORT mapping constraint")
        else:
            logger.info(f"🔄 Flexible port mode - will increment if {port} is busy")
            logger.debug(f"   DISABLE_PORT_INCREMENT={disable_increment}, PUBLIC_PORT={PUBLIC_PORT}, port={port}")
        
        # Optional: auto-select a free port within Vast range (bind-first, no pre-scan)
        vast_auto_port = os.environ.get('VAST_AUTO_PORT', '').lower() in ['true', '1', 'yes']
        vast_range = os.environ.get('VAST_PORT_RANGE', '')  # e.g. "20222-27706"
        if vast_auto_port:
            if requires_exact_port:
                logger.info("🧭 VAST_AUTO_PORT enabled - overriding fixed port requirement to find a free port")
                requires_exact_port = False
            # Build candidate list: desired port first, then the declared range
            candidates: List[int] = [int(port)]
            try:
                if '-' in vast_range:
                    a, b = vast_range.split('-', 1)
                    start, end = int(a), int(b)
                    if start > end:
                        start, end = end, start
                    for p in range(start, end + 1):
                        if p != port:
                            candidates.append(p)
            except Exception:
                pass
            # Deduplicate while preserving order
            seen_c = set()
            candidates = [p for p in candidates if (p not in seen_c and not seen_c.add(p))]
            logger.info(f"🧪 Auto-selecting port from candidates ({len(candidates)} total)")
            for cand in candidates:
                # Skip RunPod reserved port
                if (os.path.exists('/runpod') or os.path.exists('/workspace')) and cand == 8001:
                    continue
                try:
                    logger.info(f"🔌 Attempting to bind to candidate port {cand}...")
                    site = web.TCPSite(
                        runner,
                        '0.0.0.0',
                        cand,
                        reuse_port=False,
                        reuse_address=True
                    )
                    await site.start()
                    # Success
                    self._site = site
                    self.port = cand
                    try:
                        # Keep PUBLIC_PORT in sync with selected internal port for Vast 1:1 mapping
                        PUBLIC_PORT = cand
                    except Exception:
                        pass
                    logger.info(f"✅ Server running on http://0.0.0.0:{cand} (auto-selected)")
                    self.server_started = True
                    # Register and return
                    await self.register_with_main()
                    return
                except OSError as e:
                    if "address already in use" in str(e).lower():
                        logger.warning(f"⛔ Port {cand} busy, trying next candidate...")
                        continue
                    else:
                        logger.error(f"❌ Failed to bind to port {cand}: {e}")
                        continue

        # Try to start server
        max_retries = 5 if requires_exact_port else 3  # More retries for fixed port with JIT cleanup

        for retry in range(max_retries):
            try:
                # Just-in-time cleanup for fixed port (right before bind!)
                if requires_exact_port and (retry == 0 or retry > 0):
                    logger.info(f"🧹 [Attempt {retry+1}/{max_retries}] Cleaning port {port} immediately before bind...")
                    port_cleaned = await self._cleanup_port(port)
                    if port_cleaned:
                        logger.info(f"   ✅ Port {port} cleaned successfully")
                    else:
                        logger.warning(f"   ⚠️ Port {port} cleanup incomplete, attempting bind anyway")
                    # Small delay to let OS release the socket
                    await asyncio.sleep(0.5)
                
                # Configure socket options for better reliability
                logger.info(f"🔌 Attempting to bind to port {port}...")
                site = web.TCPSite(
                    runner, 
                    '0.0.0.0', 
                    port,
                    reuse_port=False,  # Don't allow multiple processes on same port
                    reuse_address=True  # Allow quick restart after shutdown
                )
                # Note: site is created fresh each attempt, not reused
                await site.start()
                # Only store site AFTER successful start
                self._site = site  # Store for clean shutdown
                # If bound to ephemeral port (0), discover actual port
                try:
                    bound_port = None
                    _srv = getattr(site, '_server', None)
                    if _srv and getattr(_srv, 'sockets', None):
                        bound_port = _srv.sockets[0].getsockname()[1]
                    if bound_port:
                        port = bound_port
                except Exception:
                    pass
                logger.info(f"✅ Server running on http://0.0.0.0:{port}")
                self.port = port  # Update port if changed/assigned
                
                # CRITICAL: Vast.ai port mapping check
                if PUBLIC_PORT and port != PUBLIC_PORT:
                    logger.warning("=" * 60)
                    logger.warning(f"⚠️ VAST.AI PORT MISMATCH WARNING!")
                    logger.warning(f"   Internal server port: {port}")
                    logger.warning(f"   External PUBLIC_PORT: {PUBLIC_PORT}")
                    logger.warning(f"   Problem: Vast.ai only forwards {PUBLIC_PORT} → {PUBLIC_PORT}")
                    logger.warning(f"   Result: External connections to port {PUBLIC_PORT} won't reach this server!")
                    logger.warning(f"   Solutions:")
                    logger.warning(f"   1. Kill process using port {PUBLIC_PORT} and restart")
                    logger.warning(f"   2. Set DISABLE_PORT_INCREMENT=true to prevent port changes")
                    logger.warning(f"   3. Use a different Vast.ai port mapping")
                    logger.warning("=" * 60)
                
                self.server_started = True
                break
            except OSError as e:
                # Clean up any partially created site (shouldn't be needed, but safe)
                site = None  # Let GC clean up
                
                if "address already in use" in str(e).lower():
                    # Enhanced diagnostic: show which process holds the port NOW
                    logger.error(f"❌ [Attempt {retry+1}/{max_retries}] Port {port} is in use at bind time!")
                    
                    # Capture what's using the port RIGHT NOW (try multiple tools)
                    try:
                        import subprocess as _sub
                        import shutil
                        
                        captured = False
                        
                        # Try ss first (most common in containers)
                        if shutil.which('ss') and not captured:
                            _res = _sub.run(['ss', '-tlnp'], capture_output=True, text=True, timeout=2)
                            if _res.stdout:
                                for line in _res.stdout.split('\n'):
                                    if f':{port} ' in line and 'LISTEN' in line:
                                        logger.error("🔍 Process holding port (via ss):")
                                        import re
                                        pid_match = re.search(r'pid=(\d+)', line)
                                        proc_match = re.search(r'\("([^"]+)"', line)
                                        if pid_match:
                                            pid = pid_match.group(1)
                                            proc = proc_match.group(1) if proc_match else 'unknown'
                                            logger.error(f"   → {proc} (PID {pid})")
                                            captured = True
                        
                        # Try lsof if ss didn't work
                        if shutil.which('lsof') and not captured:
                            _res = _sub.run(['lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN'], 
                                           capture_output=True, text=True, timeout=2)
                            if _res.stdout:
                                logger.error("🔍 Process holding port (via lsof):")
                                lines = _res.stdout.strip().split('\n')
                                for line in lines[1:]:  # Skip header
                                    if line.strip():
                                        parts = line.split()
                                        if len(parts) > 1:
                                            logger.error(f"   → {parts[0]} (PID {parts[1]})")
                                            captured = True
                                            break
                        
                        # Fall back to netstat as last resort
                        if shutil.which('netstat') and not captured:
                            _res = _sub.run(['netstat', '-tlnp'], capture_output=True, text=True, timeout=2)
                            if _res.stdout:
                                for line in _res.stdout.split('\n'):
                                    if f':{port}' in line and 'LISTEN' in line:
                                        logger.error("🔍 Process info (via netstat):")
                                        logger.error(f"   → {line.strip()}")
                                        captured = True
                                        break
                        
                        if not captured:
                            logger.error("   Could not identify process (no tools available)")
                    except Exception as ex:
                        logger.debug(f"   Could not identify process: {ex}")
                    
                    # Handle based on whether we need exact port
                    if requires_exact_port:
                        if retry < max_retries - 1:
                            # We have more retries - wait and try again with fresh cleanup
                            wait_time = 2.0 * (retry + 1)  # Progressive backoff: 2s, 4s, 6s...
                            logger.warning(f"🕰️ Waiting {wait_time}s before retry {retry+2}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                            # Loop will do cleanup again on next iteration
                        else:
                            # Final failure - provide detailed diagnostics
                            logger.error("=" * 60)
                            logger.error(f"❌ FATAL: Port {port} remains in use after {max_retries} attempts")
                            logger.error(f"   Fixed port requirement prevents fallback")
                            logger.error(f"   Environment:")
                            logger.error(f"   - NODE_PORT={os.environ.get('NODE_PORT', 'not set')}")
                            logger.error(f"   - PORT={os.environ.get('PORT', 'not set')}")
                            logger.error(f"   - PUBLIC_PORT={os.environ.get('PUBLIC_PORT', 'not set')}")
                            logger.error(f"   - DISABLE_PORT_INCREMENT={disable_increment}")
                            logger.error(f"   Manual intervention required:")
                            logger.error(f"   1. Kill the process shown above")
                            logger.error(f"   2. Or change to a different port")
                            logger.error("=" * 60)
                            raise RuntimeError(f"Port {port} unavailable after {max_retries} cleanup attempts")
                    else:
                        # Flexible mode - try next port
                        if retry < max_retries - 1:
                            port += 1
                            logger.warning(f"🔄 Port {port-1} busy, trying port {port}...")
                            if PUBLIC_PORT:
                                logger.warning(f"   ⚠️ WARNING: Port {port} may not match PUBLIC_PORT={PUBLIC_PORT}!")
                else:
                    logger.error(f"❌ Failed to start server: {e}")
                    raise
        
        # Register with main node
        await self.register_with_main()

        # Eager warmup after bind + register (if enabled and not uploading)
        if os.getenv('WARMUP_ON_START', 'true').lower() == 'true':
            # Only warm up if blockchain has full model weights
            ready, have, need = self._blockchain_model_readiness()
            if not ready:
                logger.info(f"🧊 Warmup skipped: blockchain not ready ({have}/{need} components)")
            elif getattr(self, '_upload_in_progress', False):
                logger.info("🧊 Warmup deferred: upload in progress")
            elif not getattr(self, '_warmup_scheduled', False):
                self._warmup_scheduled = True
                logger.info("🧊 Scheduling eager warmup after register...")
                asyncio.create_task(self._warmup_gpu())
            else:
                logger.info("🧊 Warmup deferred: upload in progress")

    async def register_with_main(self):
        """Register this node with the main node."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Determine public host - use provided or auto-detect
                if PUBLIC_HOST:
                    # User provided a specific host (IP or domain)
                    public_host = PUBLIC_HOST
                    logger.info(f"📡 Using configured public host: {public_host}")
                else:
                    # Auto-detect public IP
                    try:
                        resp = await client.get("https://api.ipify.org")
                        public_host = resp.text.strip() if resp.status_code == 200 else "unknown"
                        logger.info(f"📡 Auto-detected public IP: {public_host}")
                    except Exception as e:
                        logger.warning(f"Could not detect public IP: {e}")
                        public_host = "unknown"
                
                # For dense model, send layer information instead of experts
                available_experts = []
                try:
                    if self.model_manager:
                        # Dense model has layers, not experts
                        # Send layer names as "experts" for compatibility with API
                        from backend.core.param_index import ParameterIndex
                        param_index = ParameterIndex(DATA_DIR / "param_index.json")
                        available_layers = param_index.get_all_layers()
                        
                        if available_layers:
                            # Convert layer names to expert-like format for API compatibility
                            # This allows the main node to understand what the GPU node has
                            available_experts = available_layers
                            logger.info(f"   Dense model has {len(available_layers)} layers available")
                        else:
                            # If no layers in index, just report model readiness
                            available_experts = ["dense_model_ready"] if self.model_manager else []
                except Exception as e:
                    logger.warning(f"Could not enumerate layers for registration: {e}")
                    # Fallback: just indicate we have a dense model
                    if self.model_manager:
                        available_experts = ["dense_model"]
                
                # Register with main node
                # Format the endpoint URL properly
                if public_host.startswith('http://') or public_host.startswith('https://'):
                    # Host already includes protocol - use as-is
                    endpoint_url = public_host
                    # Don't add port if it's already in the URL or if using standard ports
                    if ':' not in public_host.split('://')[-1] and PUBLIC_PORT not in [80, 443]:
                        endpoint_url = f"{endpoint_url}:{PUBLIC_PORT}"
                elif 'proxy.runpod.net' in public_host:
                    # RunPod proxy always uses HTTPS on standard port
                    endpoint_url = f"https://{public_host}"
                    # RunPod uses standard HTTPS port, don't add port
                else:
                    # Default to HTTP for regular hosts
                    endpoint_url = f"http://{public_host}:{PUBLIC_PORT}"
                
                # Convert GPU info to hardware_info format expected by API
                hardware_info = {
                    "vram_gb": self.gpu_info.get("memory_gb", 0),
                    "cuda": self.gpu_info.get("cuda_version", ""),
                    "gpu_name": self.gpu_info.get("name", "Unknown"),
                    "num_gpus": self.gpu_info.get("num_gpus", 1)
                }
                
                # Get CUDA capability if available
                cuda_cap = None
                if self.gpu_available:
                    try:
                        import torch
                        cap = torch.cuda.get_device_capability()
                        cuda_cap = f"{cap[0]}.{cap[1]}"
                    except:
                        pass
                
                data = {
                    "node_id": self.node_id,
                    "host": endpoint_url,  # Send full URL instead of separate host:port
                    "port": 443 if 'https://' in endpoint_url else PUBLIC_PORT,
                    "available_experts": available_experts,
                    "node_name": f"GPU Node {self.node_id}",
                    "resource_limit": "gpu-75",  # GPU node with 75% resource allocation
                    "node_type": "gpu",
                    "hardware_info": hardware_info,  # Use hardware_info instead of gpu_info
                    "donor_mode": False,
                    # Optional device profile fields
                    "vram_gb": self.gpu_info.get("memory_gb", 0),
                    "cuda_capability": cuda_cap
                }
                
                logger.info(f"📝 Registering with endpoint: {endpoint_url}")
                logger.info(f"   Sending {len(available_experts)} experts to main node")
                if len(available_experts) > 0:
                    logger.debug(f"   First 3 experts: {available_experts[:3]}")
                if PUBLIC_PORT != self.port:
                    logger.info(f"   (Internal port: {self.port}, Public endpoint: {endpoint_url})")
                
                # Add API key with proper Authorization header
                headers = {}
                api_key = os.environ.get('BLYAN_API_KEY')
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                    logger.debug("   Using Bearer token for authentication")
                
                resp = await client.post(f"{MAIN_NODE_URL}/p2p/register", json=data, headers=headers)
                if resp.status_code == 200:
                    result = resp.json()
                    # Check if it's a standalone response (main node P2P not available)
                    if isinstance(result, dict) and result.get('status') == 'standalone':
                        logger.info("ℹ️  Main node P2P mode not available - running independently")
                        logger.info("✅ GPU node will operate in standalone mode")
                    else:
                        logger.info(f"✅ Registered with main node")
                        if isinstance(result, dict) and 'message' in result:
                            logger.debug(f"   Response: {result['message']}")
                        
                        # Also register with GPU Node Manager for atomic chat fast-path
                        try:
                            gpu_headers = {}
                            if api_key:
                                gpu_headers['Authorization'] = f'Bearer {api_key}'
                            gpu_payload = {
                                "node_id": self.node_id,
                                "api_url": endpoint_url,
                                "capabilities": {
                                    "layers": available_experts,
                                    "gpu_memory_gb": self.gpu_info.get("memory_gb", 0),
                                    "supports_bf16": self.gpu_info.get("supports_bf16", True),
                                    "gpu": self.gpu_info.get("name", "Unknown GPU"),
                                    "model": "Qwen/Qwen3-8B"
                                }
                            }
                            logger.info(f"📝 Registering GPU node with ID: {self.node_id}")
                            logger.debug(f"Registration payload: {gpu_payload}")
                            gpu_resp = await client.post(
                                f"{MAIN_NODE_URL}/gpu/register",
                                json=gpu_payload,
                                headers=gpu_headers
                            )
                            if gpu_resp.status_code == 200:
                                logger.info(f"✅ Registered GPU node '{self.node_id}' with GPU Node Manager")
                                # Start heartbeat task (always start to ensure it's running)
                                if not hasattr(self, '_heartbeat_task') or self._heartbeat_task is None:
                                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                                    logger.info("💓 Heartbeat task started")
                            else:
                                body = None
                                try:
                                    body = gpu_resp.text[:300]
                                except Exception:
                                    body = None
                                logger.warning(
                                    f"GPU Node Manager registration status: {gpu_resp.status_code}" +
                                    (f" body: {body}" if body else "")
                                )
                                # Start heartbeat anyway for retries
                                if not hasattr(self, '_heartbeat_task') or self._heartbeat_task is None:
                                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                                    logger.info("💓 Heartbeat task started (will retry registration)")
                        except Exception as gre:
                            logger.debug(f"GPU Node Manager registration skipped/failed: {gre}")
                elif resp.status_code == 500:
                    # Try to get error details
                    try:
                        error_text = resp.text
                        logger.debug(f"Full error response: {error_text}")
                    except:
                        error_text = "Could not get error details"
                    
                    # Check if it's because distributed coordinator is not initialized
                    # This is common and expected - main node may not have P2P enabled
                    if ("distributed" in error_text.lower() or 
                        "p2p" in error_text.lower() or 
                        "coordinator" in error_text.lower() or
                        "not available" in error_text.lower()):
                        logger.info("ℹ️  Main node P2P mode not available - running independently")
                        logger.info("✅ GPU node will operate in standalone mode")
                    else:
                        # Only warn for unexpected errors
                        logger.warning(f"Registration error: {error_text[:200] if error_text else 'Internal Server Error'}")
                        logger.info("💡 Continuing in standalone mode")
                else:
                    logger.info(f"Registration status: {resp.status_code}")
                    
        except Exception as e:
            logger.info(f"Could not register: {e}")
            
            # Schedule retry if registration failed
            if not hasattr(self, '_registration_retries'):
                self._registration_retries = 0
            
            self._registration_retries += 1
            if self._registration_retries <= 5:  # Max 5 retries
                retry_delay = min(30 * self._registration_retries, 300)  # 30s, 60s, 90s... max 5min
                logger.info(f"📅 Will retry registration in {retry_delay} seconds (attempt {self._registration_retries}/5)")
                asyncio.create_task(self._delayed_registration_retry(retry_delay))
            else:
                logger.warning("❌ Max registration retries reached. Running in standalone mode.")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep registration alive."""
        logger.info("💓 Starting heartbeat loop (15s interval)")
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while True:
            try:
                await asyncio.sleep(15)  # Send heartbeat every 15 seconds
                
                # Heartbeat with retry logic
                success = await self._send_heartbeat_with_retry()
                
                if success:
                    consecutive_failures = 0
                    logger.debug(f"💓 Heartbeat sent successfully")
                else:
                    consecutive_failures += 1
                    logger.warning(f"💔 Heartbeat failed (attempt {consecutive_failures}/{max_consecutive_failures})")
                    
                    # Re-register after multiple failures
                    if consecutive_failures >= max_consecutive_failures:
                        logger.info("📝 Re-registering due to multiple heartbeat failures")
                        await self.register_with_main()
                        break  # Exit this loop, registration will start new one
                            
            except Exception as e:
                logger.debug(f"Heartbeat loop error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning("Too many heartbeat errors, attempting re-registration")
                    await self.register_with_main()
                    break
    
    async def _send_heartbeat_with_retry(self, max_retries: int = 3) -> bool:
        """Send heartbeat with exponential backoff retry."""
        import httpx
        
        for attempt in range(max_retries):
            try:
                # Increase timeout for slower networks
                timeout = httpx.Timeout(10.0, connect=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    headers = {}
                    api_key = os.environ.get('BLYAN_API_KEY')
                    if api_key:
                        headers['Authorization'] = f'Bearer {api_key}'
                    
                    # Send heartbeat with detailed logging
                    logger.debug(f"Sending heartbeat for node_id: {self.node_id}")
                    resp = await client.post(
                        f"{MAIN_NODE_URL}/gpu/heartbeat",
                        json=self.node_id,  # Raw JSON string body
                        headers=headers
                    )
                    
                    if resp.status_code == 200:
                        return True
                    elif resp.status_code == 404:
                        # Node not found, need to re-register
                        logger.warning(f"Node '{self.node_id}' not found in registry, will re-register")
                        return False
                    elif resp.status_code == 401:
                        try:
                            detail = resp.json().get('detail', 'Unknown')
                        except:
                            detail = resp.text[:200] if resp.text else 'No details'
                        logger.error(f"Authentication failed: {detail}")
                        logger.error("Check BLYAN_API_KEY environment variable")
                        return False
                    elif resp.status_code == 422:
                        try:
                            detail = resp.json().get('detail', 'Unknown')
                        except:
                            detail = resp.text[:200] if resp.text else 'No details'
                        logger.error(f"Invalid request format: {detail}")
                        logger.error(f"Node ID: '{self.node_id}' - check if it contains valid characters")
                        return False
                    else:
                        logger.warning(f"Unexpected status code: {resp.status_code}")
                        try:
                            logger.debug(f"Response: {resp.text[:200]}")
                        except:
                            pass
                        
            except httpx.TimeoutException:
                logger.warning(f"Heartbeat timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.debug(f"Heartbeat error (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Exponential backoff if not the last attempt
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        return False
    
    async def _delayed_registration_retry(self, delay: int):
        """Retry registration after a delay."""
        await asyncio.sleep(delay)
        logger.info(f"🔄 Retrying registration with main node (attempt {self._registration_retries})...")
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                # Get available experts again (might have changed)
                available_experts = []
                if self.model_manager:
                    from backend.core.param_index import ParameterIndex
                    param_index = ParameterIndex(DATA_DIR / "param_index.json")
                    available_experts = param_index.get_all_layers()
                
                # Detect public host if needed
                if 'PUBLIC_HOST' in os.environ:
                    endpoint_url = f"http://{os.environ['PUBLIC_HOST']}:{os.environ.get('PUBLIC_PORT', PORT)}"
                else:
                    try:
                        async with httpx.AsyncClient() as ip_client:
                            resp = await ip_client.get('https://api.ipify.org')
                            public_ip = resp.text
                            endpoint_url = f"http://{public_ip}:{PORT}"
                    except:
                        endpoint_url = f"http://localhost:{PORT}"
                
                data = {
                    "node_id": self.node_id,
                    "host": endpoint_url,
                    "port": PORT,
                    "available_experts": available_experts,
                    "node_type": "gpu",
                    "vram_gb": self.gpu_info.get("memory_gb", 0),
                }
                
                headers = {}
                api_key = os.environ.get('BLYAN_API_KEY')
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                
                resp = await client.post(f"{MAIN_NODE_URL}/p2p/register", json=data, headers=headers)
                
                if resp.status_code == 200:
                    result = resp.json()
                    if isinstance(result, dict) and result.get('status') == 'standalone':
                        logger.info("ℹ️  Main node P2P still not available - continuing standalone")
                        # Don't retry for standalone mode - it's expected
                        self._registration_retries = 999  # Stop retrying
                    else:
                        logger.info(f"✅ Successfully registered with main node on retry!")
                        self._registration_retries = 0  # Reset counter on success
                elif resp.status_code == 500:
                    logger.info("ℹ️  Main node P2P still not available")
                    # Stop retrying - this is expected behavior
                    self._registration_retries += 1
                    if self._registration_retries <= 5:
                        retry_delay = min(60 * self._registration_retries, 300)
                        logger.info(f"📅 Will retry again in {retry_delay} seconds")
                        asyncio.create_task(self._delayed_registration_retry(retry_delay))
                else:
                    logger.warning(f"Registration failed with status: {resp.status_code}")
                    
        except Exception as e:
            logger.warning(f"Registration retry failed: {e}")
            # Schedule another retry if under limit
            if self._registration_retries < 5:
                self._registration_retries += 1
                retry_delay = min(60 * self._registration_retries, 300)
                logger.info(f"📅 Will retry again in {retry_delay} seconds")
                asyncio.create_task(self._delayed_registration_retry(retry_delay))
    
    async def _admission_scheduler(self) -> None:
        """Periodically check for admission opportunities."""
        while True:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms
                await self._try_admit()
                # Opportunistic cleanup of stale prompts (bounded per tick)
                await self._cleanup_stale_prompts(limit=50)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Admission scheduler error: {e}")

    async def _cleanup_stale_prompts(self, limit: int = 50) -> None:
        """Remove stored prompts for tickets that have terminated or expired.

        Args:
            limit: Maximum number of entries to inspect per call to bound work
        """
        try:
            if not hasattr(self, '_queued_prompts') or not self._queued_prompts:
                return
            if not self.queue_manager:
                return
            removed = 0
            # Iterate over a snapshot of keys to avoid RuntimeError on size change
            for ticket_id in list(self._queued_prompts.keys()):
                if removed >= limit:
                    break
                ticket = await self.queue_manager.get(ticket_id)
                if (ticket is None) or (ticket.state in {"done", "failed", "canceled", "expired"}):
                    self._queued_prompts.pop(ticket_id, None)
                    removed += 1
        except Exception as e:
            logger.debug(f"Prompt cleanup error: {e}")
    
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
                    
                    # Registration already happens at startup
                    pass
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
                    f"{MAIN_NODE_URL}/learning/delta/submit",  # ✅ align with service endpoint
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
        
        # Track startup steps
        startup_steps = [
            "Check GPU capabilities",
            "Initialize blockchains", 
            "Check block progress",
            "Verify integrity",
            "Sync from peers",
            "Initialize model manager",
            "Start server"
        ]
        
        total_steps = len(startup_steps)
        current_step = 0
        startup_begin = time.time()
        
        def log_step(step_name, step_num):
            elapsed = time.time() - startup_begin
            mins, secs = divmod(int(elapsed), 60)
            logger.info(f"\n📍 Step {step_num}/{total_steps}: {step_name} [{mins:02d}:{secs:02d}]")
            logger.info("-" * 60)
        
        # 1. Check GPU
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        self.check_gpu()
        
        # 2. Initialize blockchains
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        if not self.initialize_chains():
            # Degrade gracefully: write minimal health and still start server
            logger.error("Failed to initialize chains — starting in degraded mode")
            progress = {
                "meta_blocks": 0,
                "layer_blocks": 0,
                "expected_layers": 36,
                "missing_layers": [],
                "integrity_valid": False,
                "progress_percentage": 0.0
            }
            # Generate health summary with error and start server anyway
            degraded_manifest = {"hosted_count": 0, "verified_count": 0, "missing_layers": [], "ready_to_serve": False}
            try:
                self._generate_node_health_summary(progress, degraded_manifest)
            except Exception:
                pass
            # Attempt to start server so ops can query status
            try:
                await self.start_server()
            except Exception as e:
                logger.error(f"Failed to start server in degraded mode: {e}")
            return
        
        # 3. Check block progress
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        progress = self.check_block_progress()
        
        # 4. Verify integrity
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        # Perform integrity verification here (not in progress step)
        integrity_ok = self.verify_block_integrity()
        progress["integrity_valid"] = integrity_ok
        if not integrity_ok:
            logger.warning("⚠️ Blockchain integrity issue detected")
            recovery_success = await self.handle_integrity_failure()
            if recovery_success:
                # Re-check progress after recovery
                progress = self.check_block_progress()
                progress["integrity_valid"] = self.verify_block_integrity()
            else:
                logger.error("Failed to recover blockchain integrity")
                # Continue anyway but with warnings
        else:
            logger.info("✅ Integrity check passed")
        
        # 4.5 Build layer manifest and verify partial node
        layer_manifest_summary = self._build_and_verify_layer_manifest()
        
        # 4.6 Generate node health summary
        self._generate_node_health_summary(progress, layer_manifest_summary)
        
        # Continue building blocks if needed
        self.continue_block_building(progress)
        
        # 5. Initial sync attempt (non-blocking)
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        sync_success = await self.sync_from_peers()
        
        # 5.5. Wait for auto-upload to complete if it was triggered
        if AUTO_UPLOAD and hasattr(self, '_upload_triggered_from_block_building'):
            logger.info("⏳ Waiting for model upload to complete before initializing model manager...")
            max_wait = 600  # 10 minutes max
            wait_start = time.time()
            while hasattr(self, '_upload_in_progress') and self._upload_in_progress:
                if time.time() - wait_start > max_wait:
                    logger.warning("Upload taking too long, proceeding anyway")
                    break
                await asyncio.sleep(5)
                elapsed = time.time() - wait_start
                if int(elapsed) % 30 == 0:  # Log every 30 seconds
                    logger.info(f"   Still uploading... ({elapsed:.0f}s elapsed)")
            
            if hasattr(self, 'upload_completed') and self.upload_completed:
                logger.info("✅ Upload completed, proceeding with model manager initialization")
        
        # 6. Initialize model manager
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        if not self.initialize_model_manager():
            logger.warning("Running without model manager")
        else:
            # Pre-start warmup disabled by default; eager warmup runs after bind+register
            if os.getenv('ENABLE_STARTUP_WARMUP', 'false').lower() == 'true':
                try:
                    from backend.core.param_index import ParameterIndex
                    uploading = hasattr(self, '_upload_in_progress') and self._upload_in_progress
                    param_index = ParameterIndex(DATA_DIR / "param_index.json")
                    have_full = len(param_index.get_all_layers()) >= 38
                    if uploading or not have_full:
                        logger.info("🧊 Skipping warm start (uploading or incomplete blockchain)")
                    else:
                        logger.info("🧊 Performing blocking warm start...")
                        await self._warmup_gpu()
                except Exception as _e:
                    logger.warning(f"Warm start failed (continuing): {_e}")

        # 4.5 Check if we just completed an upload and need to reinit
        if hasattr(self, 'upload_completed') and self.upload_completed:
            logger.info("🔄 Upload was just completed, reinitializing model manager with blockchain weights...")
            # Wait a bit for filesystem to settle
            await asyncio.sleep(2)
            # If a local/HF model is currently loaded, unload it to free VRAM
            try:
                if self.model_manager and getattr(self.model_manager, '_loaded_from_blockchain', False) is False:
                    logger.info("🧹 Unloading preloaded raw model from VRAM")
                    import torch as _torch
                    self.model_manager.model = None
                    _torch.cuda.empty_cache()
                    # Force reinit from blockchain
                    from backend.model.manager import get_model_manager as _gmm
                    self.model_manager = _gmm(DATA_DIR, force_new=True, model_name=MODEL_NAME, device="cuda" if self.gpu_available else "cpu", use_blockchain=True, use_gpu_direct=os.getenv("USE_GPU_DIRECT", "true").lower() == "true")
            except Exception as _e:
                logger.warning(f"Could not unload local model: {_e}")
            if self.initialize_model_manager():
                logger.info("✅ Model manager reinitialized with blockchain experts")
                # Now perform warm start on the fresh blockchain model
                try:
                    await self._warmup_gpu()
                except Exception as _e:
                    logger.warning(f"Warm start after upload failed: {_e}")
            else:
                logger.error("❌ Failed to reinitialize model manager after upload")
        
        # 7. Start server
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        await self.start_server()
        
        # Start periodic sync if initial sync failed
        if not sync_success:
            asyncio.create_task(self.periodic_sync())
        
        # Registration already happens in start_server()
        
        # Initialize queue manager
        try:
            from backend.runtime.queue_manager import QueueManager
            max_queue_length = int(os.getenv('MAX_QUEUE_LENGTH', '200'))
            max_user_concurrency = int(os.getenv('MAX_USER_CONCURRENCY', '1'))
            admission_timeout = int(os.getenv('ADMISSION_TIMEOUT_S', '600'))
            job_timeout = int(os.getenv('JOB_TIMEOUT_S', '600'))
            
            self.queue_manager = QueueManager(
                max_queue_length=max_queue_length,
                max_user_concurrency=max_user_concurrency,
                admission_timeout_s=admission_timeout,
                job_timeout_s=job_timeout
            )
            logger.info(f"📋 Queue manager initialized (capacity: {self.max_concurrent_jobs}, queue: {max_queue_length})")
            
            # Start admission scheduler
            self._admission_task = asyncio.create_task(self._admission_scheduler())
            
        except Exception as e:
            logger.warning(f"Could not initialize queue manager: {e}")
        
        # Start background auditor if enabled
        if os.getenv('ENABLE_BACKGROUND_AUDIT', 'true').lower() == 'true':
            try:
                from backend.core.background_auditor import BackgroundAuditor
                self.background_auditor = BackgroundAuditor(DATA_DIR, self.chains)
                audit_interval = int(os.getenv('AUDIT_INTERVAL_SECONDS', '300'))
                await self.background_auditor.start(audit_interval)
                logger.info(f"🔍 Background auditor started (interval: {audit_interval}s)")
            except Exception as e:
                logger.warning(f"Could not start background auditor: {e}")
        
        # Final summary
        total_time = time.time() - startup_begin
        mins, secs = divmod(int(total_time), 60)
        # Get layer count for summary (dense model has layers, not experts)
        layer_count = 32  # Qwen3-8B has 32 layers
        if self.model_manager:
            try:
                from backend.model.dynamic_config import get_model_config
                config = get_model_config()
                layer_count = config.num_layers if hasattr(config, 'num_layers') else 32
            except:
                pass
        
        logger.info("\n" + "=" * 60)
        logger.info(f"🚀 NODE STARTUP COMPLETE")
        logger.info(f"⏱️  Total time: {mins:02d}:{secs:02d}")
        logger.info(f"📊 Blocks loaded: {len(self.chains['B']._hash_index) if hasattr(self.chains['B'], '_hash_index') else 0}")
        
        # Report model status
        if self.model_manager:
            from backend.core.param_index import ParameterIndex
            param_index = ParameterIndex(DATA_DIR / "param_index.json")
            layer_count = len(param_index.get_all_layers())
            if layer_count > 0:
                logger.info(f"🤖 Dense model ready: {layer_count} layers from blockchain")
        
        logger.info(f"🌐 Server: http://0.0.0.0:{self.port}")
        logger.info(f"✅ Ready for inference requests")
        logger.info("=" * 60)
        
        if not sync_success:
            logger.info("📡 Running in offline mode - will sync when main node becomes available")
        
        while True:
            await asyncio.sleep(60)
            # Periodic health check
            if hasattr(self, 'model_manager') and self.model_manager:
                logger.debug(f"Node {self.node_id} healthy - GPU: {self.gpu_available}")
    
    def _build_and_verify_layer_manifest(self) -> Dict:
        """Build and verify layer manifest for partial node support."""
        try:
            from backend.runtime.block.layer_manifest import LayerManifest
            
            logger.info("📋 Building layer manifest...")
            
            # Initialize manifest
            manifest = LayerManifest(DATA_DIR)
            
            # Build from param_index if available
            if (DATA_DIR / "param_index.json").exists():
                chain_b = self.chains.get('B')
                if chain_b:
                    manifest.build_from_param_index(chain_b)
                    
                    # Verify hosted layers
                    summary = manifest.verify_hosted_layers(chain_b)
                    
                    # Get expected layers for dense model (model_norm required, other_weights optional)
                    expected_layers = ["embedding"] + [f"layer_{i}" for i in range(36)] + ["lm_head", "model_norm"]
                    # Include other_weights in expected list only if it exists in param_index
                    if "other_weights" in manifest.get_hosted_layers():
                        expected_layers.append("other_weights")
                    missing = manifest.get_missing_layers(expected_layers)
                    summary['missing_from_full_model'] = missing
                    
                    return summary
                else:
                    logger.warning("Chain B not available for manifest building")
            else:
                logger.info("No param_index.json found, skipping manifest")
            
            return {
                'hosted_count': 0,
                'verified_count': 0,
                'missing_layers': [],
                'ready_to_serve': False
            }
            
        except Exception as e:
            logger.error(f"Failed to build layer manifest: {e}")
            return {
                'hosted_count': 0,
                'verified_count': 0,
                'missing_layers': [],
                'ready_to_serve': False,
                'error': str(e)
            }
    
    def _generate_node_health_summary(self, progress: Dict, layer_manifest: Dict) -> None:
        """Generate and save node health summary."""
        try:
            # Get chain heights
            chain_a_height = len(self.chains['A']._hash_index) if hasattr(self.chains['A'], '_hash_index') else 0
            chain_b_height = len(self.chains['B']._hash_index) if hasattr(self.chains['B'], '_hash_index') else 0
            
            # Load B-chain finality anchor if exists (weights chain)
            anchor_file = DATA_DIR / "finality_anchor_B.json"
            finality_anchor_height = 0
            if anchor_file.exists():
                try:
                    with open(anchor_file, 'r') as f:
                        anchor_data = json.load(f)
                        finality_anchor_height = anchor_data.get('height', 0)
                except:
                    pass
            
            # Get current VRAM status
            vram_stats = {}
            if self.gpu_available:
                try:
                    import torch
                    if torch.cuda.is_available():
                        vram_stats = {
                            'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                            'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
                        }
                except:
                    pass
            
            # Get background auditor stats
            auditor_stats = {}
            if self.background_auditor:
                auditor_stats = self.background_auditor.get_stats()
            
            # Create health summary
            health_summary = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'node_id': self.node_id,
                'chain_a_height': chain_a_height,
                'chain_b_height': chain_b_height,
                'finality_anchor_height': finality_anchor_height,
                'header_index_ok': (DATA_DIR / "B" / "headers.idx.jsonl").exists(),
                'tail_verified': int(os.getenv('TAIL_VERIFY_DEPTH', '128')),
                'hosted_layers': layer_manifest.get('hosted_count', 0),
                'hosted_layers_verified': layer_manifest.get('verified_count', 0),
                'ready_to_serve': layer_manifest.get('ready_to_serve', False),
                'gpu_available': self.gpu_available,
                'model_ready': self.model_manager is not None,
                'integrity_valid': progress.get('integrity_valid', False),
                'progress_percentage': progress.get('progress_percentage', 0),
                'vram_ok': self.vram_healthy,
                'vram_oom_count': self.vram_oom_count,
                'degraded_mode': self.degraded_mode,
                'vram_stats': vram_stats,
                'background_auditor': auditor_stats
            }
            
            # Save to file
            health_file = DATA_DIR / "node_health.json"
            with open(health_file, 'w') as f:
                json.dump(health_summary, f, indent=2)
            
            # Log summary
            logger.info("📊 Node Health Summary:")
            logger.info(f"  Chain A: {chain_a_height} blocks")
            logger.info(f"  Chain B: {chain_b_height} blocks")
            logger.info(f"  Finality anchor: {finality_anchor_height}")
            logger.info(f"  Hosted layers: {health_summary['hosted_layers']}")
            logger.info(f"  Verified layers: {health_summary['hosted_layers_verified']}")
            logger.info(f"  Ready to serve: {'✅' if health_summary['ready_to_serve'] else '❌'}")
            
        except Exception as e:
            logger.error(f"Failed to generate health summary: {e}")
    
    async def shutdown(self):
        """Clean shutdown of the node."""
        logger.info("🛑 Shutting down GPU node...")
        
        try:
            # Stop admission scheduler
            if hasattr(self, '_admission_task') and self._admission_task:
                self._admission_task.cancel()
                try:
                    await self._admission_task
                except asyncio.CancelledError:
                    pass
                logger.info("   Admission scheduler stopped")
            
            # Shutdown queue manager
            if hasattr(self, 'queue_manager') and self.queue_manager:
                await self.queue_manager.shutdown()
                logger.info("   Queue manager shut down")
                # Clear any stored prompts
                try:
                    if hasattr(self, '_queued_prompts') and isinstance(self._queued_prompts, dict):
                        cleared = len(self._queued_prompts)
                        self._queued_prompts.clear()
                        if cleared:
                            logger.info(f"   Cleared {cleared} queued prompt(s)")
                except Exception as e:
                    logger.debug(f"   Prompt map cleanup error: {e}")
            
            # Stop background auditor
            if hasattr(self, 'background_auditor') and self.background_auditor:
                await self.background_auditor.stop()
                logger.info("   Background auditor stopped")
            
            # Stop the HTTP server if running
            if hasattr(self, '_site') and self._site:
                await self._site.stop()
                logger.info("   HTTP server stopped")
            
            if hasattr(self, '_runner') and self._runner:
                await self._runner.cleanup()
                logger.info("   Runner cleaned up")
            
            # Clean up model resources
            if hasattr(self, 'model_manager') and self.model_manager:
                # Release GPU memory
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("   GPU memory released")
            
            logger.info("✅ Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global node instance for emergency cleanup
_global_node = None

def emergency_cleanup():
    """Synchronous emergency cleanup for atexit."""
    global _global_node
    if _global_node:
        logger.info("🛑 Emergency cleanup triggered")
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                loop.run_until_complete(_global_node.shutdown())
            else:
                # If no loop, at least try to cleanup what we can
                logger.warning("   No event loop available for async cleanup")
                # Release GPU memory if possible
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"   Emergency cleanup error: {e}")

async def main():
    """Entry point."""
    global _global_node
    node = BlyanGPUNode()
    _global_node = node
    
    # Register emergency cleanup
    atexit.register(emergency_cleanup)
    
    # Also handle SIGTERM for container shutdown
    def sigterm_handler(signum, _frame):
        logger.warning(f"⚠️ Received signal {signum}")
        raise KeyboardInterrupt()
    
    sig.signal(sig.SIGTERM, sigterm_handler)
    
    try:
        await node.run()
    except KeyboardInterrupt:
        logger.info("⚠️ Shutdown requested")
        await node.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await node.shutdown()
        raise
    finally:
        _global_node = None  # Clear reference after cleanup

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("✋ Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
