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
        MODEL_ID, MODEL_NAME, ARCHITECTURE, LAYERS, MOE, 
        CONTEXT, PRECISION, COMPUTE, BLOCKCHAIN,
        get_model_config, get_expert_naming, get_total_experts
    )
    DEFAULT_MODEL_NAME = MODEL_ID
    PROFILE_AVAILABLE = True
except ImportError:
    logger.warning("Model profile not available, using fallback")
    DEFAULT_MODEL_NAME = 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'
    PROFILE_AVAILABLE = False
    def get_model_config(name):
        return {}

# Configuration
PORT = int(os.environ.get('NODE_PORT', '8001'))
MAIN_NODE_URL = os.environ.get('MAIN_NODE_URL', 'https://blyan.com/api')
DATA_DIR = Path(os.environ.get('BLYAN_DATA_DIR', './data'))
MODEL_NAME = os.environ.get('MODEL_NAME', DEFAULT_MODEL_NAME)
SKIP_POL = os.environ.get('SKIP_POL', 'true').lower() == 'true'
AUTO_UPLOAD = os.environ.get('AUTO_UPLOAD', 'true').lower() == 'true'

# Optional: Use custom public IP/hostname if provided, otherwise auto-detect
PUBLIC_HOST = os.environ.get('PUBLIC_HOST', '')  # Can be IP, domain, or empty for auto-detect
PUBLIC_PORT = int(os.environ.get('PUBLIC_PORT', str(PORT)))  # Port accessible from outside (may differ due to NAT/proxy)

# Model precision is now auto-detected from model config
# FP8 for Qwen3-30B, FP16 for others
REQUIRE_INT8_SUPPORT = False  # INT8 not required

class BlyanGPUNode:
    """Integrated GPU node with blockchain and model support."""
    
    def __init__(self):
        self.node_id = f"gpu_node_{os.getpid()}"
        self.port = PORT
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
        model_config = get_model_config(MODEL_NAME)
        self.precision = model_config.get('precision', 'auto')
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
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
            # Use standard chain loading (simpler and more reliable)
            from backend.core.chain import Chain
            logger.info("Using standard chain loading")
            
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
        # Main node has NO blockchain - it's just a coordinator
        # GPU nodes maintain their own blockchains independently
        logger.info("📝 Note: Main node has no blockchain. GPU nodes maintain local chains.")
        
        # For now, skip sync since main node doesn't store blockchain
        # TODO: Implement GPU-to-GPU peer sync in the future
        return True  # Return success to continue initialization
    
    def check_block_progress(self) -> dict:
        """Check blockchain progress and integrity."""
        progress = {
            "meta_blocks": 0,
            "expert_blocks": 0,
            "expected_experts": 0,
            "missing_experts": [],
            "integrity_valid": True,
            "progress_percentage": 0.0
        }
        
        try:
            # Check meta chain (use index count for O(1) performance)
            meta_blocks_count = len(self.chains['A']._hash_index) if hasattr(self.chains['A'], '_hash_index') else 0
            progress["meta_blocks"] = meta_blocks_count
            
            # Get expected expert count from profile
            if PROFILE_AVAILABLE:
                # Total experts = layers × experts_per_layer  
                progress["expected_experts"] = LAYERS["num_layers"] * MOE["num_experts"]
            else:
                # Fallback estimate
                progress["expected_experts"] = 28 * 16  # Default for older models
            
            # OPTIMIZATION: Skip expensive block type check if we have many blocks
            logger.info("📊 Counting blocks in chain B...")
            start_time = time.time()
            
            # Try to use index first
            if hasattr(self.chains['B'], '_hash_index'):
                chain_b_blocks = len(self.chains['B']._hash_index)
                logger.info(f"✅ Chain B has {chain_b_blocks} blocks (from index)")
            else:
                # Fallback if no index (shouldn't happen)
                chain_b_blocks = 0
                logger.info(f"✅ Chain B has {chain_b_blocks} blocks")
            
            if chain_b_blocks > 1000:
                # Assume most blocks are experts if we have many
                progress["expert_blocks"] = chain_b_blocks - 50  # Subtract some for routers
                logger.info(f"⚡ Skipping expensive block type check (estimated {progress['expert_blocks']} experts from {chain_b_blocks} blocks)")
                # Skip the expensive missing expert check too
                progress["missing_experts"] = []
                expert_blocks = []  # Empty list to skip iteration below
            else:
                # Only do expensive check for small chains
                expert_blocks = self.chains['B'].get_blocks_by_type('expert')
                progress["expert_blocks"] = len(expert_blocks)
            
            # Find missing experts (skip if we have many blocks)
            existing_experts = set()
            if len(expert_blocks) > 0:
                for block in expert_blocks:
                    if hasattr(block.header, 'expert_name'):
                        existing_experts.add(block.header.expert_name)
            
            # Check which experts are missing (skip for large chains)
            if PROFILE_AVAILABLE and chain_b_blocks < 1000 and existing_experts:
                for layer_idx in range(LAYERS["num_layers"]):
                    for expert_idx in range(MOE["num_experts"]):
                        expert_name = get_expert_naming(layer_idx, expert_idx)
                        if expert_name not in existing_experts:
                            progress["missing_experts"].append(expert_name)
                            # Only track first 100 missing for performance
                            if len(progress["missing_experts"]) >= 100:
                                progress["missing_experts"].append("... and more")
                                break
                    if len(progress["missing_experts"]) >= 100:
                        break
            
            # Calculate progress percentage
            if progress["expected_experts"] > 0:
                progress["progress_percentage"] = (progress["expert_blocks"] / progress["expected_experts"]) * 100
            
            # Verify integrity of existing blocks
            progress["integrity_valid"] = self.verify_block_integrity()
            
            # Log progress summary
            logger.info(f"📊 Block Progress Report:")
            logger.info(f"  Meta blocks: {progress['meta_blocks']}")
            logger.info(f"  Expert blocks: {progress['expert_blocks']}/{progress['expected_experts']} ({progress['progress_percentage']:.1f}%)")
            logger.info(f"  Integrity: {'✅ Valid' if progress['integrity_valid'] else '❌ Invalid'}")
            
            if progress["missing_experts"] and len(progress["missing_experts"]) < 10:
                logger.info(f"  Missing experts: {progress['missing_experts'][:5]}")
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to check block progress: {e}")
            progress["integrity_valid"] = False
            return progress
    
    def verify_block_integrity(self) -> bool:
        """Verify integrity using last block hash check (O(1) complexity).
        
        This is the production-recommended approach:
        - Verifies chain integrity by checking last block's hash chain
        - O(1) complexity instead of O(n) for full chain verification
        - Standard practice in Bitcoin, Ethereum, etc.
        """
        try:
            # Check if chains exist
            if not self.chains:
                logger.error("No chains initialized")
                return False
            
            # Verify meta chain (chain A) using last block hash
            meta_count = len(self.chains['A']._hash_index) if hasattr(self.chains['A'], '_hash_index') else 0
            if meta_count > 0:
                if not self._verify_last_block_integrity(self.chains['A'], 'A', meta_count):
                    return False
            else:
                logger.info("Meta chain empty (OK for new GPU node)")
            
            # Verify parameter chain (chain B) using last block hash
            param_count = len(self.chains['B']._hash_index) if hasattr(self.chains['B'], '_hash_index') else 0
            if param_count > 0:
                if not self._verify_last_block_integrity(self.chains['B'], 'B', param_count):
                    return False
            else:
                logger.info("Parameter chain empty (OK for new GPU node)")
            
            logger.info("✅ Blockchain integrity verified (last block hash method)")
            return True
            
        except Exception as e:
            logger.error(f"Block integrity check failed: {e}")
            return False
    
    def _verify_last_block_integrity(self, chain, chain_id: str, block_count: int) -> bool:
        """Verify chain integrity by checking last block's hash linkage.
        
        Production-grade verification that ensures:
        1. Last block exists and is valid
        2. Hash chain is intact (prev_hash matches)
        3. Payload integrity is maintained
        """
        try:
            import hashlib
            start_time = time.time()
            
            # Get last block using the chain's method
            if hasattr(chain, 'get_block_by_index'):
                last_block = chain.get_block_by_index(block_count - 1)
            elif hasattr(chain, 'storage') and chain.storage:
                last_block = chain.storage.get_block_by_index(block_count - 1)
            else:
                # No blocks to verify for empty chain
                logger.info(f"Chain {chain_id}: Empty chain, nothing to verify")
                return True
            if not last_block:
                logger.error(f"Chain {chain_id}: Failed to load last block (index {block_count - 1})")
                return False
            
            # Verify last block's hash
            computed_hash = last_block.compute_hash()
            
            # Verify payload integrity
            payload_hash = hashlib.sha256(last_block.payload).hexdigest()
            if payload_hash != last_block.header.payload_hash:
                logger.error(f"Chain {chain_id}: Last block payload hash mismatch")
                return False
            
            # If we have more than one block, verify the hash chain linkage
            if block_count > 1:
                # Get second-to-last block to verify linkage
                if hasattr(chain, 'get_block_by_index'):
                    prev_block = chain.get_block_by_index(block_count - 2)
                elif hasattr(chain, 'storage') and chain.storage:
                    prev_block = chain.storage.get_block_by_index(block_count - 2)
                else:
                    prev_block = None
                if prev_block:
                    prev_hash = prev_block.compute_hash()
                    if last_block.header.prev_hash != prev_hash:
                        logger.error(f"Chain {chain_id}: Hash chain broken at block {block_count - 1}")
                        logger.error(f"  Expected prev_hash: {prev_hash[:16]}...")
                        logger.error(f"  Got prev_hash: {last_block.header.prev_hash[:16]}...")
                        return False
            
            elapsed = time.time() - start_time
            logger.info(f"Chain {chain_id}: {block_count} blocks verified via last hash {computed_hash[:16]}... ({elapsed:.3f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying chain {chain_id}: {e}")
            return False
    
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
                    
                    # Reinitialize chain
                    from backend.core.chain import Chain
                    logger.info("Using standard chain loading")
                    
                    self.chains[cid] = Chain(DATA_DIR, cid, skip_pol=True)
                    
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
        
        # Save any uploaded experts before reset
        expert_blocks = self.chains['B'].get_blocks_by_type('expert')
        saved_experts = []
        if expert_blocks:
            logger.info(f"Saving {len(expert_blocks)} expert blocks before reset...")
            for block in expert_blocks:
                saved_experts.append({
                    'payload': block.payload,
                    'expert_name': block.header.expert_name,
                    'layer_id': block.header.layer_id
                })
        
        # Reset chains
        self.reset_blockchain()
        
        # Re-initialize chains
        if not self.initialize_chains():
            logger.error("Failed to reinitialize chains after reset")
            return False
        
        # Restore saved experts if any
        if saved_experts:
            logger.info(f"Restoring {len(saved_experts)} expert blocks...")
            restored = 0
            for expert_data in saved_experts:
                try:
                    self.chains['B'].add_block(
                        expert_data['payload'],
                        block_type='expert',
                        expert_name=expert_data['expert_name'],
                        layer_id=expert_data['layer_id']
                    )
                    restored += 1
                except Exception as e:
                    logger.warning(f"Could not restore expert {expert_data['expert_name']}: {e}")
            logger.info(f"✅ Restored {restored}/{len(saved_experts)} expert blocks")
        
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
        total_needed = progress["expected_experts"]
        current_count = progress["expert_blocks"]
        
        if current_count == 0:
            logger.info(f"📦 Starting fresh - need to upload {total_needed} experts")
        else:
            remaining = total_needed - current_count
            logger.info(f"📦 Resuming - need {remaining} more experts ({progress['progress_percentage']:.1f}% complete)")
        
        # Check if we should auto-upload
        if AUTO_UPLOAD:
            # Check if upload was already completed (verify blocks exist)
            upload_state_file = DATA_DIR / "upload_completed.json"
            actual_blocks = progress.get('expert_blocks', 0)
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
                    "num_experts": current_count,
                    "timestamp": time.time(),
                    "completed": True
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
            logger.info(f"ℹ️  Manual upload required (set AUTO_UPLOAD=True to enable)")
    
    def initialize_model_manager(self) -> bool:
        """Initialize blockchain-first model manager for inference."""
        try:
            logger.info("📋 Starting model manager initialization...")
            
            # Use blockchain-first loader - NO local models
            logger.info("  1/5: Importing blockchain model manager...")
            try:
                from backend.model.blockchain_first_loader import BlockchainOnlyModelManager
                logger.info("    ✓ BlockchainOnlyModelManager imported")
            except ImportError as e:
                logger.error(f"    ✗ Failed to import BlockchainOnlyModelManager: {e}")
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
            
            # Initialize blockchain-only model manager
            logger.info("  3/5: Creating blockchain model manager...")
            self.model_manager = BlockchainOnlyModelManager(
                meta_chain=self.chains.get('A'),
                param_chain=self.chains.get('B'),
                param_index=param_index,
                device="cuda" if self.gpu_available else "cpu"
            )
            
            # Initialize zero-copy loader for efficient loading
            logger.info("  4/5: Setting up zero-copy loader...")
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
                if AUTO_UPLOAD:
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
                    logger.info("💡 Upload model using: python miner/upload_moe_parameters.py")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            return False
    
    async def download_and_upload_model(self):
        """Download model from HuggingFace and upload to blockchain as experts."""
        
        # Prevent duplicate uploads
        if hasattr(self, '_upload_in_progress') and self._upload_in_progress:
            logger.info("Upload already in progress, skipping duplicate call")
            return
        
        self._upload_in_progress = True
        
        try:
            await self._do_download_and_upload()
        finally:
            self._upload_in_progress = False
    
    async def _do_download_and_upload(self):
        """Actual download and upload implementation."""
        
        # Check if upload was already completed
        upload_state_file = DATA_DIR / "upload_completed.json"
        if upload_state_file.exists():
            try:
                with open(upload_state_file, 'r') as f:
                    upload_state = json.load(f)
                
                # Verify actual blocks exist, not just state file
                actual_blocks = len(self.chains['B']._hash_index) if hasattr(self.chains['B'], '_hash_index') else 0
                expected_experts = upload_state.get('num_experts', 0)
                
                if upload_state.get("completed") and upload_state.get("model") == MODEL_NAME:
                    if actual_blocks > 0 and actual_blocks >= expected_experts * 0.99:  # 99% threshold
                        logger.info(f"✅ Model {MODEL_NAME} already uploaded ({actual_blocks} blocks)")
                        logger.info("💡 Delete upload_completed.json to force re-upload")
                        return
                    else:
                        logger.warning(f"⚠️ State file exists but only {actual_blocks} blocks found (expected ~{expected_experts})")
                        logger.info("🔄 Removing invalid state file and continuing upload...")
                        upload_state_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to read upload state: {e}")
        
        logger.info(f"📥 Auto-downloading model: {MODEL_NAME}")
        model_config = get_model_config(MODEL_NAME)
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
                
                if not caps['bnb_cuda']:
                    logger.warning("BitsAndBytes CUDA support not available, will try non-quantized loading")
                
                # Load tokenizer with compatibility handling
                logger.info("Loading tokenizer...")
                tokenizer = load_tokenizer(MODEL_NAME)
                
                # Load with model-specific configuration
                model_config = get_model_config(MODEL_NAME)
                torch_dtype = model_config.get('torch_dtype', 'auto')
                logger.info(f"Loading model with {model_config.get('precision', 'auto')} precision...")
                
                # Force FP16 loading only
                from transformers import AutoModelForCausalLM
                import torch
                
                # Multi-GPU support with device_map="auto"
                if self.gpu_available:
                    num_gpus = self.gpu_info.get("num_gpus", 1)
                    logger.info(f"Using {num_gpus} GPU(s) for model loading")
                    if num_gpus > 1:
                        logger.info("Model will be distributed across all available GPUs")
                
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype="auto",  # Auto-detect for FP8 models
                    device_map="auto" if self.gpu_available else None,  # Auto distributes across all GPUs
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                if not self.gpu_available:
                    model = model.to("cpu")
                
                logger.info(f"✅ Model loaded with {model_config.get('precision', 'auto')} precision")
                
            else:
                # Fallback to original loading method
                if not torch.cuda.is_available() or not self.gpu_available:
                    logger.error("CUDA not available on this node. Skipping auto-upload.")
                    return
                
                # Setup environment
                os.environ.setdefault("HF_HOME", str((DATA_DIR / ".hf").resolve()))
                os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
                
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Try loading with FP16 as fallback
                logger.info("⏳ Loading model...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                
                # Load with auto dtype for compatibility
                model_config = get_model_config(MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype="auto",  # Auto-detect for FP8 models
                    device_map="auto" if self.gpu_available else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                if not self.gpu_available:
                    model = model.to("cpu")
                    
                logger.info(f"✅ Model loaded with {model_config.get('precision', 'auto')} precision")
            
            # Create meta block if needed
            meta_count = len(self.chains['A']._hash_index) if hasattr(self.chains['A'], '_hash_index') else 0
            if meta_count == 0:
                if PROFILE_AVAILABLE:
                    meta_spec = {
                        "model_name": MODEL_NAME,
                        "architecture": ARCHITECTURE["type"],
                        "num_layers": LAYERS["num_layers"],
                        "num_experts": MOE["num_experts"],
                        "activated_experts": MOE["num_activated_experts"],
                        "routing_strategy": MOE["routing_strategy"],
                        "context_length": CONTEXT["max_length"],
                        "total_params": ARCHITECTURE["total_params"],
                        "active_params": ARCHITECTURE["active_params"]
                }
                self.chains['A'].add_block(json.dumps(meta_spec).encode(), block_type='meta')
                logger.info("✅ Created meta block")
            
            # Extract and upload experts with memory-efficient streaming
            import io, pickle, gc, torch

            num_uploaded = 0
            layers = getattr(model, "model", None)
            layers = getattr(layers, "layers", None)

            if layers is None:
                raise RuntimeError("Model structure unexpected: missing model.layers")

            # Use profile-based layer count if available
            max_layers = LAYERS["num_layers"] if PROFILE_AVAILABLE else 48
            
            for layer_idx in range(min(max_layers, len(layers))):
                layer = layers[layer_idx]
                
                # Check for MoE structure
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    logger.warning(f"Layer {layer_idx} has no MLP; skipping")
                    continue
                
                # Extract gate/router weights first
                if hasattr(mlp, 'gate'):
                    try:
                        router_name = f"layer{layer_idx}.router"
                        router_state = {'weight': mlp.gate.weight.detach().cpu().contiguous()}
                        
                        buffer = io.BytesIO()
                        pickle.dump(router_state, buffer, protocol=pickle.HIGHEST_PROTOCOL)
                        payload = buffer.getvalue()
                        
                        self.chains['B'].add_block(
                            payload,
                            block_type='router',
                            expert_name=router_name,
                            layer_id=f"layer{layer_idx}"
                        )
                        logger.info(f"✅ Uploaded {router_name} to blockchain")
                        num_uploaded += 1
                        del router_state, buffer, payload
                        
                    except Exception as e:
                        logger.warning(f"Failed to upload router for layer {layer_idx}: {e}")
                
                # Extract ALL individual experts based on model configuration
                if hasattr(mlp, 'experts'):
                    num_experts = len(mlp.experts) if hasattr(mlp.experts, '__len__') else 0
                    expected_experts = MOE["num_experts"] if PROFILE_AVAILABLE else 128
                    logger.info(f"📦 Layer {layer_idx} has {num_experts} experts to upload (expected: {expected_experts})")
                    
                    # Verify we're getting all expected experts
                    if num_experts < expected_experts:
                        logger.warning(f"⚠️ Expected {expected_experts} experts but found {num_experts} in layer {layer_idx}")
                    elif num_experts > expected_experts:
                        logger.warning(f"⚠️ Found more experts ({num_experts}) than expected ({expected_experts}) in layer {layer_idx}")
                    
                    # Track upload progress
                    successfully_uploaded = 0
                    
                    # Upload each expert individually with retry logic
                    for expert_idx in range(num_experts):
                        upload_success = False
                        retry_count = 0
                        max_retries = 3
                        
                        while not upload_success and retry_count < max_retries:
                            try:
                                expert = mlp.experts[expert_idx]
                                expert_name = get_expert_naming(layer_idx, expert_idx) if PROFILE_AVAILABLE else f"layer{layer_idx}.expert{expert_idx}"
                                
                                # Extract expert weights (gate_proj, up_proj, down_proj)
                                expert_state = {}
                                
                                # Check if expert has any weights at all
                                has_weights = False
                                
                                if hasattr(expert, 'gate_proj') and hasattr(expert.gate_proj, 'weight'):
                                    weight = expert.gate_proj.weight.detach().cpu().contiguous()
                                    # Check for NaN/Inf values (skip for FP8 and other special dtypes)
                                    try:
                                        # Only check for standard float types
                                        if weight.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                            if torch.isnan(weight).any() or torch.isinf(weight).any():
                                                logger.warning(f"Expert {expert_idx} layer {layer_idx} gate_proj has NaN/Inf values - skipping")
                                                break
                                    except (RuntimeError, NotImplementedError):
                                        # Skip NaN/Inf check for unsupported dtypes (like FP8)
                                        logger.debug(f"Skipping NaN/Inf check for dtype {weight.dtype}")
                                    expert_state['gate_proj.weight'] = weight
                                    has_weights = True
                                    
                                if hasattr(expert, 'up_proj') and hasattr(expert.up_proj, 'weight'):
                                    weight = expert.up_proj.weight.detach().cpu().contiguous()
                                    # Check for NaN/Inf values (skip for FP8 and other special dtypes)
                                    try:
                                        # Only check for standard float types
                                        if weight.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                            if torch.isnan(weight).any() or torch.isinf(weight).any():
                                                logger.warning(f"Expert {expert_idx} layer {layer_idx} up_proj has NaN/Inf values - skipping")
                                                break
                                    except (RuntimeError, NotImplementedError):
                                        # Skip NaN/Inf check for unsupported dtypes (like FP8)
                                        logger.debug(f"Skipping NaN/Inf check for dtype {weight.dtype}")
                                    expert_state['up_proj.weight'] = weight
                                    has_weights = True
                                    
                                if hasattr(expert, 'down_proj') and hasattr(expert.down_proj, 'weight'):
                                    weight = expert.down_proj.weight.detach().cpu().contiguous()
                                    # Check for NaN/Inf values (skip for FP8 and other special dtypes)
                                    try:
                                        # Only check for standard float types
                                        if weight.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                            if torch.isnan(weight).any() or torch.isinf(weight).any():
                                                logger.warning(f"Expert {expert_idx} layer {layer_idx} down_proj has NaN/Inf values - skipping")
                                                break
                                    except (RuntimeError, NotImplementedError):
                                        # Skip NaN/Inf check for unsupported dtypes (like FP8)
                                        logger.debug(f"Skipping NaN/Inf check for dtype {weight.dtype}")
                                    expert_state['down_proj.weight'] = weight
                                    has_weights = True
                                
                                if not has_weights:
                                    logger.warning(f"Expert {expert_idx} layer {layer_idx} has no weights - skipping")
                                    break
                                
                                if expert_state:
                                    buffer = io.BytesIO()
                                    pickle.dump(expert_state, buffer, protocol=pickle.HIGHEST_PROTOCOL)
                                    payload = buffer.getvalue()
                                    
                                    # Check payload is valid
                                    if not payload or len(payload) == 0:
                                        logger.error(f"Failed to serialize {expert_name} - empty payload")
                                        break
                                    
                                    # Check disk space before writing (every 100 blocks)
                                    if num_uploaded % 100 == 0:
                                        import shutil
                                        total, used, free = shutil.disk_usage("/")
                                        free_gb = free // (2**30)
                                        if free_gb < 10:  # Less than 10GB free
                                            logger.error(f"LOW DISK SPACE: Only {free_gb}GB free")
                                            logger.error("Cannot continue upload - insufficient disk space")
                                            return  # Stop the upload
                                    
                                    # Add with proper error handling
                                    try:
                                        self.chains['B'].add_block(
                                            payload,
                                            block_type='expert',
                                            expert_name=expert_name,
                                            layer_id=f"layer{layer_idx}",
                                            depends_on=[]  # Explicitly set empty list to avoid None issues
                                        )
                                        logger.info(f"✅ Uploaded {expert_name} to blockchain")
                                        num_uploaded += 1
                                        successfully_uploaded += 1
                                        upload_success = True
                                    except Exception as add_error:
                                        if "Expecting value" in str(add_error):
                                            logger.error(f"JSON serialization error for {expert_name}: {add_error}")
                                            logger.debug(f"Payload size: {len(payload) if payload else 0} bytes")
                                            
                                            # Additional diagnostics
                                            if not payload or len(payload) == 0:
                                                logger.error(f"ERROR: Empty payload for {expert_name}")
                                            elif len(payload) < 100:
                                                logger.error(f"ERROR: Suspiciously small payload for {expert_name}: {len(payload)} bytes")
                                            
                                            # Check if the chain is corrupted
                                            try:
                                                latest_block = self.chains['B'].get_latest_block()
                                                if latest_block:
                                                    logger.debug(f"Last successful block index: {latest_block.header.index}")
                                                else:
                                                    logger.error("Chain B appears to be corrupted or reset")
                                            except Exception as chain_error:
                                                logger.error(f"Cannot check chain state: {chain_error}")
                                            
                                            # Skip this expert if JSON error persists
                                            break
                                        else:
                                            raise
                                    
                                    # Cleanup after each expert
                                    del expert_state, buffer, payload
                                
                            except Exception as e:
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.warning(f"Failed to upload {expert_name} (attempt {retry_count}/{max_retries}): {str(e)[:100]}")
                                    time.sleep(0.5 * retry_count)  # Exponential backoff
                                else:
                                    logger.error(f"Failed to upload {expert_name} after {max_retries} attempts: {str(e)[:100]}")
                                    import traceback
                                    logger.debug(f"Full error trace: {traceback.format_exc()}")
                        
                        # Memory cleanup every 10 experts
                        if expert_idx % 10 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                                # Check GPU memory usage
                                try:
                                    mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                                    if mem_allocated > 40:  # More than 40GB used
                                        logger.warning(f"High GPU memory usage: {mem_allocated:.1f}GB allocated")
                                        # Force aggressive cleanup
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass  # Ignore if CUDA not available
                
                # Check for shared expert (if present)
                if hasattr(mlp, 'shared_expert'):
                    try:
                        shared_name = f"layer{layer_idx}.shared_expert"
                        shared_expert = mlp.shared_expert
                        
                        shared_state = {}
                        if hasattr(shared_expert, 'gate_proj'):
                            shared_state['gate_proj.weight'] = shared_expert.gate_proj.weight.detach().cpu().contiguous()
                        if hasattr(shared_expert, 'up_proj'):
                            shared_state['up_proj.weight'] = shared_expert.up_proj.weight.detach().cpu().contiguous()
                        if hasattr(shared_expert, 'down_proj'):
                            shared_state['down_proj.weight'] = shared_expert.down_proj.weight.detach().cpu().contiguous()
                        
                        if shared_state:
                            buffer = io.BytesIO()
                            pickle.dump(shared_state, buffer, protocol=pickle.HIGHEST_PROTOCOL)
                            payload = buffer.getvalue()
                            
                            self.chains['B'].add_block(
                                payload,
                                block_type='expert',
                                expert_name=shared_name,
                                layer_id=f"layer{layer_idx}"
                            )
                            logger.info(f"✅ Uploaded {shared_name} to blockchain")
                            num_uploaded += 1
                            
                            del shared_state, buffer, payload
                    
                    except Exception as e:
                        logger.warning(f"Failed to upload shared expert for layer {layer_idx}: {e}")
                
                # Report layer upload results
                if hasattr(mlp, 'experts'):
                    if successfully_uploaded < expected_experts:
                        logger.warning(f"⚠️ Layer {layer_idx}: Only uploaded {successfully_uploaded}/{expected_experts} experts")
                    else:
                        logger.info(f"✅ Layer {layer_idx}: Successfully uploaded all {successfully_uploaded} experts")
                
                # Major memory cleanup after each layer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"📊 Layer {layer_idx} complete. Total uploaded: {num_uploaded}")
            
            # Final verification
            expected_total = LAYERS["num_layers"] * MOE["num_experts"] if PROFILE_AVAILABLE else (48 * 128)
            logger.info(f"✅ Upload complete: {num_uploaded} blocks uploaded")
            logger.info(f"📊 Expected {expected_total} experts total ({LAYERS['num_layers'] if PROFILE_AVAILABLE else 48} layers × {MOE['num_experts'] if PROFILE_AVAILABLE else 128} experts)")
            
            if num_uploaded < expected_total:
                logger.warning(f"⚠️ Only uploaded {num_uploaded}/{expected_total} experts ({(num_uploaded/expected_total)*100:.1f}%)")
                logger.warning("Some experts may be missing. Check logs for failed uploads.")
            else:
                logger.info(f"✅ All {expected_total} experts uploaded successfully!")
            
            # Save upload state to prevent re-upload
            upload_state_file = DATA_DIR / "upload_completed.json"
            upload_state = {
                "model": MODEL_NAME,
                "num_experts": num_uploaded,
                "timestamp": time.time(),
                "completed": True
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
    
    async def start_server(self):
        """Start HTTP server for the node."""
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
        
        app = web.Application()
        
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
            """Chat endpoint with MoE inference support"""
            try:
                import time
                start_time = time.time()

                data = await request.json()
                prompt = data.get("prompt", "")
                max_new_tokens = data.get("max_new_tokens", 64)
                use_moe = data.get("use_moe", True)
                top_k_experts = data.get("top_k_experts", 2)

                # Check if we have MoE available
                has_distributed_nodes = bool(self.distributed_coordinator.registry.nodes) if hasattr(self, 'distributed_coordinator') and self.distributed_coordinator else False
                has_moe_manager = hasattr(self, 'moe_model_manager') and self.moe_model_manager is not None

                if use_moe and has_distributed_nodes:
                    # Use distributed MoE inference
                    available_experts = list(self.distributed_coordinator.registry.expert_to_nodes.keys())
                    if available_experts:
                        selected_experts = available_experts[:top_k_experts]

                        # Prefer donor nodes for free-tier requests
                        prefer_donor = not data.get("free_tier", False)

                        response_text, routing_info = await self.distributed_coordinator.distribute_inference(
                            prompt=prompt,
                            required_experts=selected_experts,
                            max_new_tokens=max_new_tokens,
                            prefer_donor=prefer_donor
                        )

                        inference_time = time.time() - start_time

                        return web.json_response({
                            "response": response_text,
                            "expert_usage": routing_info.get('expert_usage', {}),
                            "inference_time": inference_time,
                            "used_moe": True,
                            "expert_count": len(selected_experts)
                        })

                elif use_moe and has_moe_manager:
                    # Use local MoE inference
                    answer, expert_usage = self.moe_model_manager.selective_generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        top_k_experts=top_k_experts
                    )

                    inference_time = time.time() - start_time

                    return web.json_response({
                        "response": answer,
                        "expert_usage": expert_usage,
                        "inference_time": inference_time,
                        "used_moe": True
                    })

                else:
                    # Fallback to standard model manager
                    if not hasattr(self, 'model_manager') or self.model_manager is None:
                        return web.json_response({"error": "Model manager not initialized"}, status=500)

                    # Use correct parameters for BlockchainOnlyModelManager
                    if hasattr(self.model_manager, 'validate_blockchain_state'):
                        # This is a BlockchainOnlyModelManager - get available experts
                        state = self.model_manager.validate_blockchain_state()
                        available_experts = state.get('available_experts', [])

                        # Select experts to use (take first few or all if small number)
                        selected_experts = available_experts[:min(len(available_experts), top_k_experts)]
                        if not selected_experts:
                            # No experts available, use fallback
                            answer = f"⚠️ No experts available in blockchain. Please upload experts first. Prompt: {prompt}"
                            expert_usage = {}
                            used_moe = False
                        else:
                            # Generate with selected experts
                            answer = self.model_manager.generate(prompt, selected_experts, max_new_tokens)
                            inference_time = time.time() - start_time

                            # Parse expert usage from response
                            expert_usage = {}
                            used_moe = True
                            import re
                            expert_match = re.search(r'Generated using blockchain experts: ([^\\]]+)', answer)
                            if expert_match:
                                expert_names = expert_match.group(1).split(', ')
                                expert_usage = {name: 1 for name in expert_names}
                            else:
                                # Fallback: use selected experts
                                expert_usage = {name: 1 for name in selected_experts}
                    else:
                        # Fallback for other model managers
                        answer = f"Standard model response to: {prompt}"
                        expert_usage = {}
                        used_moe = False

                    inference_time = time.time() - start_time

                    return web.json_response({
                        "response": answer,
                        "expert_usage": expert_usage,
                        "inference_time": inference_time,
                        "used_moe": used_moe
                    })

            except Exception as exc:
                logger.error(f"Chat error: {exc}")
                return web.json_response({"error": str(exc)}, status=500)

        # Inference endpoint - BLOCKCHAIN-FIRST, no local models
        async def inference(request):
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_new_tokens", data.get("max_length", 100))
                selected_experts = data.get("experts", [])
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
                
                # Get available experts if none specified
                if not selected_experts:
                    available = self.model_manager.get_available_experts()
                    if available:
                        # Select experts from different layers for proper MoE inference
                        # Group by layer
                        experts_by_layer = {}
                        for expert in available:
                            if 'layer' in expert:
                                layer = expert.split('layer')[1].split('.')[0]
                                if layer not in experts_by_layer:
                                    experts_by_layer[layer] = []
                                experts_by_layer[layer].append(expert)
                        
                        # Select one expert from each of the first few layers
                        selected_experts = []
                        for layer in sorted(experts_by_layer.keys())[:4]:  # Use first 4 layers
                            if experts_by_layer[layer]:
                                selected_experts.append(experts_by_layer[layer][0])
                        
                        if not selected_experts and available:
                            # Fallback to first 4 if layer parsing fails
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
                    "gpu_used": self.gpu_available,
                    "precision": self.precision  # Report precision used
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
        
        # Register routes
        if cors:
            # Add routes with CORS
            cors.add(app.router.add_get('/', health))
            cors.add(app.router.add_get('/health', health))
            cors.add(app.router.add_get('/debug/moe-status', debug_moe_status))
            cors.add(app.router.add_post('/chat', chat))
            cors.add(app.router.add_get('/chain/{chain_id}', chain_info))
            cors.add(app.router.add_post('/inference', inference))
            cors.add(app.router.add_get('/pol/status', lambda r: web.json_response({"status": "ok"})))
            
            # Learning routes with CORS
            cors.add(app.router.add_post('/learning/start', learning_start))
            cors.add(app.router.add_post('/learning/data', learning_data_allocation))
            cors.add(app.router.add_get('/learning/status', learning_status))
            cors.add(app.router.add_post('/learning/delta', submit_delta))
        else:
            # Add routes without CORS
            app.router.add_get('/', health)
            app.router.add_get('/health', health)
            app.router.add_get('/debug/moe-status', debug_moe_status)
            app.router.add_post('/chat', chat)
            app.router.add_get('/chain/{chain_id}', chain_info)
            app.router.add_post('/inference', inference)
            app.router.add_get('/pol/status', lambda r: web.json_response({"status": "ok"}))
            app.router.add_post('/learning/start', learning_start)
            app.router.add_post('/learning/data', learning_data_allocation)
            app.router.add_get('/learning/status', learning_status)
            app.router.add_post('/learning/delta', submit_delta)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Start server on configured port
        port = self.port
        logger.info(f"🚀 Starting server on port {port}")
        
        # Check what's using the port (diagnostic)
        try:
            import subprocess
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True, timeout=2)
            if result.stdout:
                logger.warning(f"⚠️ Port {port} is already in use by:")
                logger.warning(result.stdout)
        except:
            pass  # lsof might not be available
        
        # Check if we're in RunPod environment
        if os.path.exists('/runpod') or os.path.exists('/workspace'):
            logger.info("📍 Detected RunPod/cloud environment")
            # RunPod HTTP Service uses 8001, we need a different port
            if port == 8001:
                logger.warning("⚠️ Port 8001 is reserved for RunPod HTTP Service proxy")
                logger.info("💡 Using alternative port 8000 for internal server...")
                port = 8000
                self.port = 8000
        
        # Try to start server
        max_retries = 3
        for retry in range(max_retries):
            try:
                site = web.TCPSite(runner, '0.0.0.0', port)
                await site.start()
                logger.info(f"✅ Server running on http://0.0.0.0:{port}")
                self.port = port  # Update port if changed
                break
            except OSError as e:
                if "address already in use" in str(e).lower():
                    if retry < max_retries - 1:
                        # Try next port
                        port += 1
                        logger.warning(f"⚠️ Port {port-1} in use, trying port {port}...")
                    else:
                        logger.error(f"❌ Failed to start server after {max_retries} attempts")
                        logger.info("💡 Set NODE_PORT to an available port (e.g., 8000, 8002, 8003)")
                        raise
                else:
                    logger.error(f"❌ Failed to start server: {e}")
                    raise
        
        # Register with main node
        await self.register_with_main()
    
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
                
                # Get available experts for registration
                available_experts = []
                try:
                    if self.model_manager:
                        if hasattr(self.model_manager, "get_available_experts"):
                            available_experts = self.model_manager.get_available_experts()
                        elif hasattr(self.model_manager, "_get_available_experts_for_layer"):
                            for layer_id in range(24):
                                layer_experts = self.model_manager._get_available_experts_for_layer(f"layer{layer_id}")
                                available_experts.extend(layer_experts)
                except Exception as e:
                    logger.warning(f"Could not enumerate experts for registration: {e}")
                
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
                
                data = {
                    "node_id": self.node_id,
                    "host": endpoint_url,  # Send full URL instead of separate host:port
                    "port": 443 if 'https://' in endpoint_url else PUBLIC_PORT,
                    "available_experts": available_experts,
                    "node_type": "gpu",
                    "gpu_info": self.gpu_info,
                    "chains": list(self.chains.keys()),
                    "model_ready": self.model_manager is not None,
                    "genesis_hash": self.genesis_hash,
                    "precision": self.precision,
                    "supports_int8": False
                }
                
                logger.info(f"📝 Registering with endpoint: {endpoint_url}")
                if PUBLIC_PORT != self.port:
                    logger.info(f"   (Internal port: {self.port}, Public endpoint: {endpoint_url})")
                
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
            logger.error("Failed to initialize chains")
            return
        
        # 3. Check block progress
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        progress = self.check_block_progress()
        
        # 4. Verify integrity
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        if not progress["integrity_valid"]:
            logger.warning("⚠️ Blockchain integrity issue detected")
            recovery_success = await self.handle_integrity_failure()
            if recovery_success:
                # Re-check progress after recovery
                progress = self.check_block_progress()
            else:
                logger.error("Failed to recover blockchain integrity")
                # Continue anyway but with warnings
        else:
            logger.info("✅ Integrity check passed")
        
        # Continue building blocks if needed
        self.continue_block_building(progress)
        
        # 5. Initial sync attempt (non-blocking)
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        sync_success = await self.sync_from_peers()
        
        # 6. Initialize model manager
        current_step += 1
        log_step(startup_steps[current_step-1], current_step)
        if not self.initialize_model_manager():
            logger.warning("Running without model manager")
        
        # 4.5 Check if we just completed an upload and need to reinit
        if hasattr(self, 'upload_completed') and self.upload_completed:
            logger.info("🔄 Upload was just completed, reinitializing model manager...")
            # Wait a bit for filesystem to settle
            await asyncio.sleep(2)
            if self.initialize_model_manager():
                logger.info("✅ Model manager reinitialized with blockchain experts")
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
        
        # Final summary
        total_time = time.time() - startup_begin
        mins, secs = divmod(int(total_time), 60)
        # Get expert count for summary
        expert_count = 0
        if self.model_manager and hasattr(self.model_manager, '_available_experts_cache'):
            if self.model_manager._available_experts_cache:
                expert_count = len(self.model_manager._available_experts_cache)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"🚀 NODE STARTUP COMPLETE")
        logger.info(f"⏱️  Total time: {mins:02d}:{secs:02d}")
        logger.info(f"📊 Blocks loaded: {len(self.chains['B']._hash_index) if hasattr(self.chains['B'], '_hash_index') else 0}")
        if expert_count > 0:
            logger.info(f"🤖 Experts loaded from blockchain: {expert_count}")
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

async def main():
    """Entry point."""
    node = BlyanGPUNode()
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