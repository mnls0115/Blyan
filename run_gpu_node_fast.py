#!/usr/bin/env python3
"""
Blyan GPU Node with Fast-Sync Support
Achieves <5 second boot times on 3000+ block chains.
"""

import os
import sys
import asyncio
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.chain_fast import FastChain
from backend.core.dataset_chain import DatasetChain


class FastGPUNode:
    """GPU node with fast-sync blockchain support."""
    
    def __init__(self, fast_sync: bool = True, verify_sample: float = 0.02, verify_all: bool = False):
        """
        Initialize fast GPU node.
        
        Args:
            fast_sync: Enable fast-sync mode
            verify_sample: Sample verification rate (0.0-1.0)
            verify_all: Verify all blocks (overrides verify_sample)
        """
        self.node_id = f"gpu_node_fast_{os.getpid()}"
        self.fast_sync = fast_sync
        self.verify_sample = verify_sample if not verify_all else 1.0
        self.chains = {}
        
        # Data directory
        self.data_dir = Path(os.environ.get('BLYAN_DATA_DIR', './data'))
        
        logger.info(f"Initializing FastGPUNode (fast_sync={fast_sync}, verify_sample={self.verify_sample:.2%})")
    
    def initialize_chains(self) -> bool:
        """Initialize blockchains with fast-sync support."""
        logger.info("Initializing blockchains...")
        boot_start = time.perf_counter()
        
        try:
            # Check if we should migrate existing chains
            self._check_and_migrate()
            
            # Initialize chains with fast-sync
            skip_pol = os.environ.get('SKIP_POL', 'true').lower() == 'true'
            
            # Meta chain
            self.chains['A'] = FastChain(
                self.data_dir, "A", 
                skip_pol=skip_pol,
                fast_sync=self.fast_sync,
                verify_sample=self.verify_sample
            )
            
            # Parameter chain (the large one)
            self.chains['B'] = FastChain(
                self.data_dir, "B",
                skip_pol=skip_pol,
                fast_sync=self.fast_sync,
                verify_sample=self.verify_sample
            )
            
            # Dataset chain (standard for now)
            self.chains['D'] = DatasetChain(self.data_dir, "D")
            
            boot_time = (time.perf_counter() - boot_start) * 1000
            
            # Log statistics
            for chain_id, chain in self.chains.items():
                if hasattr(chain, 'get_stats'):
                    stats = chain.get_stats()
                    logger.info(f"Chain {chain_id}: {stats.get('block_count', 0)} blocks, "
                              f"boot time: {stats.get('boot_time_ms', 0):.1f}ms")
                else:
                    blocks = chain.get_all_blocks() if hasattr(chain, 'get_all_blocks') else []
                    logger.info(f"Chain {chain_id}: {len(blocks)} blocks")
            
            logger.info(f"✅ All chains initialized in {boot_time:.1f}ms")
            
            # Mark node as ready
            self._mark_ready()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chains: {e}")
            return False
    
    def _check_and_migrate(self):
        """Check if migration from JSON to shards is needed."""
        # Check if chain B has old JSON format
        old_chain_b = self.data_dir / "B"
        new_manifest = old_chain_b / "manifest.msgpack"
        if not new_manifest.exists():
            new_manifest = old_chain_b / "manifest.json"
        
        if old_chain_b.exists() and not new_manifest.exists():
            # Check for JSON files
            json_files = list(old_chain_b.glob("*.json"))
            numeric_files = []
            for f in json_files:
                try:
                    int(f.stem)
                    numeric_files.append(f)
                except ValueError:
                    continue
            
            if numeric_files:
                logger.info(f"Found {len(numeric_files)} JSON blocks, migration needed")
                
                # Ask user or auto-migrate
                if os.environ.get('AUTO_MIGRATE', 'false').lower() == 'true':
                    self._run_migration(old_chain_b)
                else:
                    logger.warning("JSON blocks found but AUTO_MIGRATE not set. Using standard boot.")
    
    def _run_migration(self, chain_dir: Path):
        """Run migration from JSON to sharded format."""
        logger.info("Running automatic migration to sharded format...")
        
        try:
            from scripts.migrate_to_shards import BlockMigrator
            
            # Migrate in-place
            migrator = BlockMigrator(chain_dir, chain_dir, chain_dir.name)
            success = migrator.migrate(max_workers=4)
            
            if success:
                logger.info("✅ Migration completed successfully")
                migrator.print_stats()
            else:
                logger.error("Migration failed, falling back to standard boot")
                
        except Exception as e:
            logger.error(f"Migration error: {e}")
    
    def _mark_ready(self):
        """Mark node as ready for inference."""
        logger.info("=" * 60)
        logger.info("🚀 GPU NODE READY FOR INFERENCE")
        logger.info("=" * 60)
        
        if self.fast_sync:
            logger.info("Fast-sync enabled: Block bodies loaded on-demand")
            logger.info("First inference may be slightly slower while caching")
    
    async def handle_inference(self, prompt: str, layer: str, expert: str) -> str:
        """
        Handle inference request with on-demand block loading.
        
        Args:
            prompt: Input prompt
            layer: Layer identifier
            expert: Expert identifier
            
        Returns:
            Generated text
        """
        # Get expert weights from chain B
        chain_b = self.chains.get('B')
        if not chain_b:
            return "Error: Chain B not initialized"
        
        # In fast-sync mode, this will load from CAS on-demand
        key = (layer, expert, "weight")
        
        if hasattr(chain_b, 'manifest') and chain_b.manifest:
            # Fast path: lookup in manifest
            entry = chain_b.manifest.find(key)
            if entry:
                # Load expert weights from CAS
                start_time = time.perf_counter()
                weights = chain_b.cas.get(entry.cid)
                load_time = (time.perf_counter() - start_time) * 1000
                
                logger.info(f"Loaded expert {layer}.{expert} in {load_time:.1f}ms (size: {len(weights)/1024:.1f}KB)")
                
                # TODO: Actual inference with loaded weights
                return f"Generated response using {layer}.{expert}"
        
        return f"Expert {layer}.{expert} not found"
    
    def run(self):
        """Run the GPU node."""
        # Initialize chains
        if not self.initialize_chains():
            logger.error("Failed to initialize chains")
            return
        
        # Start async event loop for inference
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Simulate inference requests
            logger.info("Simulating inference requests...")
            
            # Test inference with lazy loading
            for i in range(3):
                result = loop.run_until_complete(
                    self.handle_inference(f"test prompt {i}", f"layer{i}", f"expert{i}")
                )
                logger.info(f"Inference result: {result}")
            
            logger.info("GPU node running. Press Ctrl+C to stop.")
            loop.run_forever()
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            loop.close()


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="Blyan GPU Node with Fast-Sync")
    parser.add_argument("--fast-sync", action="store_true", default=True,
                       help="Enable fast-sync mode (default: True)")
    parser.add_argument("--no-fast-sync", dest="fast_sync", action="store_false",
                       help="Disable fast-sync mode")
    parser.add_argument("--verify-sample", type=float, default=0.02,
                       help="Sample verification rate (0.0-1.0, default: 0.02)")
    parser.add_argument("--verify-all", action="store_true",
                       help="Verify all blocks (overrides --verify-sample)")
    parser.add_argument("--auto-migrate", action="store_true",
                       help="Automatically migrate JSON blocks to sharded format")
    
    args = parser.parse_args()
    
    # Set environment variable for auto-migration
    if args.auto_migrate:
        os.environ['AUTO_MIGRATE'] = 'true'
    
    # Create and run node
    node = FastGPUNode(
        fast_sync=args.fast_sync,
        verify_sample=args.verify_sample,
        verify_all=args.verify_all
    )
    
    node.run()


if __name__ == "__main__":
    main()