#!/usr/bin/env python3
"""
Migration tool to convert existing per-block JSON files to sharded storage with manifest.
"""

import json
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.cas import ContentAddressableStore
from backend.storage.manifest import Manifest, ManifestEntry
from backend.storage.index_cache import IndexCache
from backend.core.block import Block

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BlockMigrator:
    """Migrate JSON blocks to sharded storage."""
    
    def __init__(self, src_dir: Path, dst_dir: Path, chain_id: str = "B"):
        """
        Initialize migrator.
        
        Args:
            src_dir: Source directory with JSON blocks
            dst_dir: Destination directory for sharded storage
            chain_id: Chain identifier
        """
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.chain_id = chain_id
        
        # Initialize storage components
        self.cas = ContentAddressableStore(self.dst_dir / "cas")
        self.manifest = Manifest(chain_id)
        
        self.stats = {
            'total_blocks': 0,
            'migrated_blocks': 0,
            'failed_blocks': 0,
            'total_bytes': 0,
            'dedup_savings': 0
        }
    
    def _process_block_file(self, json_path: Path) -> Optional[Dict]:
        """Process a single block JSON file."""
        try:
            # Extract block index from filename
            block_idx = int(json_path.stem)
            
            # Load block JSON
            with open(json_path, 'r') as f:
                block_data = json.load(f)
            
            # Create Block object
            block = Block.from_dict(block_data)
            
            # Prepare payload (the actual expert weights or data)
            payload = block.payload
            
            # Compute merkle root (simplified - just hash for now)
            merkle_root = hashlib.sha256(payload).hexdigest()
            
            # Determine key based on block type
            if block.header.block_type == 'expert':
                layer = block.header.layer_id or f"layer{block_idx // 100}"
                expert = block.header.expert_name or f"expert{block_idx % 100}"
                part = "weight"
            else:
                layer = "meta"
                expert = block.header.block_type or "unknown"
                part = str(block_idx)
            
            key = (layer, expert, part)
            
            return {
                'block': block,
                'key': key,
                'payload': payload,
                'merkle_root': merkle_root,
                'block_idx': block_idx,
                'json_path': json_path
            }
            
        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")
            return None
    
    def migrate(self, max_workers: int = 4) -> bool:
        """
        Perform migration from JSON to shards.
        
        Args:
            max_workers: Number of parallel workers
            
        Returns:
            True if successful
        """
        logger.info(f"Starting migration from {self.src_dir} to {self.dst_dir}")
        
        # Find all JSON block files
        json_files = sorted(self.src_dir.glob("*.json"))
        
        # Filter to only numeric block files
        block_files = []
        for f in json_files:
            try:
                int(f.stem)
                block_files.append(f)
            except ValueError:
                continue
        
        if not block_files:
            logger.warning("No block files found to migrate")
            return False
        
        logger.info(f"Found {len(block_files)} blocks to migrate")
        self.stats['total_blocks'] = len(block_files)
        
        # Process blocks in parallel for parsing
        logger.info("Phase 1: Parsing blocks...")
        blocks_to_migrate = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_block_file, f): f 
                      for f in block_files}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    blocks_to_migrate.append(result)
                else:
                    self.stats['failed_blocks'] += 1
        
        # Sort by block index for sequential writing
        blocks_to_migrate.sort(key=lambda x: x['block_idx'])
        
        # Phase 2: Write to CAS and build manifest
        logger.info("Phase 2: Writing to sharded storage...")
        
        for block_info in blocks_to_migrate:
            try:
                # Store payload in CAS
                cid, shard, offset, length = self.cas.put(block_info['payload'])
                
                # Track stats
                self.stats['total_bytes'] += length
                
                # Create manifest entry
                entry = ManifestEntry(
                    key=block_info['key'],
                    cid=cid,
                    size=length,
                    shard=shard,
                    offset=offset,
                    length=length,
                    merkle_root=block_info['merkle_root'],
                    parent_hash=block_info['block'].header.prev_hash,
                    timestamp=block_info['block'].header.timestamp,
                    block_index=block_info['block_idx'],
                    block_type=block_info['block'].header.block_type
                )
                
                self.manifest.add_entry(entry)
                self.stats['migrated_blocks'] += 1
                
                # Progress indicator
                if self.stats['migrated_blocks'] % 100 == 0:
                    pct = (self.stats['migrated_blocks'] / len(blocks_to_migrate)) * 100
                    logger.info(f"  Progress: {self.stats['migrated_blocks']}/{len(blocks_to_migrate)} ({pct:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to migrate block {block_info['block_idx']}: {e}")
                self.stats['failed_blocks'] += 1
        
        # Phase 3: Save manifest and index cache
        logger.info("Phase 3: Saving manifest and building index cache...")
        
        # Save manifest
        manifest_path = self.dst_dir / f"manifest_{self.chain_id}.msgpack"
        if not manifest_path.parent.exists():
            manifest_path = self.dst_dir / f"manifest_{self.chain_id}.json"
        
        self.manifest.save(manifest_path)
        logger.info(f"Saved manifest to {manifest_path}")
        
        # Build and save index cache
        cache_dir = self.dst_dir / "cache"
        cache = IndexCache(cache_dir, self.manifest.head)
        cache.build_from_manifest(self.manifest)
        logger.info(f"Built index cache with {len(self.manifest.objects)} entries")
        
        # Calculate deduplication savings
        cas_stats = self.cas.get_stats()
        if cas_stats['total_refs'] > cas_stats['total_objects']:
            self.stats['dedup_savings'] = (
                (cas_stats['total_refs'] - cas_stats['total_objects']) * 
                cas_stats['avg_size']
            )
        
        # Close CAS
        self.cas.close()
        
        return self.stats['failed_blocks'] == 0
    
    def verify(self) -> bool:
        """Verify migration by spot-checking some blocks."""
        logger.info("Verifying migration...")
        
        # Load manifest
        manifest_path = self.dst_dir / f"manifest_{self.chain_id}.msgpack"
        if not manifest_path.exists():
            manifest_path = self.dst_dir / f"manifest_{self.chain_id}.json"
        
        if not manifest_path.exists():
            logger.error("Manifest not found")
            return False
        
        manifest = Manifest.load(manifest_path)
        
        # Verify manifest
        is_valid, errors = manifest.validate()
        if not is_valid:
            logger.error(f"Manifest validation failed: {errors}")
            return False
        
        # Spot check some entries
        cas = ContentAddressableStore(self.dst_dir / "cas")
        
        sample_size = min(10, len(manifest.objects))
        sample_indices = [i * (len(manifest.objects) // sample_size) 
                         for i in range(sample_size)]
        
        for idx in sample_indices:
            entry = manifest.objects[idx]
            
            try:
                # Read from CAS
                data = cas.get(entry.cid)
                
                # Verify size
                if len(data) != entry.size:
                    logger.error(f"Size mismatch for {entry.key}: {len(data)} != {entry.size}")
                    return False
                
                # Verify CID
                if cas._compute_cid(data) != entry.cid:
                    logger.error(f"CID mismatch for {entry.key}")
                    return False
                
            except Exception as e:
                logger.error(f"Failed to verify {entry.key}: {e}")
                return False
        
        cas.close()
        
        logger.info("✅ Migration verified successfully")
        return True
    
    def print_stats(self):
        """Print migration statistics."""
        print("\n" + "=" * 60)
        print("MIGRATION STATISTICS")
        print("=" * 60)
        print(f"Total blocks:      {self.stats['total_blocks']}")
        print(f"Migrated:          {self.stats['migrated_blocks']}")
        print(f"Failed:            {self.stats['failed_blocks']}")
        print(f"Total size:        {self.stats['total_bytes'] / (1024**2):.1f} MB")
        
        if self.stats['dedup_savings'] > 0:
            print(f"Dedup savings:     {self.stats['dedup_savings'] / (1024**2):.1f} MB")
        
        # CAS stats
        cas_stats = self.cas.get_stats()
        print(f"\nCAS Statistics:")
        print(f"  Unique objects:  {cas_stats['total_objects']}")
        print(f"  Total refs:      {cas_stats['total_refs']}")
        print(f"  Shard count:     {cas_stats['shard_count']}")
        print(f"  Avg object size: {cas_stats['avg_size'] / 1024:.1f} KB")


def main():
    """Main migration entry point."""
    parser = argparse.ArgumentParser(description="Migrate blockchain from JSON to sharded storage")
    parser.add_argument("--src", required=True, help="Source directory with JSON blocks")
    parser.add_argument("--dst", required=True, help="Destination directory for sharded storage")
    parser.add_argument("--chain", default="B", help="Chain ID (default: B)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--verify", action="store_true", help="Verify after migration")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    
    if not src_dir.exists():
        logger.error(f"Source directory not found: {src_dir}")
        sys.exit(1)
    
    # Create destination
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Run migration
    migrator = BlockMigrator(src_dir, dst_dir, args.chain)
    
    start_time = time.time()
    success = migrator.migrate(max_workers=args.workers)
    elapsed = time.time() - start_time
    
    # Print statistics
    migrator.print_stats()
    print(f"\nMigration time: {elapsed:.1f} seconds")
    
    # Verify if requested
    if args.verify and success:
        if not migrator.verify():
            logger.error("Verification failed!")
            sys.exit(1)
    
    if not success:
        logger.error("Migration failed!")
        sys.exit(1)
    
    print("\n✅ Migration completed successfully!")


if __name__ == "__main__":
    main()