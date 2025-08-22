from __future__ import annotations

import json
import sys
import time
import pickle
import concurrent.futures
from pathlib import Path
from typing import Iterator, Optional, Dict, List, Tuple
from threading import Lock

from .block import Block


class OptimizedBlockStorage:
    """Optimized file-system backed storage with batch loading and caching."""

    def __init__(self, dir_path: Path):
        self.dir_path = dir_path
        self._cache_lock = Lock()
        self._block_cache: Dict[int, Block] = {}
        self._index_cache_file = dir_path / ".block_index_cache.pkl"
        self._metadata_cache_file = dir_path / ".block_metadata.pkl"
        
    def ensure_dir(self) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def _block_path(self, index: int) -> Path:
        return self.dir_path / f"{index:08d}.json"

    def save_block(self, block: Block) -> None:
        """Save block and update cache."""
        self.ensure_dir()
        path = self._block_path(block.header.index)
        try:
            block_dict = block.to_dict()
            with path.open("w") as fp:
                json.dump(block_dict, fp)
                fp.flush()
            
            # Update cache
            with self._cache_lock:
                self._block_cache[block.header.index] = block
                
        except (TypeError, ValueError) as e:
            print(f"ERROR: Failed to save block {block.header.index}: {e}", file=sys.stderr)
            raise

    def load_block(self, index: int) -> Optional[Block]:
        """Load single block with caching."""
        # Check cache first
        with self._cache_lock:
            if index in self._block_cache:
                return self._block_cache[index]
        
        path = self._block_path(index)
        if not path.exists():
            return None
            
        try:
            with path.open() as fp:
                content = fp.read()
                if not content or content.strip() == "":
                    print(f"WARNING: Block file {path} is empty", file=sys.stderr)
                    return None
                data = json.loads(content)
            
            block = Block.from_dict(data)
            
            # Cache the block
            with self._cache_lock:
                self._block_cache[index] = block
                
            return block
            
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse block {index}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"ERROR: Failed to load block {index}: {e}", file=sys.stderr)
            return None

    def _load_block_batch(self, indices: List[int], max_workers: int = 8) -> Dict[int, Block]:
        """Load multiple blocks in parallel."""
        blocks = {}
        
        def load_single(idx):
            try:
                block = self.load_block(idx)
                if block:
                    return idx, block
            except Exception as e:
                print(f"Error loading block {idx}: {e}", file=sys.stderr)
            return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single, idx) for idx in indices]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    idx, block = result
                    blocks[idx] = block
                    
        return blocks

    def get_latest_block(self) -> Optional[Block]:
        """Get the latest block."""
        self.ensure_dir()
        json_files = list(self.dir_path.glob("*.json"))
        if not json_files:
            return None
            
        numeric_files = []
        for p in json_files:
            try:
                int(p.stem)
                numeric_files.append(p)
            except ValueError:
                continue
                
        if not numeric_files:
            return None
            
        latest_idx = max(int(p.stem) for p in numeric_files)
        return self.load_block(latest_idx)

    def iter_blocks(self) -> Iterator[Block]:
        """Iterate through all blocks."""
        self.ensure_dir()
        
        # Get all block indices
        indices = []
        for path in self.dir_path.glob("*.json"):
            try:
                idx = int(path.stem)
                indices.append(idx)
            except ValueError:
                continue
        
        indices.sort()
        
        # Load in batches for better performance
        batch_size = 100
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_blocks = self._load_block_batch(batch_indices)
            
            for idx in batch_indices:
                if idx in batch_blocks:
                    yield batch_blocks[idx]

    def iter_blocks_fast(self) -> Iterator[Block]:
        """Fast iteration using parallel loading."""
        self.ensure_dir()
        
        # Get all valid block files
        block_files = []
        for path in sorted(self.dir_path.glob("*.json")):
            try:
                idx = int(path.stem)
                block_files.append((idx, path))
            except ValueError:
                continue
        
        if not block_files:
            return
        
        print(f"Loading {len(block_files)} blocks using parallel processing...")
        start_time = time.time()
        
        # Load all blocks in parallel batches
        batch_size = 200
        max_workers = 16  # Increase parallel workers
        
        for i in range(0, len(block_files), batch_size):
            batch = block_files[i:i+batch_size]
            indices = [idx for idx, _ in batch]
            
            batch_blocks = self._load_block_batch(indices, max_workers=max_workers)
            
            for idx in indices:
                if idx in batch_blocks:
                    yield batch_blocks[idx]
            
            # Progress indicator
            if (i + batch_size) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + batch_size) / elapsed
                print(f"  Loaded {i + batch_size}/{len(block_files)} blocks ({rate:.0f} blocks/sec)")
        
        elapsed = time.time() - start_time
        print(f"Loaded all {len(block_files)} blocks in {elapsed:.1f} seconds")

    def build_index_cache(self) -> Dict[str, Dict]:
        """Build and cache block index for fast lookups."""
        if self._index_cache_file.exists():
            try:
                with open(self._index_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Check if cache is recent (less than 1 hour old)
                    if time.time() - cache_data['timestamp'] < 3600:
                        print("Using cached block index")
                        return cache_data['indexes']
            except Exception as e:
                print(f"Cache load failed: {e}, rebuilding...")
        
        print("Building block index cache...")
        start_time = time.time()
        
        hash_index = {}
        dependency_index = {}
        
        # Use fast parallel iteration
        for block in self.iter_blocks_fast():
            block_hash = block.compute_hash()
            hash_index[block_hash] = block.header.index
            
            if block.header.depends_on:
                for dep_hash in block.header.depends_on:
                    if dep_hash not in dependency_index:
                        dependency_index[dep_hash] = set()
                    dependency_index[dep_hash].add(block_hash)
        
        indexes = {
            'hash_index': hash_index,
            'dependency_index': dependency_index
        }
        
        # Save cache
        try:
            cache_data = {
                'timestamp': time.time(),
                'indexes': indexes
            }
            with open(self._index_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved index cache ({len(hash_index)} blocks)")
        except Exception as e:
            print(f"Failed to save cache: {e}")
        
        elapsed = time.time() - start_time
        print(f"Built indexes in {elapsed:.1f} seconds")
        
        return indexes

    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Alias for load_block to maintain compatibility."""
        return self.load_block(index)

    def clear_cache(self):
        """Clear all caches."""
        with self._cache_lock:
            self._block_cache.clear()
        
        if self._index_cache_file.exists():
            self._index_cache_file.unlink()
        if self._metadata_cache_file.exists():
            self._metadata_cache_file.unlink()