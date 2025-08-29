from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional, List, Literal, Dict, Set
from collections import defaultdict

from .block import Block, BlockHeader, validate_dag_structure, topological_sort
from .pow import find_pol_nonce, verify_pol_nonce
from .storage_optimized import OptimizedBlockStorage
from ..utils.json_canonical import dumps_canonical, loads_canonical


class OptimizedChain:
    """Optimized blockchain with fast loading and caching."""

    def __init__(self, root_dir: Path, chain_id: str, difficulty: int = 1, skip_pol: bool = False, chain_validator=None):
        self.chain_id = chain_id
        
        # Support environment variable for difficulty override
        env_difficulty = os.environ.get('CHAIN_DIFFICULTY')
        if env_difficulty is not None:
            try:
                self.difficulty = int(env_difficulty)
                print(f"Using CHAIN_DIFFICULTY={self.difficulty} from environment")
            except ValueError:
                self.difficulty = difficulty
                print(f"Invalid CHAIN_DIFFICULTY, using default: {difficulty}")
        else:
            self.difficulty = difficulty
        
        self.skip_pol = skip_pol
        if skip_pol:
            print(f"⚠️  Anti-spam PoL disabled for chain {chain_id} (development mode)")
        
        # PoL validation support
        self.chain_validator = chain_validator
        self.enable_pol = os.environ.get('ENABLE_POL', 'false').lower() in ('true', '1', 'yes')
        if self.enable_pol and self.chain_validator:
            print(f"🧠 PoL validation enabled for chain {chain_id}")
        
        # Use optimized storage
        self.storage = OptimizedBlockStorage(root_dir / chain_id)
        self.storage.ensure_dir()
        
        # Performance optimization: In-memory hash index for O(1) lookups
        self._hash_index: Dict[str, int] = {}
        self._dependency_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Build indexes with caching
        self._build_indexes_optimized()

    def _build_indexes_optimized(self):
        """Build indexes using cached data when available."""
        print(f"Loading chain {self.chain_id} indexes...")
        start_time = time.time()
        
        # Try to use cached indexes
        indexes = self.storage.build_index_cache()
        
        self._hash_index = indexes['hash_index']
        self._dependency_index = defaultdict(set)
        for key, values in indexes['dependency_index'].items():
            self._dependency_index[key] = values
        
        elapsed = time.time() - start_time
        block_count = len(self._hash_index)
        
        if block_count > 0:
            print(f"✅ Chain {self.chain_id}: Loaded {block_count} blocks in {elapsed:.1f}s ({block_count/elapsed:.0f} blocks/sec)")
        else:
            print(f"Chain {self.chain_id}: Empty (0 blocks)")

    def _update_indexes(self, block: Block):
        """Update indexes with new block (incremental update)."""
        block_hash = block.compute_hash()
        self._hash_index[block_hash] = block.header.index
        
        if block.header.depends_on:
            for dep_hash in block.header.depends_on:
                self._dependency_index[dep_hash].add(block_hash)

    def verify_incremental(self, new_block: Block) -> bool:
        """
        Verify only the new block against existing chain state (O(1)).
        Much faster than verify_chain() which is O(n²).
        """
        # 1. Verify previous block hash matches
        if new_block.header.index > 0:
            prev_block = self.storage.get_block_by_index(new_block.header.index - 1)
            if not prev_block:
                return False
            if new_block.header.prev_hash != prev_block.compute_hash():
                return False
        
        # 2. Verify dependencies exist
        if new_block.header.depends_on:
            if len(self._hash_index) == 0 and new_block.header.block_type != "genesis_pact":
                pass  # Empty chain bootstrap case
            else:
                for dep_hash in new_block.header.depends_on:
                    if dep_hash not in self._hash_index:
                        return False
        
        # 3. Verify no cycles would be created
        if not self._check_no_cycles_incremental(new_block):
            return False
        
        # 4. Verify block signature/PoL
        if not self.skip_pol:
            contributor_id = new_block.miner_pub or "anonymous"
            if not verify_pol_nonce(
                new_block.header.to_json().encode() + new_block.payload,
                contributor_id,
                new_block.header.nonce,
                self.difficulty
            ):
                return False
        
        return True

    def _check_no_cycles_incremental(self, new_block: Block) -> bool:
        """Check if adding new_block would create a cycle (O(dependencies))."""
        if not new_block.header.depends_on:
            return True
        
        visited = set()
        stack = [new_block.compute_hash()]
        
        while stack:
            current = stack.pop()
            if current in visited:
                return False  # Cycle detected
            visited.add(current)
            
            # Get dependencies of current block
            if current == new_block.compute_hash():
                deps = new_block.header.depends_on or []
            else:
                block_idx = self._hash_index.get(current)
                if block_idx is not None:
                    block = self.storage.get_block_by_index(block_idx)
                    if block:
                        deps = block.header.depends_on or []
                    else:
                        deps = []
                else:
                    deps = []
            
            stack.extend(deps)
        
        return True

    def _latest(self) -> Optional[Block]:
        return self.storage.get_latest_block()

    def add_block(
        self,
        payload: bytes,
        points_to: Optional[str] = None,
        miner_pub: Optional[str] = None,
        payload_sig: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        block_type: Literal['meta', 'expert', 'router', 'genesis_pact', 'dataset', 'layer', 'dense_layer'] = 'meta',
        expert_name: Optional[str] = None,
        layer_name: Optional[str] = None,
        layer_id: Optional[str] = None,
    ) -> Block:
        """Create, mine, and persist a new block."""
        import threading
        if not hasattr(self, '_add_lock'):
            self._add_lock = threading.Lock()
        
        with self._add_lock:
            prev_block = self._latest()
            index = (prev_block.header.index + 1) if prev_block else 0
            prev_hash = prev_block.compute_hash() if prev_block else "0" * 64
        
        payload_hash = hashlib.sha256(payload).hexdigest()
        
        # Ensure all non-genesis blocks depend on Genesis Pact
        final_depends_on = self._ensure_genesis_dependency(depends_on or [], block_type)
        
        header = BlockHeader(
            index=index,
            timestamp=time.time(),
            prev_hash=prev_hash,
            chain_id=self.chain_id,
            points_to=points_to,
            payload_hash=payload_hash,
            payload_size=len(payload),
            nonce=0,
            depends_on=final_depends_on,
            block_type=block_type,
            expert_name=expert_name,
            layer_name=layer_name,
            layer_id=layer_id,
        )

        # Generate anti-spam proof
        if self.skip_pol:
            header.nonce = 12345
        else:
            contributor_id = miner_pub or "anonymous"
            header.nonce = find_pol_nonce(header.to_json().encode() + payload, contributor_id, self.difficulty)
        
        block = Block(
            header=header,
            payload=payload,
            miner_pub=miner_pub,
            payload_sig=payload_sig,
        )
        
        # Validate and save
        if not self.verify_incremental(block):
            raise ValueError("Block validation failed")
        
        self.storage.save_block(block)
        self._update_indexes(block)
        
        return block

    def verify_chain(self) -> bool:
        """Verify entire chain integrity."""
        blocks = self.get_all_blocks()
        
        if not validate_dag_structure(blocks):
            return False
        
        for block in blocks:
            if block.header.payload_hash != hashlib.sha256(block.payload).hexdigest():
                return False
            
            if not self.skip_pol:
                contributor_id = block.miner_pub or "anonymous"
                if not verify_pol_nonce(
                    block.header.to_json().encode() + block.payload,
                    block.header.nonce,
                    contributor_id
                ):
                    return False
            
            block_hashes = {b.compute_hash() for b in blocks}
            for dep_hash in block.header.depends_on:
                if dep_hash not in block_hashes:
                    return False
        
        return True

    def _ensure_genesis_dependency(self, depends_on: List[str], block_type: str) -> List[str]:
        """Ensure all non-genesis blocks depend on Genesis Pact."""
        if block_type == "genesis_pact":
            return depends_on
        
        if len(self._hash_index) == 0:
            return depends_on or []
        
        if not hasattr(self, '_genesis_hash_cache'):
            self._genesis_hash_cache = self._get_genesis_pact_hash()
        
        if self._genesis_hash_cache and self._genesis_hash_cache not in depends_on:
            depends_on = depends_on + [self._genesis_hash_cache]
        
        return depends_on

    def _get_genesis_pact_hash(self) -> Optional[str]:
        """Get Genesis Pact hash (cached for performance)."""
        try:
            genesis_file = Path("./data/genesis_pact_hash.txt")
            if genesis_file.exists():
                return genesis_file.read_text().strip()
            
            for block in self.storage.iter_blocks():
                if block.header.block_type == "genesis_pact":
                    hash_val = block.compute_hash()
                    genesis_file.write_text(hash_val)
                    return hash_val
            
            return None
        except Exception:
            return None

    def __iter__(self) -> Iterator[Block]:
        return self.storage.iter_blocks()

    def get_all_blocks(self) -> List[Block]:
        """Get all blocks in the chain."""
        return list(self.storage.iter_blocks())

    def verify_dag(self) -> bool:
        """Verify that the entire chain forms a valid DAG."""
        blocks = self.get_all_blocks()
        return validate_dag_structure(blocks)

    def get_topological_order(self) -> Optional[List[str]]:
        """Get blocks in topological order."""
        blocks = self.get_all_blocks()
        return topological_sort(blocks)

    def get_blocks_by_type(self, block_type: Literal['meta', 'expert', 'router']) -> List[Block]:
        """Get all blocks of a specific type."""
        return [block for block in self.storage.iter_blocks() 
                if block.header.block_type == block_type]

    def get_expert_blocks(self, expert_name: str) -> List[Block]:
        """Get all blocks for a specific expert."""
        return [block for block in self.storage.iter_blocks() 
                if (block.header.block_type == 'expert' and 
                    block.header.expert_name == expert_name)]

    def get_blocks_by_layer(self, layer_id: str) -> List[Block]:
        """Get all blocks for a specific layer."""
        return [block for block in self.storage.iter_blocks() 
                if block.header.layer_id == layer_id]

    def _get_expert_previous_score(self, expert_name: str) -> Optional[float]:
        """Get the previous best score for an expert."""
        if not expert_name:
            return None
        
        existing_blocks = self.get_expert_blocks(expert_name)
        
        if not existing_blocks:
            return None
        
        return 0.75

    def _store_validation_metrics(self, block: Block, validation_details: Dict):
        """Store validation metrics for future reference."""
        metrics_file = self.storage.storage_dir / f"validation_metrics_{block.compute_hash()[:16]}.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(validation_details, f, default=str, indent=2)
        except Exception as e:
            print(f"Warning: Could not store validation metrics: {e}")
