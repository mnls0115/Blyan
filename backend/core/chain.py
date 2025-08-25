from __future__ import annotations

import hashlib
import json
import os
import time
import logging
from pathlib import Path
from typing import Iterator, Optional, List, Literal, Dict, Set
from collections import defaultdict

from .block import Block, BlockHeader, validate_dag_structure, topological_sort
from .pow import find_pol_nonce, verify_pol_nonce
from .storage import BlockStorage
from ..utils.json_canonical import dumps_canonical, loads_canonical

logger = logging.getLogger(__name__)


class Chain:
    """Represents a single append-only blockchain (meta or parameter)."""

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
        
        self.storage = BlockStorage(root_dir / chain_id)
        self.storage.ensure_dir()
        
        # Performance optimization: In-memory hash index for O(1) lookups
        self._hash_index: Dict[str, int] = {}  # block_hash -> block_index
        self._dependency_index: Dict[str, Set[str]] = defaultdict(set)  # block_hash -> set of dependent hashes
        self._index_built = False  # Lazy loading flag
        
        # Use tip index for fast startup
        from .chain_tip import ChainTipIndex
        self.tip_index = ChainTipIndex(root_dir)
        
        # Trust tip by default, verify in background if requested
        self.verify_on_start = os.environ.get('VERIFY_ON_START', 'false').lower() in ('true', '1', 'yes')
        
        if self.verify_on_start:
            self._build_indexes()  # Full verification
        else:
            self._build_indexes_lazy()  # Fast startup

    # ------------------------------------------------------------------
    # Performance optimization methods
    # ------------------------------------------------------------------
    def _build_indexes_lazy(self):
        """Build minimal index metadata without loading all blocks - ULTRA FAST."""
        self._hash_index.clear()
        self._dependency_index.clear()
        
        # Use tip index if available (instant)
        tip = self.tip_index.get_tip(self.chain_id)
        if tip and not self.tip_index.is_stale(self.chain_id):
            # Trust the tip, just set the count
            height = tip.get("height", 0)
            for i in range(height + 1):
                placeholder_hash = f"block_{i}"
                self._hash_index[placeholder_hash] = i
            logger.debug(f"Chain {self.chain_id}: Using tip index (height={height})")
            self._index_built = False
            return
        
        # Fallback: Just count block files without loading them
        self.storage.ensure_dir()
        block_files = list(self.storage.dir_path.glob("*.json"))
        
        # Build a minimal index with just block counts
        for filepath in block_files:
            try:
                # Extract index from filename (e.g., "00000001.json" -> 1)
                filename = filepath.stem
                if filename.isdigit():
                    block_index = int(filename)
                    # Use placeholder hash for now - will be loaded on demand
                    placeholder_hash = f"block_{block_index}"
                    self._hash_index[placeholder_hash] = block_index
            except:
                continue
        
        self._index_built = False  # Mark as not fully built
    
    def _build_indexes(self):
        """Build full in-memory indexes for O(1) block lookups - SLOW."""
        if self._index_built:
            return  # Already built
            
        self._hash_index.clear()
        self._dependency_index.clear()
        
        for block in self.storage.iter_blocks():
            block_hash = block.compute_hash()
            self._hash_index[block_hash] = block.header.index
            
            # Build dependency index
            if block.header.depends_on:
                for dep_hash in block.header.depends_on:
                    self._dependency_index[dep_hash].add(block_hash)
        
        self._index_built = True  # Mark as fully built
    
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
        
        # 2. Verify dependencies exist (skip if chain is empty or dependencies list is empty)
        if new_block.header.depends_on:
            # If this is the first block and has dependencies, that's invalid
            # unless the dependency is itself (genesis block case)
            if len(self._hash_index) == 0 and new_block.header.block_type != "genesis_pact":
                # Empty chain but has dependencies - skip check for now
                # This happens during chain bootstrap
                pass
            else:
                for dep_hash in new_block.header.depends_on:
                    if dep_hash not in self._hash_index:
                        return False
        
        # 3. Verify no cycles would be created (using cached indexes)
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
        
        # DFS from new block to check for cycles
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
                # Look up block by hash
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

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def _latest(self) -> Optional[Block]:
        return self.storage.get_latest_block()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_block(
        self,
        payload: bytes,
        points_to: Optional[str] = None,
        miner_pub: Optional[str] = None,
        payload_sig: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        block_type: Literal['meta', 'expert', 'router', 'genesis_pact', 'dataset', 'layer', 'dense_layer'] = 'meta',
        expert_name: Optional[str] = None,
        layer_name: Optional[str] = None,  # For dense model layers
        layer_id: Optional[str] = None,
    ) -> Block:
        """Create, mine, and persist a new block."""
        # Use lock to prevent race conditions (if threading is used)
        import threading
        if not hasattr(self, '_add_lock'):
            self._add_lock = threading.Lock()
        
        with self._add_lock:
            prev_block = self._latest()
            index = (prev_block.header.index + 1) if prev_block else 0
            prev_hash = prev_block.compute_hash() if prev_block else "0" * 64
        
        payload_hash = hashlib.sha256(payload).hexdigest()
        
        # Ensure all non-genesis blocks depend on Genesis Pact (lightweight check)
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
            expert_name=expert_name,  # Keep for backward compatibility
            layer_name=layer_name,  # New dense model field
            layer_id=layer_id,
        )

        # Generate anti-spam proof (or skip if in development mode)  
        if self.skip_pol:
            # Skip anti-spam PoL but still include a dummy nonce
            header.nonce = 12345  # Dummy nonce for development
            print(f"⚡ Skipped anti-spam PoL for block {index} (dev mode)")
        else:
            # Lightweight anti-spam challenge instead of wasteful PoW
            contributor_id = miner_pub or "anonymous"
            header.nonce = find_pol_nonce(header.to_json().encode() + payload, contributor_id, self.difficulty)
        block = Block(
            header=header,
            payload=payload,
            miner_pub=miner_pub,
            payload_sig=payload_sig,
        )
        
        # PoL validation for expert/router blocks
        if self.enable_pol and self.chain_validator and block_type in ('expert', 'router'):
            print(f"🧠 Running PoL validation for {block_type} block...")
            
            # Get previous score for this expert if it exists
            previous_score = self._get_expert_previous_score(expert_name) if expert_name else None
            
            is_valid, validation_details = self.chain_validator.validate_block(block, previous_score)
            
            if not is_valid:
                failure_reason = validation_details.get('failure_reason', 'Unknown validation failure')
                raise ValueError(f"PoL validation failed: {failure_reason}")
            
            print(f"✅ PoL validation passed for {expert_name}")
            
            # Store validation metrics for future reference
            self._store_validation_metrics(block, validation_details)
        
        # Use incremental verification instead of full DAG validation (O(1) vs O(n²))
        if not self.verify_incremental(block):
            raise ValueError("Block validation failed - would create invalid chain state")
        
        # persist
        self.storage.save_block(block)
        
        # Update indexes for O(1) lookups
        self._update_indexes(block)
        
        # Update tip index for fast startup
        self.tip_index.update_tip(
            self.chain_id,
            block.compute_hash(),
            block.header.index,
            block.header.timestamp
        )
        
        return block

    def verify_chain(self) -> bool:
        """Verify entire chain integrity & anti-spam PoL results."""
        blocks = self.get_all_blocks()
        
        # 1. Verify DAG structure
        if not validate_dag_structure(blocks):
            return False
        
        # 2. Verify individual blocks
        for block in blocks:
            # Payload integrity
            if block.header.payload_hash != hashlib.sha256(block.payload).hexdigest():
                return False
            # Anti-spam PoL validity (skip if disabled in development)
            if not self.skip_pol:
                contributor_id = block.miner_pub or "anonymous"
                if not verify_pol_nonce(
                    block.header.to_json().encode() + block.payload,
                    block.header.nonce,
                    contributor_id
                ):
                    return False
            # Dependency validation (all dependencies must exist)
            block_hashes = {b.compute_hash() for b in blocks}
            for dep_hash in block.header.depends_on:
                if dep_hash not in block_hashes:
                    return False
        
        return True

    def _ensure_genesis_dependency(self, depends_on: List[str], block_type: str) -> List[str]:
        """Ensure all non-genesis blocks depend on Genesis Pact (lightweight operation)."""
        
        # Genesis blocks don't depend on themselves
        if block_type == "genesis_pact":
            return depends_on
        
        # Only the meta chain ('A') enforces a dependency on the Genesis Pact.
        # Other chains must not inject cross-chain dependencies via depends_on
        # (use points_to for cross-chain references instead).
        if getattr(self, 'chain_id', None) != 'A':
            return depends_on or []
        
        # If chain is empty, don't add genesis dependency (bootstrap case)
        if len(self._hash_index) == 0:
            return depends_on or []
        
        # Get Genesis Pact hash (cached after first lookup)
        if not hasattr(self, '_genesis_hash_cache'):
            self._genesis_hash_cache = self._get_genesis_pact_hash()
        
        # Add Genesis dependency if not already present
        if self._genesis_hash_cache and self._genesis_hash_cache not in depends_on:
            depends_on = depends_on + [self._genesis_hash_cache]
        
        return depends_on
    
    def _get_genesis_pact_hash(self) -> Optional[str]:
        """Get Genesis Pact hash (cached for performance)."""
        try:
            # Check if genesis hash file exists (fastest method)
            genesis_file = Path("./data/genesis_pact_hash.txt")
            if genesis_file.exists():
                return genesis_file.read_text().strip()
            
            # Fallback: search for genesis block (slower, but works)
            for block in self.storage.iter_blocks():
                if block.header.block_type == "genesis_pact":
                    hash_val = block.compute_hash()
                    # Cache it for next time
                    genesis_file.write_text(hash_val)
                    return hash_val
            
            return None
        except Exception:
            return None

    def __iter__(self) -> Iterator[Block]:
        return self.storage.iter_blocks()
    
    # ------------------------------------------------------------------
    # DAG-specific methods
    # ------------------------------------------------------------------
    def _validate_dag_before_add(self, new_block: Block) -> bool:
        """Validate that adding this block maintains DAG structure."""
        # Get all existing blocks
        existing_blocks = list(self.storage.iter_blocks())
        all_blocks = existing_blocks + [new_block]
        
        # Check if adding this block would create a cycle or invalid structure
        return validate_dag_structure(all_blocks)
    
    def get_all_blocks(self) -> List[Block]:
        """Get all blocks in the chain."""
        return list(self.storage.iter_blocks())
    
    def verify_dag(self) -> bool:
        """Verify that the entire chain forms a valid DAG."""
        blocks = self.get_all_blocks()
        return validate_dag_structure(blocks)
    
    def get_topological_order(self) -> Optional[List[str]]:
        """Get blocks in topological order, None if cycle exists."""
        blocks = self.get_all_blocks()
        return topological_sort(blocks)
    
    def get_blocks_by_type(self, block_type: Literal['meta', 'layer', 'dense_layer']) -> List[Block]:
        """Get all blocks of a specific type."""
        return [block for block in self.storage.iter_blocks() 
                if block.header.block_type == block_type]
    
    def get_layer_blocks(self, layer_name: str) -> List[Block]:
        """Get all blocks for a specific layer."""
        return [block for block in self.storage.iter_blocks() 
                if (block.header.block_type in ['layer', 'dense_layer'] and 
                    block.header.layer_name == layer_name)]
    
    def get_blocks_by_layer(self, layer_id: str) -> List[Block]:
        """Get all blocks for a specific layer."""
        return [block for block in self.storage.iter_blocks() 
                if block.header.layer_id == layer_id]
    
    def _get_layer_previous_score(self, layer_name: str) -> Optional[float]:
        """Get the previous best score for a layer."""
        if not layer_name:
            return None
        
        # Look for existing blocks with this layer_name
        existing_blocks = self.get_layer_blocks(layer_name)
        
        if not existing_blocks:
            return None
        
        # For now, return a mock previous score
        # In practice, this would be stored in block metadata or separate tracking
        return 0.75  # Mock baseline score
    
    def _store_validation_metrics(self, block: Block, validation_details: Dict):
        """Store validation metrics for future reference."""
        # In practice, you might want to store this in a separate metrics database
        # or as metadata in the block itself
        metrics_file = self.storage.storage_dir / f"validation_metrics_{block.compute_hash()[:16]}.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(validation_details, f, default=str, indent=2)
        except Exception as e:
            print(f"Warning: Could not store validation metrics: {e}") 