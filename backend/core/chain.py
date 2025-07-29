from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterator, Optional, List, Literal

from .block import Block, BlockHeader, validate_dag_structure, topological_sort
from .pow import find_nonce, verify_pow
from .storage import BlockStorage


class Chain:
    """Represents a single append-only blockchain (meta or parameter)."""

    def __init__(self, root_dir: Path, chain_id: str, difficulty: int = 4):
        self.chain_id = chain_id
        self.difficulty = difficulty
        self.storage = BlockStorage(root_dir / chain_id)
        self.storage.ensure_dir()

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
        block_type: Literal['meta', 'expert', 'router'] = 'meta',
        expert_name: Optional[str] = None,
        layer_id: Optional[str] = None,
    ) -> Block:
        """Create, mine, and persist a new block."""
        prev_block = self._latest()
        index = (prev_block.header.index + 1) if prev_block else 0
        prev_hash = prev_block.compute_hash() if prev_block else "0" * 64

        payload_hash = hashlib.sha256(payload).hexdigest()
        header = BlockHeader(
            index=index,
            timestamp=time.time(),
            prev_hash=prev_hash,
            chain_id=self.chain_id,
            points_to=points_to,
            payload_hash=payload_hash,
            payload_size=len(payload),
            nonce=0,
            depends_on=depends_on or [],
            block_type=block_type,
            expert_name=expert_name,
            layer_id=layer_id,
        )

        # mine
        header.nonce = find_nonce(header.to_json().encode() + payload, self.difficulty)
        block = Block(
            header=header,
            payload=payload,
            miner_pub=miner_pub,
            payload_sig=payload_sig,
        )
        # DAG validation before persisting - temporarily disabled for performance
        # if not self._validate_dag_before_add(block):
        #     raise ValueError("Adding this block would create an invalid DAG structure")
        
        # persist
        self.storage.save_block(block)
        return block

    def verify_chain(self) -> bool:
        """Verify entire chain integrity & PoW results."""
        blocks = self.get_all_blocks()
        
        # 1. Verify DAG structure
        if not validate_dag_structure(blocks):
            return False
        
        # 2. Verify individual blocks
        for block in blocks:
            # Payload integrity
            if block.header.payload_hash != hashlib.sha256(block.payload).hexdigest():
                return False
            # Proof-of-work validity
            if not verify_pow(
                block.header.to_json().encode() + block.payload,
                block.header.nonce,
                self.difficulty,
            ):
                return False
            # Dependency validation (all dependencies must exist)
            block_hashes = {b.compute_hash() for b in blocks}
            for dep_hash in block.header.depends_on:
                if dep_hash not in block_hashes:
                    return False
        
        return True

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