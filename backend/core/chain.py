from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterator, Optional

from .block import Block, BlockHeader
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
        )

        # mine
        header.nonce = find_nonce(header.to_json().encode() + payload, self.difficulty)
        block = Block(
            header=header,
            payload=payload,
            miner_pub=miner_pub,
            payload_sig=payload_sig,
        )
        # persist
        self.storage.save_block(block)
        return block

    def verify_chain(self) -> bool:
        """Verify entire chain integrity & PoW results."""
        prev_hash = "0" * 64
        for block in self.storage.iter_blocks():
            # 1. link hashing
            if block.header.prev_hash != prev_hash:
                return False
            # 2. payload integrity
            if block.header.payload_hash != hashlib.sha256(block.payload).hexdigest():
                return False
            # 3. proof-of-work validity
            if not verify_pow(
                block.header.to_json().encode() + block.payload,
                block.header.nonce,
                self.difficulty,
            ):
                return False
            prev_hash = block.compute_hash()
        return True

    def __iter__(self) -> Iterator[Block]:
        return self.storage.iter_blocks() 