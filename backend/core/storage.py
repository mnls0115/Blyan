from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .block import Block


class BlockStorage:
    """File-system backed storage: one JSON file per block."""

    def __init__(self, dir_path: Path):
        self.dir_path = dir_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def ensure_dir(self) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def _block_path(self, index: int) -> Path:
        return self.dir_path / f"{index:08d}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save_block(self, block: Block) -> None:
        self.ensure_dir()
        path = self._block_path(block.header.index)
        with path.open("w") as fp:
            json.dump(block.to_dict(), fp)

    def load_block(self, index: int) -> Optional[Block]:
        path = self._block_path(index)
        if not path.exists():
            return None
        with path.open() as fp:
            data = json.load(fp)
        return Block.from_dict(data)

    def get_latest_block(self) -> Optional[Block]:
        self.ensure_dir()
        json_files = list(self.dir_path.glob("*.json"))
        if not json_files:
            return None
        latest_idx = max(int(p.stem) for p in json_files)
        return self.load_block(latest_idx)

    def iter_blocks(self) -> Iterator[Block]:
        self.ensure_dir()
        for path in sorted(self.dir_path.glob("*.json")):
            idx = int(path.stem)
            blk = self.load_block(idx)
            if blk is not None:
                yield blk 