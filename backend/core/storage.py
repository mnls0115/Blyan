from __future__ import annotations

import json
import sys
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
        try:
            block_dict = block.to_dict()
            # Ensure file is properly closed even on error
            with path.open("w") as fp:
                json.dump(block_dict, fp)
                fp.flush()  # Force write to disk
                # OS will close file when exiting 'with' block
        except (TypeError, ValueError) as e:
            # If JSON serialization fails, try to identify the issue
            import sys
            print(f"ERROR: Failed to save block {block.header.index}: {e}", file=sys.stderr)
            print(f"Block type: {block.header.block_type}, Expert: {getattr(block.header, 'expert_name', 'N/A')}", file=sys.stderr)
            raise

    def load_block(self, index: int) -> Optional[Block]:
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
            return Block.from_dict(data)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse block {index} from {path}: {e}", file=sys.stderr)
            print(f"File size: {path.stat().st_size} bytes", file=sys.stderr)
            return None
        except Exception as e:
            print(f"ERROR: Failed to load block {index}: {e}", file=sys.stderr)
            return None

    def get_latest_block(self) -> Optional[Block]:
        self.ensure_dir()
        json_files = list(self.dir_path.glob("*.json"))
        if not json_files:
            return None
        # Filter to only numeric filenames
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
        self.ensure_dir()
        for path in sorted(self.dir_path.glob("*.json")):
            # Skip non-numeric files (like dataset_state.json)
            try:
                idx = int(path.stem)
                blk = self.load_block(idx)
                if blk is not None:
                    yield blk
            except ValueError:
                # Skip non-numeric filenames
                continue 
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """Alias for load_block to maintain compatibility."""
        return self.load_block(index)