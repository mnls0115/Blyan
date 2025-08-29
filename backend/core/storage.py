from __future__ import annotations

import json
import sys
import hashlib
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
            
            # Append to header index for fast verification
            self._append_to_header_index(block)
            
        except (TypeError, ValueError) as e:
            # If JSON serialization fails, try to identify the issue
            import sys
            print(f"ERROR: Failed to save block {block.header.index}: {e}", file=sys.stderr)
            print(f"Block type: {block.header.block_type}, Expert: {getattr(block.header, 'expert_name', 'N/A')}", file=sys.stderr)
            raise
    
    def _append_to_header_index(self, block: Block) -> None:
        """Append header record to index after saving block."""
        try:
            from .header_index import HeaderIndex, HeaderRecord
            
            # Get header index for this chain
            header_index = HeaderIndex(self.dir_path)
            
            # Compute cumulative digest
            prev_record = header_index.get_last_record()
            prev_cum_digest = bytes.fromhex(prev_record.cum_digest) if prev_record else bytes(32)
            
            # Create header record
            block_hash = block.compute_hash()
            payload_hash = hashlib.sha256(block.payload).hexdigest()
            
            cum_digest = header_index.compute_cum_digest_for_record(
                index=block.header.index,
                block_hash=block_hash,
                prev_hash=block.header.prev_hash,
                payload_hash=payload_hash,
                prev_cum_digest=prev_cum_digest
            )
            
            record = HeaderRecord(
                index=block.header.index,
                hash=block_hash,
                prev_hash=block.header.prev_hash,
                payload_hash=payload_hash,
                cum_digest=cum_digest
            )
            
            # Append to index
            header_index.append(record)
            
            # Update finality anchor if needed (every FINALITY_DEPTH blocks)
            self._update_finality_anchor_if_needed(block.header.index, cum_digest)
            
        except Exception as e:
            # Log but don't fail block save if index update fails
            print(f"WARNING: Failed to update header index: {e}", file=sys.stderr)
    
    def _update_finality_anchor_if_needed(self, height: int, cum_digest: str) -> None:
        """Update finality anchor every FINALITY_DEPTH blocks."""
        import os
        from datetime import datetime
        
        FINALITY_DEPTH = int(os.getenv('FINALITY_DEPTH', '512'))
        
        # Check if we should update anchor
        # Maintain per-chain anchor file to avoid conflicts between A/B
        anchor_file = self.dir_path.parent / f"finality_anchor_{self.dir_path.name}.json"
        
        # Load existing anchor if any
        last_anchor_height = -1
        if anchor_file.exists():
            try:
                with open(anchor_file, 'r') as f:
                    anchor_data = json.load(f)
                    last_anchor_height = anchor_data.get('height', -1)
            except:
                pass
        
        # Update if we've advanced FINALITY_DEPTH blocks
        if height >= last_anchor_height + FINALITY_DEPTH:
            try:
                anchor_data = {
                    'height': height,
                    'cum_digest': cum_digest,
                    'updated_at': datetime.utcnow().isoformat() + 'Z'
                }
                
                # Write atomically
                anchor_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file = anchor_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(anchor_data, f, indent=2)
                temp_file.replace(anchor_file)
                
            except Exception as e:
                print(f"WARNING: Failed to update finality anchor: {e}", file=sys.stderr)

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
