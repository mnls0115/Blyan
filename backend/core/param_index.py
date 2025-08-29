from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional


class ParameterIndex:
    """Persistent mapping from parameter name -> block index.

    Stored as a JSON file on disk for simplicity (can be swapped for LevelDB later).
    """

    def __init__(self, path: Path):
        self.path = path
        self.checksum_path = path.with_suffix('.sha256')
        self._index: Dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                # Check integrity first
                if not self._verify_checksum():
                    import logging
                    logging.warning(f"Param index checksum mismatch - rebuilding from blockchain")
                    self._index = {}
                    # Could trigger rebuild here if we have access to chain
                    return
                
                with self.path.open() as fp:
                    self._index = {k: int(v) for k, v in json.load(fp).items()}
            except Exception:
                self._index = {}
        else:
            self._index = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write index
        with self.path.open("w") as fp:
            json.dump(self._index, fp)
        
        # Write checksum
        self._write_checksum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set(self, name: str, block_index: int) -> None:
        self._index[name] = block_index
        self._save()

    def bulk_set(self, mapping: Dict[str, int]) -> None:
        self._index.update(mapping)
        self._save()

    def get(self, name: str) -> Optional[int]:
        return self._index.get(name)

    def all(self) -> Dict[str, int]:
        return dict(self._index)
    
    def get_all_layers(self) -> list[str]:
        """Get all layer names from the index."""
        return list(self._index.keys())
    
    # ------------------------------------------------------------------
    # Integrity methods
    # ------------------------------------------------------------------
    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of the index file."""
        if not self.path.exists():
            return ""
        
        sha256 = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _write_checksum(self) -> None:
        """Write checksum file for the index."""
        try:
            checksum = self._compute_checksum()
            with open(self.checksum_path, 'w') as f:
                f.write(checksum)
        except Exception as e:
            import logging
            logging.warning(f"Failed to write param index checksum: {e}")
    
    def _verify_checksum(self) -> bool:
        """Verify the index file against its checksum."""
        if not self.checksum_path.exists():
            # No checksum file yet - this is OK for migration
            return True
        
        try:
            with open(self.checksum_path, 'r') as f:
                expected = f.read().strip()
            
            actual = self._compute_checksum()
            return actual == expected
        except Exception:
            # If we can't verify, assume it's OK to avoid breaking existing systems
            return True
    
    def rebuild_if_corrupted(self, chain_b) -> bool:
        """Rebuild index from blockchain if corrupted."""
        if self._verify_checksum():
            return False  # No rebuild needed
        
        import logging
        logging.warning("Param index corrupted - rebuilding from blockchain")
        
        try:
            # Clear current index
            self._index = {}
            
            # Rebuild from chain B
            if hasattr(chain_b, 'get_blocks_by_type'):
                layer_blocks = chain_b.get_blocks_by_type('dense_layer')
                for block in layer_blocks:
                    if hasattr(block.header, 'layer_name'):
                        self._index[block.header.layer_name] = block.header.index
                
                self._save()
                logging.info(f"Rebuilt param index with {len(self._index)} entries")
                return True
        except Exception as e:
            logging.error(f"Failed to rebuild param index: {e}")
        
        return False