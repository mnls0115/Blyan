from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class ParameterIndex:
    """Persistent mapping from parameter name -> block index.

    Stored as a JSON file on disk for simplicity (can be swapped for LevelDB later).
    """

    def __init__(self, path: Path):
        self.path = path
        self._index: Dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open() as fp:
                    self._index = {k: int(v) for k, v in json.load(fp).items()}
            except Exception:
                self._index = {}
        else:
            self._index = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fp:
            json.dump(self._index, fp)

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