"""
Chain Tip Index - Fast chain state tracking
============================================
Maintains a compact index of chain tips to avoid full scans on startup.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


class ChainTipIndex:
    """Maintains compact chain tip information for fast startup."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.tip_file = self.root_dir / "chain_tips.json"
        self.tips: Dict[str, Dict] = {}
        self._load()
    
    def _load(self) -> None:
        """Load tips from disk."""
        if self.tip_file.exists():
            try:
                with open(self.tip_file, 'r') as f:
                    self.tips = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load chain tips: {e}")
                self.tips = {}
    
    def _save(self) -> None:
        """Save tips to disk."""
        try:
            self.tip_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tip_file, 'w') as f:
                json.dump(self.tips, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chain tips: {e}")
    
    def update_tip(self, chain_id: str, block_hash: str, height: int, timestamp: Optional[float] = None) -> None:
        """Update chain tip information."""
        self.tips[chain_id] = {
            "hash": block_hash,
            "height": height,
            "timestamp": timestamp or time.time(),
            "updated": time.time()
        }
        self._save()
    
    def get_tip(self, chain_id: str) -> Optional[Dict]:
        """Get tip information for a chain."""
        return self.tips.get(chain_id)
    
    def get_height(self, chain_id: str) -> int:
        """Get cached height for a chain."""
        tip = self.tips.get(chain_id)
        return tip["height"] if tip else 0
    
    def is_stale(self, chain_id: str, max_age_seconds: int = 3600) -> bool:
        """Check if tip information is stale."""
        tip = self.tips.get(chain_id)
        if not tip:
            return True
        
        age = time.time() - tip.get("updated", 0)
        return age > max_age_seconds
    
    def clear(self) -> None:
        """Clear all tips."""
        self.tips = {}
        self._save()