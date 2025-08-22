#!/usr/bin/env python3
"""
Save and load upload progress for resume capability.
"""

import json
from pathlib import Path
from typing import Dict, Any

PROGRESS_FILE = Path("./data/upload_progress.json")

def save_progress(layer_idx: int, expert_idx: int, total_uploaded: int, metadata: Dict[str, Any] = None):
    """Save current upload progress."""
    progress = {
        "last_layer": layer_idx,
        "last_expert": expert_idx,
        "total_uploaded": total_uploaded,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_FILE.open("w") as f:
        json.dump(progress, f, indent=2)

def load_progress() -> Dict[str, Any]:
    """Load previous upload progress."""
    if not PROGRESS_FILE.exists():
        return None
    
    try:
        with PROGRESS_FILE.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load progress: {e}")
        return None

def clear_progress():
    """Clear saved progress after successful completion."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

def should_skip(layer_idx: int, expert_idx: int, progress: Dict[str, Any]) -> bool:
    """Check if this expert should be skipped (already uploaded)."""
    if not progress:
        return False
    
    last_layer = progress.get("last_layer", -1)
    last_expert = progress.get("last_expert", -1)
    
    # Skip if we're before the last checkpoint
    if layer_idx < last_layer:
        return True
    elif layer_idx == last_layer and expert_idx <= last_expert:
        return True
    
    return False