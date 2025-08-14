#!/usr/bin/env python3
"""VRAM/memory monitoring for pipeline nodes"""
from __future__ import annotations

import time
import psutil

def monitor_vram():
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_stats(i)
                allocated = mem.get('allocated_bytes.all.current', 0) / 1e9
                reserved = mem.get('reserved_bytes.all.current', 0) / 1e9
                print(f"GPU{i}: {allocated:.1f}GB alloc, {reserved:.1f}GB reserved")
    except ImportError:
        pass
    
    # System memory
    mem = psutil.virtual_memory()
    print(f"RAM: {mem.used/1e9:.1f}GB used / {mem.total/1e9:.1f}GB total ({mem.percent:.1f}%)")

if __name__ == "__main__":
    while True:
        monitor_vram()
        time.sleep(5)