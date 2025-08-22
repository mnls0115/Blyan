"""
Sharded storage reader - reads data from shard files using memory-mapped I/O.
"""

import mmap
import struct
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple
import os


class ShardReader:
    """
    Reads data from sharded files using memory-mapped I/O for performance.
    Thread-safe with LRU cache for open mmaps.
    """
    
    MAGIC = b'SHRD'
    HEADER_SIZE = 16
    MAX_OPEN_MMAPS = 32  # Limit open file descriptors
    
    def __init__(self, shard_dir: Path):
        """
        Initialize shard reader.
        
        Args:
            shard_dir: Directory containing shard files
        """
        self.shard_dir = Path(shard_dir)
        self._lock = threading.Lock()
        self._mmaps: Dict[int, Tuple[mmap.mmap, any]] = {}  # shard_idx -> (mmap, file)
        self._access_order = []  # LRU tracking
    
    def _get_shard_path(self, idx: int) -> Path:
        """Get path for a shard file by index."""
        return self.shard_dir / f"shard_{idx:06d}.dat"
    
    def _open_mmap(self, shard_idx: int) -> Optional[mmap.mmap]:
        """Open a memory-mapped shard file."""
        path = self._get_shard_path(shard_idx)
        
        if not path.exists():
            return None
        
        try:
            # Open file in read-only mode
            f = open(path, 'rb')
            
            # Verify header
            header = f.read(self.HEADER_SIZE)
            if len(header) < self.HEADER_SIZE:
                f.close()
                return None
            
            magic, version = struct.unpack('>4sH10x', header)
            if magic != self.MAGIC:
                f.close()
                raise ValueError(f"Invalid shard file magic: {magic}")
            
            # Create memory map
            f.seek(0)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            return mm, f
            
        except Exception as e:
            if 'f' in locals():
                f.close()
            raise
    
    def _get_mmap(self, shard_idx: int) -> Optional[mmap.mmap]:
        """Get mmap for a shard, with LRU cache management."""
        with self._lock:
            # Check if already open
            if shard_idx in self._mmaps:
                # Update LRU order
                if shard_idx in self._access_order:
                    self._access_order.remove(shard_idx)
                self._access_order.append(shard_idx)
                return self._mmaps[shard_idx][0]
            
            # Check if we need to evict
            if len(self._mmaps) >= self.MAX_OPEN_MMAPS:
                # Evict least recently used
                if self._access_order:
                    evict_idx = self._access_order.pop(0)
                    mm, f = self._mmaps.pop(evict_idx)
                    mm.close()
                    f.close()
            
            # Open new mmap
            result = self._open_mmap(shard_idx)
            if result:
                mm, f = result
                self._mmaps[shard_idx] = (mm, f)
                self._access_order.append(shard_idx)
                return mm
            
            return None
    
    def read(self, shard_idx: int, offset: int, length: int) -> bytes:
        """
        Read data from a shard.
        
        Args:
            shard_idx: Shard index
            offset: Byte offset within shard
            length: Number of bytes to read (excluding length prefix)
            
        Returns:
            Data bytes
        """
        mm = self._get_mmap(shard_idx)
        if not mm:
            raise FileNotFoundError(f"Shard {shard_idx} not found")
        
        # Read from mmap
        start = offset + self.HEADER_SIZE
        
        # Read length prefix
        if start + 4 > len(mm):
            raise ValueError(f"Invalid offset {offset} in shard {shard_idx}")
        
        stored_len = struct.unpack('>I', mm[start:start+4])[0]
        
        if stored_len != length:
            raise ValueError(f"Length mismatch: expected {length}, got {stored_len}")
        
        # Read data
        data_start = start + 4
        data_end = data_start + length
        
        if data_end > len(mm):
            raise ValueError(f"Data extends beyond shard boundary")
        
        return bytes(mm[data_start:data_end])
    
    def read_range(self, shard_idx: int, offset: int, length: int, 
                   range_start: int, range_len: int) -> bytes:
        """
        Read a range within data from a shard.
        
        Args:
            shard_idx: Shard index
            offset: Byte offset within shard
            length: Total length of data
            range_start: Start offset within data
            range_len: Number of bytes to read from range_start
            
        Returns:
            Partial data bytes
        """
        if range_start >= length or range_start + range_len > length:
            raise ValueError("Range exceeds data boundaries")
        
        mm = self._get_mmap(shard_idx)
        if not mm:
            raise FileNotFoundError(f"Shard {shard_idx} not found")
        
        # Calculate actual read position
        start = offset + self.HEADER_SIZE + 4 + range_start  # +4 for length prefix
        end = start + range_len
        
        if end > len(mm):
            raise ValueError(f"Range extends beyond shard boundary")
        
        return bytes(mm[start:end])
    
    def verify_shard(self, shard_idx: int) -> bool:
        """Verify shard file integrity."""
        path = self._get_shard_path(shard_idx)
        
        if not path.exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                header = f.read(self.HEADER_SIZE)
                if len(header) < self.HEADER_SIZE:
                    return False
                
                magic, version = struct.unpack('>4sH10x', header)
                return magic == self.MAGIC
        except:
            return False
    
    def close(self):
        """Close all open mmaps."""
        with self._lock:
            for mm, f in self._mmaps.values():
                mm.close()
                f.close()
            self._mmaps.clear()
            self._access_order.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_stats(self) -> dict:
        """Get reader statistics."""
        with self._lock:
            shard_files = list(self.shard_dir.glob("shard_*.dat"))
            total_size = sum(f.stat().st_size for f in shard_files)
            
            return {
                'shard_count': len(shard_files),
                'open_mmaps': len(self._mmaps),
                'total_size_mb': total_size / (1024 * 1024),
                'cache_order': list(self._access_order)
            }