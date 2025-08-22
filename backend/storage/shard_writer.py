"""
Sharded storage writer - writes data to large shard files instead of individual files.
"""

import os
import struct
import threading
from pathlib import Path
from typing import Tuple, Optional
import fcntl


class ShardWriter:
    """
    Writes data to sharded files with configurable size limits.
    Thread-safe with file locking.
    """
    
    # Header format: magic (4 bytes) + version (2 bytes) + reserved (10 bytes)
    MAGIC = b'SHRD'
    VERSION = 1
    HEADER_SIZE = 16
    
    def __init__(self, shard_dir: Path, shard_size_mb: int = 512):
        """
        Initialize shard writer.
        
        Args:
            shard_dir: Directory to store shard files
            shard_size_mb: Maximum size of each shard file in MB
        """
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        
        self._lock = threading.Lock()
        self._current_shard = None
        self._current_file = None
        self._current_offset = 0
        self._current_shard_idx = self._find_latest_shard()
    
    def _find_latest_shard(self) -> int:
        """Find the index of the latest shard file."""
        shard_files = list(self.shard_dir.glob("shard_*.dat"))
        if not shard_files:
            return 0
        
        # Extract shard indices
        indices = []
        for f in shard_files:
            try:
                idx = int(f.stem.split('_')[1])
                indices.append(idx)
            except (ValueError, IndexError):
                continue
        
        return max(indices) if indices else 0
    
    def _get_shard_path(self, idx: int) -> Path:
        """Get path for a shard file by index."""
        return self.shard_dir / f"shard_{idx:06d}.dat"
    
    def _open_shard(self, idx: int, mode: str = 'ab') -> Tuple[any, int]:
        """Open a shard file and return file handle and current size."""
        path = self._get_shard_path(idx)
        
        # Create new shard with header if it doesn't exist
        if not path.exists():
            with open(path, 'wb') as f:
                # Write header
                header = struct.pack(
                    '>4sH10s',
                    self.MAGIC,
                    self.VERSION,
                    b'\x00' * 10  # Reserved space
                )
                f.write(header)
                
                # Pre-allocate space on Linux for better performance
                if hasattr(os, 'posix_fallocate'):
                    try:
                        initial_size = min(self.shard_size_bytes // 4, 128 * 1024 * 1024)
                        os.posix_fallocate(f.fileno(), 0, initial_size)
                    except:
                        pass  # Not critical if it fails
        
        # Open for appending
        f = open(path, mode)
        
        # Get current size
        f.seek(0, 2)  # Seek to end
        size = f.tell()
        
        return f, size
    
    def append(self, data: bytes) -> Tuple[int, int, int]:
        """
        Append data to shard storage.
        
        Args:
            data: Bytes to write
            
        Returns:
            Tuple of (shard_index, offset, length)
        """
        if not data:
            raise ValueError("Cannot write empty data")
        
        data_len = len(data)
        
        with self._lock:
            # Check if we need a new shard
            if (self._current_file is None or 
                self._current_offset + data_len > self.shard_size_bytes):
                
                # Close current shard if open
                if self._current_file:
                    self._current_file.flush()
                    os.fsync(self._current_file.fileno())
                    self._current_file.close()
                
                # Open new shard
                if self._current_file is None or self._current_offset + data_len > self.shard_size_bytes:
                    self._current_shard_idx += 1
                
                self._current_file, self._current_offset = self._open_shard(self._current_shard_idx)
                self._current_shard = self._current_shard_idx
            
            # Write length prefix (4 bytes) + data
            shard = self._current_shard
            offset = self._current_offset
            
            # Write length-prefixed data
            self._current_file.write(struct.pack('>I', data_len))
            self._current_file.write(data)
            
            # Update offset
            self._current_offset += 4 + data_len
            
            # Periodic flush for durability
            if self._current_offset % (1024 * 1024) < (4 + data_len):
                self._current_file.flush()
            
            return (shard, offset, data_len)
    
    def flush(self):
        """Flush current shard to disk."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
    
    def close(self):
        """Close the writer and flush pending data."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
                self._current_file.close()
                self._current_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_stats(self) -> dict:
        """Get writer statistics."""
        with self._lock:
            return {
                'current_shard': self._current_shard,
                'current_offset': self._current_offset,
                'shard_count': self._current_shard_idx + 1,
                'shard_size_mb': self.shard_size_bytes // (1024 * 1024)
            }