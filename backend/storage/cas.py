"""
Content-Addressable Store (CAS) with deduplication and BLAKE3 hashing.
"""

import sqlite3
import threading
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False
    print("Warning: blake3 not available, falling back to SHA256")

from .shard_writer import ShardWriter
from .shard_reader import ShardReader


class ContentAddressableStore:
    """
    Content-addressable storage with automatic deduplication.
    Uses BLAKE3 for fast cryptographic hashing.
    """
    
    def __init__(self, storage_dir: Path, shard_size_mb: int = 512):
        """
        Initialize CAS.
        
        Args:
            storage_dir: Directory for storage
            shard_size_mb: Size of each shard file
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.shard_dir = self.storage_dir / "shards"
        self.db_path = self.storage_dir / "cas_index.db"
        
        self.writer = ShardWriter(self.shard_dir, shard_size_mb)
        self.reader = ShardReader(self.shard_dir)
        
        self._lock = threading.Lock()
        self._init_db()
        
        # Cache for recent CIDs
        self._cid_cache: Dict[str, Tuple[int, int, int]] = {}
        self._cache_size = 10000
    
    def _init_db(self):
        """Initialize SQLite index database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Create CID index table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cid_index (
                    cid TEXT PRIMARY KEY,
                    shard INTEGER NOT NULL,
                    offset INTEGER NOT NULL,
                    length INTEGER NOT NULL,
                    refcount INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shard ON cid_index(shard)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cid_index(created_at)")
            
            conn.commit()
    
    def _compute_cid(self, data: bytes) -> str:
        """Compute content identifier using BLAKE3 or SHA256."""
        if HAS_BLAKE3:
            return blake3.blake3(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()
    
    def put(self, data: bytes) -> Tuple[str, int, int, int]:
        """
        Store data in CAS.
        
        Args:
            data: Bytes to store
            
        Returns:
            Tuple of (cid, shard, offset, length)
        """
        if not data:
            raise ValueError("Cannot store empty data")
        
        # Compute CID
        cid = self._compute_cid(data)
        
        with self._lock:
            # Check cache first
            if cid in self._cid_cache:
                shard, offset, length = self._cid_cache[cid]
                # Increment refcount
                self._increment_refcount(cid)
                return (cid, shard, offset, length)
            
            # Check database
            existing = self._lookup_cid(cid)
            if existing:
                shard, offset, length = existing
                # Update cache
                self._update_cache(cid, shard, offset, length)
                # Increment refcount
                self._increment_refcount(cid)
                return (cid, shard, offset, length)
            
            # Write new data
            shard, offset, length = self.writer.append(data)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cid_index (cid, shard, offset, length)
                    VALUES (?, ?, ?, ?)
                """, (cid, shard, offset, length))
                conn.commit()
            
            # Update cache
            self._update_cache(cid, shard, offset, length)
            
            return (cid, shard, offset, length)
    
    def get(self, cid: str, range_hint: Optional[Tuple[int, int]] = None) -> bytes:
        """
        Retrieve data by CID.
        
        Args:
            cid: Content identifier
            range_hint: Optional (start, length) for partial read
            
        Returns:
            Data bytes
        """
        # Check cache
        if cid in self._cid_cache:
            shard, offset, length = self._cid_cache[cid]
        else:
            # Lookup in database
            location = self._lookup_cid(cid)
            if not location:
                raise KeyError(f"CID not found: {cid}")
            shard, offset, length = location
            
            # Update cache
            self._update_cache(cid, shard, offset, length)
        
        # Read data
        if range_hint:
            range_start, range_len = range_hint
            return self.reader.read_range(shard, offset, length, range_start, range_len)
        else:
            return self.reader.read(shard, offset, length)
    
    def _lookup_cid(self, cid: str) -> Optional[Tuple[int, int, int]]:
        """Lookup CID in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT shard, offset, length FROM cid_index
                WHERE cid = ?
            """, (cid,))
            row = cursor.fetchone()
            
            if row:
                return tuple(row)
            return None
    
    def _increment_refcount(self, cid: str):
        """Increment reference count for a CID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE cid_index SET refcount = refcount + 1
                WHERE cid = ?
            """, (cid,))
            conn.commit()
    
    def _decrement_refcount(self, cid: str) -> int:
        """Decrement reference count for a CID, return new count."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE cid_index SET refcount = MAX(0, refcount - 1)
                WHERE cid = ?
            """, (cid,))
            
            cursor = conn.execute("""
                SELECT refcount FROM cid_index WHERE cid = ?
            """, (cid,))
            row = cursor.fetchone()
            conn.commit()
            
            return row[0] if row else 0
    
    def _update_cache(self, cid: str, shard: int, offset: int, length: int):
        """Update LRU cache."""
        # Simple cache eviction
        if len(self._cid_cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest = next(iter(self._cid_cache))
            del self._cid_cache[oldest]
        
        self._cid_cache[cid] = (shard, offset, length)
    
    def exists(self, cid: str) -> bool:
        """Check if CID exists in store."""
        if cid in self._cid_cache:
            return True
        return self._lookup_cid(cid) is not None
    
    def delete(self, cid: str) -> bool:
        """
        Mark CID for deletion (decrements refcount).
        Actual deletion happens during garbage collection.
        """
        new_count = self._decrement_refcount(cid)
        
        # Remove from cache if refcount is 0
        if new_count == 0 and cid in self._cid_cache:
            del self._cid_cache[cid]
        
        return new_count == 0
    
    def garbage_collect(self) -> int:
        """
        Remove entries with refcount=0.
        Note: This doesn't reclaim shard space (would need compaction).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM cid_index WHERE refcount = 0
                RETURNING cid
            """)
            deleted = cursor.fetchall()
            conn.commit()
            
            # Clear from cache
            for (cid,) in deleted:
                if cid in self._cid_cache:
                    del self._cid_cache[cid]
            
            return len(deleted)
    
    def get_stats(self) -> dict:
        """Get CAS statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_objects,
                    SUM(length) as total_bytes,
                    SUM(refcount) as total_refs,
                    AVG(length) as avg_size
                FROM cid_index
            """)
            row = cursor.fetchone()
            
            stats = {
                'total_objects': row[0] or 0,
                'total_bytes': row[1] or 0,
                'total_refs': row[2] or 0,
                'avg_size': row[3] or 0,
                'cache_size': len(self._cid_cache),
                'using_blake3': HAS_BLAKE3
            }
            
            # Add shard stats
            stats.update(self.writer.get_stats())
            
            return stats
    
    def close(self):
        """Close CAS and flush pending writes."""
        self.writer.close()
        self.reader.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()