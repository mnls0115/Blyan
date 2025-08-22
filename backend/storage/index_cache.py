"""
Index cache layer for instant restarts, keyed by manifest head/version.
"""

import sqlite3
import time
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class IndexCache:
    """
    Fast index cache using SQLite for persistence.
    Enables sub-200ms warm boots.
    """
    
    def __init__(self, cache_dir: Path, manifest_head: str):
        """
        Initialize index cache.
        
        Args:
            cache_dir: Directory to store cache files
            manifest_head: Manifest head hash for versioning
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_head = manifest_head
        
        # Use manifest head in filename for versioning
        safe_head = manifest_head[:16] if manifest_head else "unknown"
        self.db_path = self.cache_dir / f"index_cache_{safe_head}.db"
        
        self.entries: Dict[Tuple, Tuple[str, int, int, int]] = {}
        self._initialized = False
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Create metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Create index table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS block_index (
                    key TEXT PRIMARY KEY,
                    cid TEXT NOT NULL,
                    shard INTEGER NOT NULL,
                    offset INTEGER NOT NULL,
                    length INTEGER NOT NULL
                )
            """)
            
            # Store metadata
            conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('manifest_head', ?), ('created_at', ?), ('entry_count', ?)
            """, (self.manifest_head, str(time.time()), '0'))
            
            conn.commit()
    
    @classmethod
    def load(cls, cache_dir: Path, manifest_head: str) -> Optional['IndexCache']:
        """
        Load cache from disk if it matches manifest head.
        
        Args:
            cache_dir: Cache directory
            manifest_head: Expected manifest head
            
        Returns:
            IndexCache if valid, None if mismatch or not found
        """
        start_time = time.perf_counter()
        
        cache = cls(cache_dir, manifest_head)
        
        if not cache.db_path.exists():
            logger.info(f"Cache miss: file not found")
            return None
        
        try:
            with sqlite3.connect(cache.db_path) as conn:
                # Check manifest head
                cursor = conn.execute("""
                    SELECT value FROM metadata WHERE key = 'manifest_head'
                """)
                row = cursor.fetchone()
                
                if not row or row[0] != manifest_head:
                    logger.info(f"Cache miss: manifest head mismatch")
                    return None
                
                # Load all entries
                cursor = conn.execute("""
                    SELECT key, cid, shard, offset, length FROM block_index
                """)
                
                for row in cursor:
                    key_str, cid, shard, offset, length = row
                    # Parse key tuple from string
                    key = tuple(key_str.split('|'))
                    cache.entries[key] = (cid, shard, offset, length)
                
                cache._initialized = True
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"Cache hit: loaded {len(cache.entries)} entries in {elapsed_ms:.1f}ms")
                
                return cache
                
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def save(self, entries: Dict[Tuple, Tuple[str, int, int, int]]):
        """
        Save entries to cache.
        
        Args:
            entries: Dictionary of key -> (cid, shard, offset, length)
        """
        start_time = time.perf_counter()
        
        self._init_db()
        
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing entries
            conn.execute("DELETE FROM block_index")
            
            # Batch insert new entries
            batch = []
            for key, (cid, shard, offset, length) in entries.items():
                # Convert key tuple to string
                key_str = '|'.join(str(k) for k in key)
                batch.append((key_str, cid, shard, offset, length))
            
            conn.executemany("""
                INSERT INTO block_index (key, cid, shard, offset, length)
                VALUES (?, ?, ?, ?, ?)
            """, batch)
            
            # Update metadata
            conn.execute("""
                UPDATE metadata SET value = ? WHERE key = 'entry_count'
            """, (str(len(entries)),))
            
            conn.commit()
        
        self.entries = entries
        self._initialized = True
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Saved {len(entries)} entries to cache in {elapsed_ms:.1f}ms")
    
    def get(self, key: Tuple) -> Optional[Tuple[str, int, int, int]]:
        """
        Get entry from cache.
        
        Args:
            key: Entry key tuple
            
        Returns:
            Tuple of (cid, shard, offset, length) or None
        """
        return self.entries.get(key)
    
    def build_from_manifest(self, manifest):
        """
        Build cache from manifest.
        
        Args:
            manifest: Manifest object
        """
        start_time = time.perf_counter()
        
        entries = {}
        for entry in manifest.objects:
            entries[entry.key] = (entry.cid, entry.shard, entry.offset, entry.length)
        
        self.save(entries)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Built cache from manifest in {elapsed_ms:.1f}ms")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            'manifest_head': self.manifest_head[:16] + '...',
            'entry_count': len(self.entries),
            'db_size_kb': self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0,
            'initialized': self._initialized
        }
        
        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT value FROM metadata WHERE key = 'created_at'
                    """)
                    row = cursor.fetchone()
                    if row:
                        stats['created_at'] = float(row[0])
                        stats['age_seconds'] = time.time() - float(row[0])
            except:
                pass
        
        return stats
    
    def invalidate(self):
        """Invalidate cache by removing the database file."""
        if self.db_path.exists():
            self.db_path.unlink()
            logger.info(f"Cache invalidated: {self.db_path}")
        self.entries.clear()
        self._initialized = False
    
    @staticmethod
    def clean_old_caches(cache_dir: Path, keep_recent: int = 3):
        """
        Clean old cache files, keeping only the most recent ones.
        
        Args:
            cache_dir: Cache directory
            keep_recent: Number of recent caches to keep
        """
        cache_dir = Path(cache_dir)
        cache_files = list(cache_dir.glob("index_cache_*.db"))
        
        if len(cache_files) <= keep_recent:
            return
        
        # Sort by modification time
        cache_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old caches
        for old_cache in cache_files[keep_recent:]:
            try:
                old_cache.unlink()
                logger.info(f"Removed old cache: {old_cache}")
            except Exception as e:
                logger.warning(f"Failed to remove old cache: {e}")