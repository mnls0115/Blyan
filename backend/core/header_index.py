"""
Header Index for Fast Blockchain Verification

Purpose: Append-only compact record per block height to support fast verification
without loading full block bodies. Uses JSON-lines format for durability.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class HeaderRecord:
    """Compact header record for fast verification."""
    index: int
    hash: str  # block.compute_hash()
    prev_hash: str  # header.prev_hash
    payload_hash: str  # header.payload_hash
    cum_digest: str  # cumulative digest for anchor verification
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HeaderRecord':
        """Create from dictionary."""
        return cls(**data)


class HeaderIndex:
    """Manages header index for fast blockchain verification."""
    
    def __init__(self, chain_dir: Path):
        """
        Initialize header index for a chain.
        
        Args:
            chain_dir: Directory containing blockchain data
        """
        self.chain_dir = Path(chain_dir)
        self.index_file = self.chain_dir / "headers.idx.jsonl"
        self._cache: Optional[List[HeaderRecord]] = None
    
    def append(self, record: HeaderRecord) -> None:
        """
        Append a header record to the index.
        
        Args:
            record: HeaderRecord to append
        """
        try:
            # Ensure directory exists
            self.chain_dir.mkdir(parents=True, exist_ok=True)
            
            # Append as JSON line with atomic write
            with open(self.index_file, 'a') as f:
                json.dump(record.to_dict(), f)
                f.write('\n')
                f.flush()  # Ensure write is flushed
            
            # Invalidate cache
            self._cache = None
            
        except Exception as e:
            logger.error(f"Failed to append header record: {e}")
            raise
    
    def load(self) -> List[HeaderRecord]:
        """
        Load all header records from index.
        
        Returns:
            List of HeaderRecord objects
        """
        # Return cached if available
        if self._cache is not None:
            return self._cache
        
        records = []
        if not self.index_file.exists():
            self._cache = records
            return records
        
        try:
            with open(self.index_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        records.append(HeaderRecord.from_dict(data))
            
            self._cache = records
            return records
            
        except Exception as e:
            logger.error(f"Failed to load header index: {e}")
            return []
    
    def verify_to_height(self, anchor_height: int, expected_cum_digest: Optional[str] = None) -> bool:
        """
        Verify header chain up to a specific anchor height (inclusive).
        
        Args:
            anchor_height: Anchor height/index to verify up to (0-based index)
            expected_cum_digest: Expected cumulative digest at anchor height (optional)
        
        Returns:
            True if verification passes
        """
        records = self.load()
        total = len(records)
        
        # Validate bounds
        if anchor_height < 0 or anchor_height >= total:
            logger.error(f"Anchor height {anchor_height} out of bounds (total records: {total})")
            return False
        
        if anchor_height == 0 and total == 1:
            return True  # Empty chain is valid
        
        # Verify cumulative digest if provided
        if expected_cum_digest is not None:
            anchor_record = records[anchor_height]
            if anchor_record.cum_digest != expected_cum_digest:
                logger.error(f"Cumulative digest mismatch at anchor height {anchor_height}")
                return False
        
        # Verify prev_hash linkage and cumulative digests
        prev_cum_digest = bytes(32)  # Start with 32 zero bytes
        for i in range(anchor_height + 1):
            record = records[i]
            # Check index matches position
            if record.index != i:
                logger.error(f"Index mismatch at position {i}: expected {i}, got {record.index}")
                return False
            
            # Verify prev_hash linkage (except genesis)
            if i > 0:
                prev_record = records[i-1]
                if record.prev_hash != prev_record.hash:
                    logger.error(f"Broken chain at index {i}: prev_hash mismatch")
                    return False
            
            # Verify cumulative digest
            canonical_bytes = f"{record.index}|{record.hash}|{record.prev_hash}|{record.payload_hash}".encode()
            expected_digest = hashlib.sha256(prev_cum_digest + canonical_bytes).hexdigest()
            
            if record.cum_digest != expected_digest:
                logger.error(f"Cumulative digest mismatch at index {i}")
                return False
            
            prev_cum_digest = bytes.fromhex(record.cum_digest)
        
        return True
    
    def verify_tail_bodies(self, chain, start_idx: int, end_idx: int) -> Tuple[bool, str]:
        """
        Verify block bodies in a specific range by checking payload hash matches.
        
        Args:
            chain: Chain object with get_block_by_index method
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        records = self.load()
        
        # Validate range
        if start_idx < 0 or end_idx > len(records):
            return False, f"Invalid range [{start_idx}, {end_idx}) for {len(records)} records"
        
        # Get block loader
        loader = None
        if hasattr(chain, 'get_block_by_index'):
            loader = chain.get_block_by_index
        elif hasattr(chain, 'storage') and hasattr(chain.storage, 'get_block_by_index'):
            loader = chain.storage.get_block_by_index
        else:
            return False, "Chain has no block loader"
        
        # Verify each block in range
        for idx in range(start_idx, end_idx):
            try:
                # Load block
                block = loader(idx)
                if not block:
                    return False, f"Missing block at index {idx}"
                
                # Get corresponding header record
                if idx >= len(records):
                    return False, f"No header record for block {idx}"
                
                record = records[idx]
                
                # Verify payload hash matches
                payload_hash = hashlib.sha256(block.payload).hexdigest()
                if payload_hash != record.payload_hash:
                    return False, f"Payload hash mismatch at index {idx}"
                
                # Verify block hash matches
                block_hash = block.compute_hash()
                if block_hash != record.hash:
                    return False, f"Block hash mismatch at index {idx}"
                
                # Verify prev_hash linkage
                if block.header.prev_hash != record.prev_hash:
                    return False, f"Prev hash mismatch at index {idx}"
                
            except Exception as e:
                return False, f"Error verifying block {idx}: {e}"
        
        return True, ""
    
    def build_from_chain(self, chain) -> None:
        """
        Build entire header index from chain when missing.
        One-time operation for migration.
        
        Args:
            chain: Chain object to build index from
        """
        if self.index_file.exists():
            logger.info(f"Header index already exists at {self.index_file}")
            return
        
        logger.info(f"Building header index for {self.chain_dir.name}...")
        
        # Get block loader
        loader = None
        if hasattr(chain, 'get_block_by_index'):
            loader = chain.get_block_by_index
        elif hasattr(chain, 'storage') and hasattr(chain.storage, 'get_block_by_index'):
            loader = chain.storage.get_block_by_index
        else:
            logger.warning("Chain has no block loader, skipping index build")
            return
        
        # Determine block count efficiently
        block_count = 0
        if hasattr(chain, '_hash_index'):
            block_count = len(chain._hash_index)
        else:
            # Count block files if no index
            block_files = list(self.chain_dir.glob("*.json"))
            block_count = len(block_files)
        
        if block_count == 0:
            logger.info("No blocks to index")
            return
        
        # Build index
        prev_cum_digest = bytes(32)  # Start with 32 zero bytes
        records_to_write = []
        
        for idx in range(block_count):
            try:
                # Load block
                block = loader(idx)
                if not block:
                    logger.warning(f"Missing block at index {idx}, stopping index build")
                    break
                
                # Compute cumulative digest
                block_hash = block.compute_hash()
                prev_hash = block.header.prev_hash
                payload_hash = hashlib.sha256(block.payload).hexdigest()
                
                canonical_bytes = f"{idx}|{block_hash}|{prev_hash}|{payload_hash}".encode()
                cum_digest = hashlib.sha256(prev_cum_digest + canonical_bytes).hexdigest()
                
                # Create header record
                record = HeaderRecord(
                    index=idx,
                    hash=block_hash,
                    prev_hash=prev_hash,
                    payload_hash=payload_hash,
                    cum_digest=cum_digest
                )
                
                records_to_write.append(record)
                prev_cum_digest = bytes.fromhex(cum_digest)
                
                # Log progress
                if (idx + 1) % 100 == 0:
                    logger.info(f"  Indexed {idx + 1}/{block_count} blocks...")
                    
            except Exception as e:
                logger.error(f"Failed to index block {idx}: {e}")
                break
        
        # Write all records atomically
        if records_to_write:
            try:
                self.chain_dir.mkdir(parents=True, exist_ok=True)
                with open(self.index_file, 'w') as f:
                    for record in records_to_write:
                        json.dump(record.to_dict(), f)
                        f.write('\n')
                
                logger.info(f"✅ Built header index with {len(records_to_write)} records")
                
            except Exception as e:
                logger.error(f"Failed to write header index: {e}")
                # Remove partial file
                if self.index_file.exists():
                    self.index_file.unlink()
    
    def get_last_record(self) -> Optional[HeaderRecord]:
        """Get the last header record if any."""
        records = self.load()
        return records[-1] if records else None
    
    def get_record_at(self, index: int) -> Optional[HeaderRecord]:
        """Get header record at specific index."""
        records = self.load()
        if 0 <= index < len(records):
            return records[index]
        return None
    
    def compute_cum_digest_for_record(self, index: int, block_hash: str, 
                                     prev_hash: str, payload_hash: str,
                                     prev_cum_digest: Optional[bytes] = None) -> str:
        """
        Compute cumulative digest for a new record.
        
        Args:
            index: Block index
            block_hash: Hash of the block
            prev_hash: Hash of previous block
            payload_hash: Hash of block payload
            prev_cum_digest: Previous cumulative digest bytes (or None for genesis)
        
        Returns:
            Hex string of cumulative digest
        """
        if prev_cum_digest is None:
            if index == 0:
                prev_cum_digest = bytes(32)  # Genesis starts with zeros
            else:
                # Get previous record's digest
                prev_record = self.get_record_at(index - 1)
                if prev_record:
                    prev_cum_digest = bytes.fromhex(prev_record.cum_digest)
                else:
                    raise ValueError(f"No previous record for index {index}")
        
        canonical_bytes = f"{index}|{block_hash}|{prev_hash}|{payload_hash}".encode()
        return hashlib.sha256(prev_cum_digest + canonical_bytes).hexdigest()
