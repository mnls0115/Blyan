"""
Background Auditor for Long-term Blockchain Integrity

Purpose: Slowly verify bodies beyond the K-tail after startup and rotate anchors.
Keeps long-term integrity high with zero impact on startup.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BackgroundAuditor:
    """Background auditor for continuous blockchain integrity verification."""
    
    def __init__(self, data_dir: Path, chains: Dict[str, Any]):
        """
        Initialize background auditor.
        
        Args:
            data_dir: Data directory containing chains
            chains: Dictionary of chain objects
        """
        self.data_dir = Path(data_dir)
        self.chains = chains
        self.running = False
        self.audit_task: Optional[asyncio.Task] = None
        self.last_audit_time = 0
        self.audit_stats = {
            'blocks_audited': 0,
            'errors_found': 0,
            'anchors_rotated': 0
        }
    
    async def start(self, interval_seconds: int = 300):
        """
        Start background auditing.
        
        Args:
            interval_seconds: Seconds between audit cycles (default 5 minutes)
        """
        if self.running:
            logger.debug("Background auditor already running")
            return
        
        self.running = True
        self.audit_task = asyncio.create_task(self._audit_loop(interval_seconds))
        logger.info(f"🔍 Background auditor started (interval: {interval_seconds}s)")
    
    async def stop(self):
        """Stop background auditing."""
        self.running = False
        if self.audit_task:
            self.audit_task.cancel()
            try:
                await self.audit_task
            except asyncio.CancelledError:
                pass
        logger.info("🔍 Background auditor stopped")
    
    async def _audit_loop(self, interval_seconds: int):
        """Main audit loop."""
        while self.running:
            try:
                # Wait for interval
                await asyncio.sleep(interval_seconds)
                
                # Run audit cycle
                await self._audit_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background audit error: {e}")
                # Continue after error
    
    async def _audit_cycle(self):
        """Run one audit cycle."""
        start_time = time.time()
        logger.debug("Starting background audit cycle")
        
        # Audit each chain
        for chain_id in ['A', 'B']:
            chain = self.chains.get(chain_id)
            if not chain:
                continue
            
            try:
                # Run in thread pool to avoid blocking
                await asyncio.to_thread(self._audit_chain, chain_id, chain)
            except Exception as e:
                logger.error(f"Failed to audit chain {chain_id}: {e}")
                self.audit_stats['errors_found'] += 1
        
        # Update stats
        self.last_audit_time = time.time()
        elapsed = self.last_audit_time - start_time
        logger.debug(f"Audit cycle completed in {elapsed:.2f}s")
    
    def _audit_chain(self, chain_id: str, chain):
        """
        Audit a single chain (runs in thread pool).
        
        Args:
            chain_id: Chain identifier
            chain: Chain object
        """
        # Load finality anchor
        anchor_file = self.data_dir / f"finality_anchor_{chain_id}.json"
        anchor_data = None
        
        if anchor_file.exists():
            try:
                with open(anchor_file, 'r') as f:
                    anchor_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load anchor for chain {chain_id}: {e}")
                return
        
        if not anchor_data:
            logger.debug(f"No anchor for chain {chain_id}, skipping audit")
            return
        
        # Get current chain height
        current_height = 0
        if hasattr(chain, '_hash_index'):
            current_height = len(chain._hash_index)
        else:
            # Count block files
            chain_dir = self.data_dir / chain_id
            if chain_dir.exists():
                current_height = len(list(chain_dir.glob("*.json")))
        
        anchor_height = anchor_data.get('height', 0)
        
        # Check if we should rotate anchor
        FINALITY_DEPTH = int(os.getenv('FINALITY_DEPTH', '512'))
        AUDIT_BATCH_SIZE = min(50, FINALITY_DEPTH // 10)  # Audit 50 blocks or 10% of finality depth
        
        if current_height >= anchor_height + FINALITY_DEPTH:
            # Time to rotate anchor
            logger.info(f"Chain {chain_id}: Rotating finality anchor from {anchor_height} to {current_height - FINALITY_DEPTH // 2}")
            
            # Audit blocks between old anchor and new anchor
            audit_start = anchor_height
            audit_end = min(anchor_height + AUDIT_BATCH_SIZE, current_height - FINALITY_DEPTH // 2)
            
            if self._verify_block_range(chain, audit_start, audit_end):
                # Update anchor
                new_anchor_height = current_height - FINALITY_DEPTH // 2
                
                # Get cumulative digest at new anchor
                from backend.core.header_index import HeaderIndex
                header_index = HeaderIndex(self.data_dir / chain_id)
                records = header_index.load()
                
                if new_anchor_height < len(records):
                    new_anchor_record = records[new_anchor_height]
                    
                    new_anchor_data = {
                        'height': new_anchor_height,
                        'cum_digest': new_anchor_record.cum_digest,
                        'updated_at': datetime.utcnow().isoformat() + 'Z',
                        'auditor': 'background'
                    }
                    
                    # Write new anchor atomically
                    temp_file = anchor_file.with_suffix('.tmp')
                    with open(temp_file, 'w') as f:
                        json.dump(new_anchor_data, f, indent=2)
                    temp_file.replace(anchor_file)
                    
                    logger.info(f"✅ Chain {chain_id}: Anchor rotated to height {new_anchor_height}")
                    self.audit_stats['anchors_rotated'] += 1
                
                self.audit_stats['blocks_audited'] += (audit_end - audit_start)
            else:
                logger.error(f"Chain {chain_id}: Audit failed, not rotating anchor")
                self.audit_stats['errors_found'] += 1
        else:
            # Just audit a small batch of old blocks
            if anchor_height > AUDIT_BATCH_SIZE:
                audit_start = anchor_height - AUDIT_BATCH_SIZE
                audit_end = anchor_height
                
                if self._verify_block_range(chain, audit_start, audit_end):
                    self.audit_stats['blocks_audited'] += AUDIT_BATCH_SIZE
                    logger.debug(f"Chain {chain_id}: Audited {AUDIT_BATCH_SIZE} blocks before anchor")
                else:
                    self.audit_stats['errors_found'] += 1
    
    def _verify_block_range(self, chain, start_idx: int, end_idx: int) -> bool:
        """
        Verify a range of blocks.
        
        Args:
            chain: Chain object
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
        
        Returns:
            True if all blocks verify
        """
        # Get block loader
        loader = None
        if hasattr(chain, 'get_block_by_index'):
            loader = chain.get_block_by_index
        elif hasattr(chain, 'storage') and hasattr(chain.storage, 'get_block_by_index'):
            loader = chain.storage.get_block_by_index
        else:
            return False
        
        prev_hash = None
        for idx in range(start_idx, end_idx):
            try:
                block = loader(idx)
                if not block:
                    logger.error(f"Missing block at index {idx}")
                    return False
                
                # Verify payload hash
                payload_hash = hashlib.sha256(block.payload).hexdigest()
                if payload_hash != block.header.payload_hash:
                    logger.error(f"Payload hash mismatch at index {idx}")
                    return False
                
                # Verify prev linkage (skip genesis)
                if idx > 0 and prev_hash is not None:
                    if block.header.prev_hash != prev_hash:
                        logger.error(f"Broken chain at index {idx}")
                        return False
                
                prev_hash = block.compute_hash()
                
            except Exception as e:
                logger.error(f"Error verifying block {idx}: {e}")
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get audit statistics."""
        return {
            'running': self.running,
            'last_audit': self.last_audit_time,
            'blocks_audited': self.audit_stats['blocks_audited'],
            'errors_found': self.audit_stats['errors_found'],
            'anchors_rotated': self.audit_stats['anchors_rotated']
        }


import os  # Import for FINALITY_DEPTH env var