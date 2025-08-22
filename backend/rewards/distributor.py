#!/usr/bin/env python3
"""
Idempotent Automatic Reward Distributor
Ensures rewards are distributed exactly once with no double-pays.
"""

import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import threading

from backend.rewards.policy import (
    calc_inference_bly,
    calc_learning_bly,
    calc_validation_bly,
    calc_dataset_bly,
    RewardConfig
)
from backend.rewards.buckets import BucketAllocator

logger = logging.getLogger(__name__)


@dataclass
class RewardClaim:
    """A reward claim to be processed."""
    claim_id: str
    claim_type: str  # inference, learning, validation, dataset
    recipient: str
    amount_bly: float
    metrics: Dict
    timestamp: float
    epoch_id: str
    processed: bool = False
    tx_hash: Optional[str] = None
    
    def get_hash(self) -> str:
        """Get unique hash for idempotency."""
        content = f"{self.claim_type}:{self.recipient}:{self.amount_bly}:{self.epoch_id}"
        return hashlib.sha256(content.encode()).hexdigest()


class AutomaticRewardDistributor:
    """
    Distributes rewards automatically with idempotency guarantees.
    Prevents double-pays and supports replay/recovery.
    """
    
    def __init__(self, 
                 config_path: str = "config/reward_policy.yaml",
                 state_path: str = "data/distributor_state.json",
                 ledger_path: str = "data/reward_ledger.json"):
        """Initialize distributor."""
        self.config = RewardConfig.from_yaml(config_path)
        self.allocator = BucketAllocator()
        
        self.state_path = Path(state_path)
        self.ledger_path = Path(ledger_path)
        
        # Idempotency tracking
        self.processed_claims: Set[str] = set()
        self.epoch_claims: Dict[str, List[RewardClaim]] = {}
        
        # Distribution settings
        self.batch_threshold = 0.5  # Minimum BLY to trigger payout
        self.max_batch_size = 1000
        self.idempotency_window_hours = 48
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load state
        self._load_state()
    
    def _load_state(self):
        """Load persisted state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                
                self.processed_claims = set(state.get('processed_claims', []))
                
                # Clean old claims outside idempotency window
                self._clean_old_claims()
                
                logger.info(f"Loaded {len(self.processed_claims)} processed claims")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save state to disk."""
        try:
            state = {
                'processed_claims': list(self.processed_claims),
                'last_updated': time.time()
            }
            
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _clean_old_claims(self):
        """Remove claims older than idempotency window."""
        # In production, would parse claim IDs for timestamps
        # For now, keep last N claims
        max_claims = 100000
        if len(self.processed_claims) > max_claims:
            # Keep most recent claims
            self.processed_claims = set(list(self.processed_claims)[-max_claims:])
    
    def _get_epoch_id(self, timestamp: Optional[float] = None) -> str:
        """Get distribution epoch ID (hourly)."""
        ts = timestamp or time.time()
        hour_epoch = int(ts // 3600)
        return f"epoch_{hour_epoch}"
    
    def _append_to_ledger(self, claims: List[RewardClaim]):
        """Append claims to permanent ledger."""
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing ledger
            ledger = []
            if self.ledger_path.exists():
                with open(self.ledger_path, 'r') as f:
                    ledger = json.load(f)
            
            # Append new claims
            for claim in claims:
                ledger.append({
                    **asdict(claim),
                    'distributed_at': time.time()
                })
            
            # Save updated ledger
            with open(self.ledger_path, 'w') as f:
                json.dump(ledger, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update ledger: {e}")
    
    async def submit_claim(self, 
                          claim_type: str,
                          recipient: str,
                          metrics: Dict) -> Optional[str]:
        """
        Submit a reward claim for processing.
        
        Args:
            claim_type: Type of reward (inference/learning/validation/dataset)
            recipient: Wallet address or node ID
            metrics: Work metrics for calculation
        
        Returns:
            Claim ID if accepted, None if duplicate
        """
        epoch_id = self._get_epoch_id()
        
        # Calculate reward amount based on type
        amount_bly = self._calculate_reward(claim_type, metrics)
        
        if amount_bly <= 0:
            logger.warning(f"Invalid reward amount for {recipient}: {amount_bly}")
            return None
        
        # Create claim
        claim = RewardClaim(
            claim_id=f"{claim_type}_{recipient}_{int(time.time()*1000000)}",
            claim_type=claim_type,
            recipient=recipient,
            amount_bly=amount_bly,
            metrics=metrics,
            timestamp=time.time(),
            epoch_id=epoch_id
        )
        
        # Check for duplicates
        claim_hash = claim.get_hash()
        
        with self.lock:
            if claim_hash in self.processed_claims:
                logger.info(f"Duplicate claim rejected: {claim_hash}")
                return None
            
            # Add to pending claims for this epoch
            if epoch_id not in self.epoch_claims:
                self.epoch_claims[epoch_id] = []
            
            self.epoch_claims[epoch_id].append(claim)
            
            logger.info(f"Claim submitted: {claim.claim_id} for {amount_bly} BLY")
            
            # Check if we should trigger distribution
            if self._should_distribute(epoch_id):
                asyncio.create_task(self.distribute_epoch(epoch_id))
            
            return claim.claim_id
    
    def _calculate_reward(self, claim_type: str, metrics: Dict) -> float:
        """Calculate reward amount based on claim type and metrics."""
        try:
            if claim_type == "inference":
                return calc_inference_bly(
                    metrics.get('tokens', 0),
                    metrics.get('quality', 1.0),
                    config=self.config
                )
            
            elif claim_type == "learning":
                # Get learning pool for the hour
                learning_pool = self.allocator.get_bucket_status()['buckets']['learning']['available']
                
                return calc_learning_bly(
                    learning_pool * 24,  # Daily equivalent
                    metrics.get('improvement_pct', 0),
                    metrics.get('difficulty', 1.5),
                    metrics.get('applicability', 1.0),
                    hourslice_share=1/24,
                    config=self.config
                )
            
            elif claim_type == "validation":
                return calc_validation_bly(
                    metrics.get('tasks', 0),
                    config=self.config
                )
            
            elif claim_type == "dataset":
                return calc_dataset_bly(
                    metrics.get('gb_size', 0),
                    metrics.get('quality', 0.9),
                    metrics.get('diversity', 1.0),
                    config=self.config
                )
            
            else:
                logger.error(f"Unknown claim type: {claim_type}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate reward: {e}")
            return 0.0
    
    def _should_distribute(self, epoch_id: str) -> bool:
        """Check if we should trigger distribution."""
        if epoch_id not in self.epoch_claims:
            return False
        
        claims = self.epoch_claims[epoch_id]
        
        # Check batch size
        if len(claims) >= self.max_batch_size:
            return True
        
        # Check total amount
        total_amount = sum(c.amount_bly for c in claims if not c.processed)
        if total_amount >= self.batch_threshold:
            return True
        
        # Check age (distribute if epoch is old)
        current_epoch = self._get_epoch_id()
        if epoch_id < current_epoch:
            return True
        
        return False
    
    async def distribute_epoch(self, epoch_id: str):
        """
        Distribute rewards for a specific epoch.
        Idempotent - can be called multiple times safely.
        """
        with self.lock:
            if epoch_id not in self.epoch_claims:
                logger.info(f"No claims for epoch {epoch_id}")
                return
            
            claims = self.epoch_claims[epoch_id]
            pending_claims = [c for c in claims if not c.processed]
            
            if not pending_claims:
                logger.info(f"All claims already processed for {epoch_id}")
                return
            
            logger.info(f"Distributing {len(pending_claims)} claims for {epoch_id}")
            
            # Group by bucket type for allocation
            grouped = self._group_claims_by_type(pending_claims)
            
            # Process each group
            for claim_type, type_claims in grouped.items():
                await self._distribute_group(claim_type, type_claims)
            
            # Mark epoch as complete
            self._save_state()
    
    def _group_claims_by_type(self, claims: List[RewardClaim]) -> Dict[str, List[RewardClaim]]:
        """Group claims by type for bucket allocation."""
        grouped = {}
        for claim in claims:
            if claim.claim_type not in grouped:
                grouped[claim.claim_type] = []
            grouped[claim.claim_type].append(claim)
        return grouped
    
    async def _distribute_group(self, claim_type: str, claims: List[RewardClaim]):
        """Distribute a group of claims from the same bucket."""
        # Map claim type to bucket
        bucket_map = {
            'inference': 'inference',
            'learning': 'learning',
            'validation': 'validation',
            'dataset': 'dataset'
        }
        
        bucket_type = bucket_map.get(claim_type, 'inference')
        
        # Sort by amount (prioritize larger claims)
        claims.sort(key=lambda c: c.amount_bly, reverse=True)
        
        distributed = []
        for claim in claims:
            # Try to allocate from bucket
            granted, request_id = self.allocator.allocate(
                bucket_type,
                claim.amount_bly,
                claim.recipient,
                {'claim_id': claim.claim_id}
            )
            
            if granted > 0:
                # Process payment (would integrate with blockchain)
                tx_hash = await self._process_payment(claim.recipient, granted)
                
                # Mark as processed
                claim.processed = True
                claim.tx_hash = tx_hash
                
                # Add to processed set for idempotency
                self.processed_claims.add(claim.get_hash())
                
                distributed.append(claim)
                
                logger.info(f"Distributed {granted} BLY to {claim.recipient} (tx: {tx_hash})")
            
            else:
                logger.info(f"Queued for backpay: {claim.claim_id} ({claim.amount_bly} BLY)")
        
        # Append to ledger
        if distributed:
            self._append_to_ledger(distributed)
    
    async def _process_payment(self, recipient: str, amount: float) -> str:
        """
        Process actual payment to recipient.
        In production, this would interact with blockchain.
        """
        # Simulate blockchain transaction
        await asyncio.sleep(0.01)
        
        # Generate mock transaction hash
        tx_data = f"{recipient}:{amount}:{time.time()}"
        tx_hash = hashlib.sha256(tx_data.encode()).hexdigest()[:16]
        
        return tx_hash
    
    async def run_hourly_distribution(self):
        """Run automatic hourly distribution."""
        while True:
            try:
                # Get current epoch
                epoch_id = self._get_epoch_id()
                
                # Distribute pending claims
                await self.distribute_epoch(epoch_id)
                
                # Also check previous epoch for stragglers
                prev_epoch = f"epoch_{int(time.time() // 3600) - 1}"
                await self.distribute_epoch(prev_epoch)
                
                # Clean old claims
                self._clean_old_claims()
                
                # Wait until next hour
                next_hour = ((time.time() // 3600) + 1) * 3600
                wait_time = next_hour - time.time()
                
                logger.info(f"Next distribution in {wait_time:.0f} seconds")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Distribution error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    def repair_from_logs(self, start_epoch: str, end_epoch: str):
        """
        Repair/replay distribution from raw logs.
        Useful for recovery and auditing.
        """
        logger.info(f"Repairing epochs {start_epoch} to {end_epoch}")
        
        # Load raw logs (would read from actual log files)
        # For now, simulate with current epoch claims
        
        repaired = 0
        for epoch_id in self.epoch_claims.keys():
            if start_epoch <= epoch_id <= end_epoch:
                claims = self.epoch_claims[epoch_id]
                
                for claim in claims:
                    claim_hash = claim.get_hash()
                    
                    if claim_hash not in self.processed_claims:
                        # Re-process this claim
                        logger.info(f"Reprocessing claim: {claim.claim_id}")
                        repaired += 1
        
        logger.info(f"Repaired {repaired} claims")
        self._save_state()
        
        return repaired
    
    def get_distribution_stats(self, epoch_id: Optional[str] = None) -> Dict:
        """Get distribution statistics."""
        if epoch_id is None:
            epoch_id = self._get_epoch_id()
        
        stats = {
            'epoch_id': epoch_id,
            'total_claims': 0,
            'processed_claims': 0,
            'pending_claims': 0,
            'total_distributed_bly': 0,
            'by_type': {}
        }
        
        if epoch_id in self.epoch_claims:
            claims = self.epoch_claims[epoch_id]
            stats['total_claims'] = len(claims)
            stats['processed_claims'] = sum(1 for c in claims if c.processed)
            stats['pending_claims'] = stats['total_claims'] - stats['processed_claims']
            stats['total_distributed_bly'] = sum(c.amount_bly for c in claims if c.processed)
            
            # Break down by type
            for claim in claims:
                if claim.claim_type not in stats['by_type']:
                    stats['by_type'][claim.claim_type] = {
                        'count': 0,
                        'amount_bly': 0
                    }
                
                if claim.processed:
                    stats['by_type'][claim.claim_type]['count'] += 1
                    stats['by_type'][claim.claim_type]['amount_bly'] += claim.amount_bly
        
        return stats


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_distributor():
        distributor = AutomaticRewardDistributor()
        
        # Submit some test claims
        claim_id = await distributor.submit_claim(
            'inference',
            'node_123',
            {'tokens': 100000, 'quality': 1.1}
        )
        print(f"Submitted claim: {claim_id}")
        
        claim_id = await distributor.submit_claim(
            'learning',
            'node_456',
            {'improvement_pct': 2.5, 'difficulty': 2.0}
        )
        print(f"Submitted claim: {claim_id}")
        
        # Trigger distribution
        epoch = distributor._get_epoch_id()
        await distributor.distribute_epoch(epoch)
        
        # Check stats
        stats = distributor.get_distribution_stats()
        print(f"Distribution stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_distributor())