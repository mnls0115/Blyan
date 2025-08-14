#!/usr/bin/env python3
"""
Consensus-based Distributed Learning for Blyan
Ensures all nodes train from same base state
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np

class LearningPhase(Enum):
    """Learning phases for synchronization"""
    SYNC = "sync"          # Synchronize to same base version
    TRAIN = "train"        # Local training
    AGGREGATE = "aggregate" # Delta aggregation
    COMMIT = "commit"      # Commit to blockchain

@dataclass
class LearningEpoch:
    """Represents a learning epoch with consensus"""
    epoch_id: str
    base_version: str  # All nodes must use this version
    participants: List[str]
    start_time: float
    end_time: Optional[float] = None
    aggregated_delta: Optional[Any] = None
    consensus_achieved: bool = False

class ConsensusLearningCoordinator:
    """
    Coordinates distributed learning with consensus
    
    Key features:
    1. All nodes start from same base version
    2. Synchronous epochs with checkpoints
    3. Byzantine fault tolerance for delta aggregation
    4. Deterministic versioning
    """
    
    def __init__(self, node_id: str, blockchain_manager: Any):
        self.node_id = node_id
        self.blockchain = blockchain_manager
        
        # Epoch management
        self.current_epoch: Optional[LearningEpoch] = None
        self.epoch_history: List[LearningEpoch] = []
        
        # Node synchronization
        self.peer_nodes: Dict[str, Dict] = {}  # node_id -> {last_seen, version}
        self.sync_threshold = 0.67  # 2/3 consensus required
        
        # Learning state
        self.base_tile_cache: Dict[str, torch.Tensor] = {}
        self.pending_deltas: List[Dict] = []
        
    async def start_epoch(self, tile_id: str, dataset_batch: Any) -> LearningEpoch:
        """
        Start a new learning epoch with consensus on base version
        """
        # Phase 1: Agree on base version
        base_version = await self.consensus_base_version(tile_id)
        
        # Create epoch
        epoch_id = hashlib.sha256(
            f"{tile_id}_{base_version}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        epoch = LearningEpoch(
            epoch_id=epoch_id,
            base_version=base_version,
            participants=[self.node_id],
            start_time=time.time()
        )
        
        self.current_epoch = epoch
        
        # Phase 2: Load base tile (all nodes load same version)
        base_tile = await self.load_base_tile(tile_id, base_version)
        self.base_tile_cache[tile_id] = base_tile
        
        print(f"📍 Epoch {epoch_id} started with base {base_version[:8]}")
        return epoch
    
    async def consensus_base_version(self, tile_id: str) -> str:
        """
        Achieve consensus on which base version to use
        """
        # Get latest committed version from blockchain
        latest_version = self.blockchain.get_latest_tile_version(tile_id)
        
        # Broadcast our version to peers
        votes = {self.node_id: latest_version}
        
        # Collect votes from peers
        for peer_id in self.peer_nodes:
            peer_version = await self.query_peer_version(peer_id, tile_id)
            if peer_version:
                votes[peer_id] = peer_version
        
        # Find majority version (Byzantine fault tolerant)
        version_counts = {}
        for version in votes.values():
            version_counts[version] = version_counts.get(version, 0) + 1
        
        # Select version with most votes
        consensus_version = max(version_counts, key=version_counts.get)
        consensus_ratio = version_counts[consensus_version] / len(votes)
        
        if consensus_ratio < self.sync_threshold:
            raise Exception(f"Consensus failed: only {consensus_ratio:.1%} agreement")
        
        print(f"✅ Consensus achieved: {consensus_ratio:.1%} agree on {consensus_version[:8]}")
        return consensus_version
    
    async def train_local(self, dataset_batch: Any) -> torch.Tensor:
        """
        Perform local training on agreed base
        """
        if not self.current_epoch:
            raise Exception("No active epoch")
        
        tile_id = list(self.base_tile_cache.keys())[0]
        base_tile = self.base_tile_cache[tile_id]
        
        # Local SGD steps
        delta = torch.zeros_like(base_tile)
        learning_rate = 0.01
        
        for _ in range(10):  # Local iterations
            # Mock gradient computation
            gradient = torch.randn_like(base_tile) * 0.1
            delta -= learning_rate * gradient
        
        return delta
    
    async def submit_delta(self, delta: torch.Tensor) -> str:
        """
        Submit delta for aggregation
        """
        # Create delta submission
        submission = {
            'node_id': self.node_id,
            'epoch_id': self.current_epoch.epoch_id,
            'delta': delta,
            'timestamp': time.time(),
            'signature': self.sign_delta(delta)
        }
        
        self.pending_deltas.append(submission)
        
        # Broadcast to peers
        await self.broadcast_delta(submission)
        
        return submission['signature']
    
    async def aggregate_deltas(self, min_participants: int = 3) -> torch.Tensor:
        """
        Aggregate deltas with Byzantine fault tolerance
        """
        # Wait for minimum participants
        timeout = 30  # seconds
        start_time = time.time()
        
        while len(self.pending_deltas) < min_participants:
            if time.time() - start_time > timeout:
                print(f"⚠️ Timeout: only {len(self.pending_deltas)} deltas received")
                break
            await asyncio.sleep(1)
        
        if len(self.pending_deltas) < min_participants:
            raise Exception("Insufficient participants for aggregation")
        
        # Byzantine-robust aggregation (e.g., Krum, trimmed mean)
        deltas = [d['delta'] for d in self.pending_deltas]
        
        # Simple averaging for now (should use Krum or similar)
        aggregated = torch.stack(deltas).mean(dim=0)
        
        # Verify aggregation doesn't diverge too much
        max_norm = 10.0
        if aggregated.norm() > max_norm:
            aggregated = aggregated * (max_norm / aggregated.norm())
        
        return aggregated
    
    async def commit_epoch(self, aggregated_delta: torch.Tensor) -> str:
        """
        Commit aggregated delta to blockchain
        """
        if not self.current_epoch:
            raise Exception("No active epoch")
        
        # Create consensus block
        block_data = {
            'epoch_id': self.current_epoch.epoch_id,
            'base_version': self.current_epoch.base_version,
            'participants': list(set(d['node_id'] for d in self.pending_deltas)),
            'aggregation_method': 'federated_averaging',
            'delta_hash': hashlib.sha256(aggregated_delta.numpy().tobytes()).hexdigest(),
            'consensus_ratio': len(self.pending_deltas) / len(self.peer_nodes)
        }
        
        # Commit to blockchain
        block_hash = self.blockchain.create_consensus_delta_block(
            aggregated_delta,
            block_data
        )
        
        # Mark epoch complete
        self.current_epoch.end_time = time.time()
        self.current_epoch.aggregated_delta = aggregated_delta
        self.current_epoch.consensus_achieved = True
        self.epoch_history.append(self.current_epoch)
        
        # Clear state
        self.current_epoch = None
        self.pending_deltas.clear()
        self.base_tile_cache.clear()
        
        print(f"📦 Epoch committed: {block_hash[:8]}")
        return block_hash
    
    async def run_learning_round(self, tile_id: str, dataset_batch: Any):
        """
        Run a complete learning round with consensus
        """
        try:
            # 1. Start epoch with consensus
            epoch = await self.start_epoch(tile_id, dataset_batch)
            
            # 2. Local training
            print(f"🏃 Training locally...")
            delta = await self.train_local(dataset_batch)
            
            # 3. Submit delta
            await self.submit_delta(delta)
            
            # 4. Wait and aggregate
            print(f"⏳ Waiting for other nodes...")
            aggregated = await self.aggregate_deltas()
            
            # 5. Commit to blockchain
            block_hash = await self.commit_epoch(aggregated)
            
            print(f"✅ Round complete: {block_hash[:8]}")
            return block_hash
            
        except Exception as e:
            print(f"❌ Learning round failed: {e}")
            raise
    
    # Helper methods
    async def query_peer_version(self, peer_id: str, tile_id: str) -> Optional[str]:
        """Query peer for their tile version"""
        # Mock implementation - should use P2P network
        return self.blockchain.get_latest_tile_version(tile_id)
    
    async def broadcast_delta(self, submission: Dict):
        """Broadcast delta to peers"""
        # Mock implementation - should use P2P network
        pass
    
    async def load_base_tile(self, tile_id: str, version: str) -> torch.Tensor:
        """Load specific tile version from blockchain"""
        # Mock implementation
        return torch.randn(256, 768)  # Example shape
    
    def sign_delta(self, delta: torch.Tensor) -> str:
        """Sign delta for authenticity"""
        return hashlib.sha256(
            f"{self.node_id}_{delta.sum().item()}".encode()
        ).hexdigest()[:16]


# Example usage
async def demo_consensus_learning():
    """Demonstrate consensus-based learning"""
    
    # Mock blockchain manager
    class MockBlockchain:
        def get_latest_tile_version(self, tile_id):
            return "v1_abc123"
        
        def create_consensus_delta_block(self, delta, metadata):
            return hashlib.sha256(str(metadata).encode()).hexdigest()
    
    # Create coordinator
    coordinator = ConsensusLearningCoordinator(
        node_id="node_1",
        blockchain_manager=MockBlockchain()
    )
    
    # Add mock peers
    coordinator.peer_nodes = {
        "node_2": {"last_seen": time.time(), "version": "v1_abc123"},
        "node_3": {"last_seen": time.time(), "version": "v1_abc123"},
    }
    
    # Run learning round
    await coordinator.run_learning_round(
        tile_id="layer0.expert0.weight",
        dataset_batch=None
    )

if __name__ == "__main__":
    asyncio.run(demo_consensus_learning())