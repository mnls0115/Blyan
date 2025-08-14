#!/usr/bin/env python3
"""
Demo: Consensus-based Distributed Learning
Shows how nodes synchronize and train together
"""

import asyncio
import torch
import hashlib
import time
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.learning.consensus_learning import ConsensusLearningCoordinator

class SimulatedNode:
    """Simulated learning node"""
    
    def __init__(self, node_id: str, peers: list):
        self.node_id = node_id
        self.peers = peers
        self.coordinator = None
        self.tile_version = "v1_initial"
        
    async def initialize(self, blockchain):
        """Initialize the node"""
        self.coordinator = ConsensusLearningCoordinator(
            node_id=self.node_id,
            blockchain_manager=blockchain
        )
        
        # Add peers
        for peer_id in self.peers:
            if peer_id != self.node_id:
                self.coordinator.peer_nodes[peer_id] = {
                    "last_seen": time.time(),
                    "version": self.tile_version
                }
        
        print(f"🤖 Node {self.node_id} initialized with {len(self.coordinator.peer_nodes)} peers")
    
    async def participate_in_round(self, round_num: int):
        """Participate in a learning round"""
        print(f"\n{'='*60}")
        print(f"Node {self.node_id} - Round {round_num}")
        print(f"{'='*60}")
        
        try:
            # Simulate consensus phase
            print(f"📍 [SYNC] Agreeing on base version...")
            await asyncio.sleep(0.5)  # Simulate network delay
            
            # Simulate local training
            print(f"🏃 [TRAIN] Training on local data...")
            delta = torch.randn(256, 768) * 0.01  # Small delta
            await asyncio.sleep(1.0)  # Simulate training time
            
            # Simulate aggregation
            print(f"⏳ [AGGREGATE] Waiting for other nodes...")
            await asyncio.sleep(0.5)
            
            # Simulate commit
            print(f"📦 [COMMIT] Committing to blockchain...")
            block_hash = hashlib.sha256(
                f"{self.node_id}_round_{round_num}".encode()
            ).hexdigest()[:8]
            
            print(f"✅ Round {round_num} complete: block {block_hash}")
            
            # Update local version
            self.tile_version = f"v{round_num+1}_{block_hash}"
            
            return True
            
        except Exception as e:
            print(f"❌ Node {self.node_id} failed: {e}")
            return False

async def simulate_distributed_learning():
    """
    Simulate distributed learning with multiple nodes
    """
    print("\n" + "="*80)
    print("🚀 CONSENSUS-BASED DISTRIBUTED LEARNING DEMO")
    print("="*80)
    
    # Create mock blockchain
    class MockBlockchain:
        def __init__(self):
            self.blocks = []
            self.latest_version = "v1_initial"
        
        def get_latest_tile_version(self, tile_id):
            return self.latest_version
        
        def create_consensus_delta_block(self, delta, metadata):
            block_hash = hashlib.sha256(str(metadata).encode()).hexdigest()
            self.blocks.append({
                'hash': block_hash,
                'metadata': metadata,
                'timestamp': time.time()
            })
            self.latest_version = f"v{len(self.blocks)}_{block_hash[:8]}"
            return block_hash
    
    blockchain = MockBlockchain()
    
    # Create nodes
    node_ids = ["node_alpha", "node_beta", "node_gamma", "node_delta"]
    nodes = []
    
    for node_id in node_ids:
        node = SimulatedNode(node_id, node_ids)
        await node.initialize(blockchain)
        nodes.append(node)
    
    print(f"\n📊 Network initialized with {len(nodes)} nodes")
    
    # Run multiple rounds
    num_rounds = 3
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'#'*80}")
        print(f"🔄 ROUND {round_num}/{num_rounds}")
        print(f"{'#'*80}")
        
        # All nodes participate in parallel
        tasks = [node.participate_in_round(round_num) for node in nodes]
        results = await asyncio.gather(*tasks)
        
        # Check consensus
        successful = sum(results)
        print(f"\n📈 Round {round_num} Results:")
        print(f"  - Successful nodes: {successful}/{len(nodes)}")
        print(f"  - Consensus achieved: {'✅' if successful >= len(nodes)*2/3 else '❌'}")
        print(f"  - Latest version: {blockchain.latest_version}")
        
        # Small delay between rounds
        await asyncio.sleep(1)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"  - Total rounds: {num_rounds}")
    print(f"  - Total blocks: {len(blockchain.blocks)}")
    print(f"  - Final version: {blockchain.latest_version}")
    
    # Show convergence
    print(f"\n🎯 Convergence Check:")
    for node in nodes:
        print(f"  - {node.node_id}: {node.tile_version}")
    
    all_same = len(set(node.tile_version for node in nodes)) == 1
    print(f"\n{'✅ All nodes converged!' if all_same else '⚠️ Nodes diverged!'}")

if __name__ == "__main__":
    asyncio.run(simulate_distributed_learning())