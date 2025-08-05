"""
Tile Primary Ownership System for Blyan
Distributed tile ownership with stake-based election and fault tolerance
"""

import time
import asyncio
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import json
import hashlib
from pathlib import Path

class NodeStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    QUARANTINED = "quarantined"
    ELECTION_PENDING = "election_pending"

@dataclass
class NodeInfo:
    """Information about a node in the network"""
    node_id: str
    host: str
    port: int
    stake_amount: float
    avg_latency_ms: float
    last_heartbeat: float
    status: NodeStatus = NodeStatus.ACTIVE
    reputation_score: float = 1.0
    tiles_owned: Set[str] = field(default_factory=set)
    tiles_secondary: Set[str] = field(default_factory=set)
    region: str = "unknown"
    gpu_memory_gb: float = 0.0
    bandwidth_mbps: float = 0.0
    
    def get_election_score(self) -> float:
        """Calculate election score for primary selection"""
        # Score = (stake * reputation) / (latency * load_factor)
        load_factor = max(1.0, len(self.tiles_owned) / 10.0)  # Normalize load
        latency_penalty = max(1.0, self.avg_latency_ms / 50.0)  # Penalty for high latency
        
        base_score = (self.stake_amount * self.reputation_score) / (latency_penalty * load_factor)
        
        # Regional bonus (prefer local nodes)
        regional_bonus = 1.2 if self.region != "unknown" else 1.0
        
        return base_score * regional_bonus
    
    def is_healthy(self, current_time: float, heartbeat_timeout: float = 60.0) -> bool:
        """Check if node is healthy and responsive"""
        return (self.status == NodeStatus.ACTIVE and 
                (current_time - self.last_heartbeat) < heartbeat_timeout)

@dataclass
class TileOwnership:
    """Ownership information for a specific tile"""
    tile_id: str
    primary_node: str
    secondary_nodes: List[str] = field(default_factory=list)
    election_height: int = 0
    last_update: float = 0.0
    version: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'tile_id': self.tile_id,
            'primary_node': self.primary_node,
            'secondary_nodes': self.secondary_nodes,
            'election_height': self.election_height,
            'last_update': self.last_update,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TileOwnership':
        return cls(
            tile_id=data['tile_id'],
            primary_node=data['primary_node'],
            secondary_nodes=data.get('secondary_nodes', []),
            election_height=data.get('election_height', 0),
            last_update=data.get('last_update', 0.0),
            version=data.get('version', 1)
        )

class TileOwnershipRegistry:
    """
    Distributed registry for tile ownership management
    
    Features:
    - Stake-based primary election
    - Automatic failover on node failure
    - Load balancing across nodes
    - Regional awareness
    """
    
    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or Path("./data/tile_ownership.json")
        
        # Core data structures
        self.nodes: Dict[str, NodeInfo] = {}
        self.tile_ownership: Dict[str, TileOwnership] = {}
        self.election_history: List[Dict] = []
        
        # Configuration
        self.min_stake_amount = 100.0  # Minimum stake to participate
        self.heartbeat_timeout = 60.0  # Seconds
        self.election_cooldown = 30.0  # Seconds between elections
        self.max_tiles_per_node = 20  # Load balancing
        self.target_secondary_count = 2  # Redundancy
        
        # State tracking
        self.last_election_time: Dict[str, float] = {}
        self.pending_elections: Set[str] = set()
        
        # Load existing registry
        self._load_registry()
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node in the registry"""
        if node_info.stake_amount < self.min_stake_amount:
            raise ValueError(f"Insufficient stake: {node_info.stake_amount} < {self.min_stake_amount}")
        
        # Update heartbeat
        node_info.last_heartbeat = time.time()
        
        # Add to registry
        self.nodes[node_info.node_id] = node_info
        
        print(f"✅ Node {node_info.node_id} registered with stake {node_info.stake_amount}")
        
        # Trigger rebalancing if needed
        self._trigger_rebalancing()
        
        # Save registry
        self._save_registry()
        
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node and trigger elections for its tiles"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Trigger elections for owned tiles
        for tile_id in list(node.tiles_owned):
            self._trigger_election(tile_id, reason=f"Primary node {node_id} unregistered")
        
        # Remove from secondary assignments
        for tile_id in list(node.tiles_secondary):
            if tile_id in self.tile_ownership:
                ownership = self.tile_ownership[tile_id]
                if node_id in ownership.secondary_nodes:
                    ownership.secondary_nodes.remove(node_id)
                    ownership.last_update = time.time()
        
        # Remove node
        del self.nodes[node_id]
        
        print(f"❌ Node {node_id} unregistered")
        self._save_registry()
        
        return True
    
    def update_heartbeat(self, node_id: str, latency_ms: float = None) -> bool:
        """Update node heartbeat and latency"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        node.last_heartbeat = time.time()
        
        if latency_ms is not None:
            # Exponential moving average for latency
            alpha = 0.3
            node.avg_latency_ms = (alpha * latency_ms + (1 - alpha) * node.avg_latency_ms)
        
        return True
    
    def elect_primary(self, tile_id: str, candidates: Optional[List[str]] = None) -> Optional[str]:
        """Elect primary owner for a tile"""
        current_time = time.time()
        
        # Check election cooldown
        if (tile_id in self.last_election_time and 
            current_time - self.last_election_time[tile_id] < self.election_cooldown):
            return None
        
        # Get candidates
        if candidates is None:
            candidates = self._get_eligible_candidates(tile_id)
        
        if not candidates:
            print(f"⚠️ No eligible candidates for tile {tile_id}")
            return None
        
        # Calculate election scores
        candidate_scores = []
        for node_id in candidates:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.is_healthy(current_time, self.heartbeat_timeout):
                    score = node.get_election_score()
                    candidate_scores.append((node_id, score))
        
        if not candidate_scores:
            return None
        
        # Weighted random selection based on scores
        total_score = sum(score for _, score in candidate_scores)
        if total_score == 0:
            # Fallback to random selection
            winner = random.choice(candidate_scores)[0]
        else:
            # Weighted selection
            rand_val = random.uniform(0, total_score)
            cumulative = 0
            winner = candidate_scores[0][0]  # fallback
            
            for node_id, score in candidate_scores:
                cumulative += score
                if rand_val <= cumulative:
                    winner = node_id
                    break
        
        # Update ownership
        old_primary = None
        if tile_id in self.tile_ownership:
            old_primary = self.tile_ownership[tile_id].primary_node
            if old_primary in self.nodes:
                self.nodes[old_primary].tiles_owned.discard(tile_id)
        
        # Create new ownership
        self.tile_ownership[tile_id] = TileOwnership(
            tile_id=tile_id,
            primary_node=winner,
            election_height=len(self.election_history),
            last_update=current_time
        )
        
        # Update node assignments
        if winner in self.nodes:
            self.nodes[winner].tiles_owned.add(tile_id)
        
        # Assign secondary nodes
        self._assign_secondary_nodes(tile_id)
        
        # Record election
        self.election_history.append({
            'tile_id': tile_id,
            'winner': winner,
            'old_primary': old_primary,
            'candidates': candidates,
            'timestamp': current_time,
            'scores': dict(candidate_scores)
        })
        
        self.last_election_time[tile_id] = current_time
        self.pending_elections.discard(tile_id)
        
        print(f"🗳️ Elected {winner} as primary for tile {tile_id} (score: {dict(candidate_scores).get(winner, 0):.2f})")
        
        self._save_registry()
        return winner
    
    def _get_eligible_candidates(self, tile_id: str) -> List[str]:
        """Get eligible candidates for tile primary election"""
        current_time = time.time()
        candidates = []
        
        for node_id, node in self.nodes.items():
            # Check eligibility criteria
            if (node.is_healthy(current_time, self.heartbeat_timeout) and
                node.stake_amount >= self.min_stake_amount and
                len(node.tiles_owned) < self.max_tiles_per_node and
                node.status == NodeStatus.ACTIVE):
                candidates.append(node_id)
        
        return candidates
    
    def _assign_secondary_nodes(self, tile_id: str):
        """Assign secondary nodes for a tile"""
        if tile_id not in self.tile_ownership:
            return
        
        ownership = self.tile_ownership[tile_id]
        primary_node = ownership.primary_node
        
        # Get candidates (excluding primary)
        candidates = [node_id for node_id in self._get_eligible_candidates(tile_id) 
                     if node_id != primary_node]
        
        # Select top candidates by score
        candidate_scores = []
        current_time = time.time()
        
        for node_id in candidates:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.is_healthy(current_time, self.heartbeat_timeout):
                    score = node.get_election_score()
                    candidate_scores.append((node_id, score))
        
        # Sort by score and select top N
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        secondary_count = min(self.target_secondary_count, len(candidate_scores))
        
        # Update assignments
        for node_id in ownership.secondary_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].tiles_secondary.discard(tile_id)
        
        ownership.secondary_nodes = [node_id for node_id, _ in candidate_scores[:secondary_count]]
        
        for node_id in ownership.secondary_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].tiles_secondary.add(tile_id)
        
        ownership.last_update = time.time()
    
    def _trigger_election(self, tile_id: str, reason: str = ""):
        """Trigger election for a tile"""
        if tile_id not in self.pending_elections:
            self.pending_elections.add(tile_id)
            print(f"⚡ Election triggered for tile {tile_id}: {reason}")
    
    def _trigger_rebalancing(self):
        """Trigger load rebalancing across nodes"""
        # Find overloaded nodes
        overloaded_nodes = [
            node_id for node_id, node in self.nodes.items()
            if len(node.tiles_owned) > self.max_tiles_per_node
        ]
        
        # Trigger elections for some tiles from overloaded nodes
        for node_id in overloaded_nodes:
            node = self.nodes[node_id]
            excess_tiles = len(node.tiles_owned) - self.max_tiles_per_node
            
            # Select tiles to rebalance (prefer newer ones)
            tiles_to_rebalance = list(node.tiles_owned)[:excess_tiles]
            
            for tile_id in tiles_to_rebalance:
                self._trigger_election(tile_id, reason=f"Load balancing from {node_id}")
    
    def get_primary_node(self, tile_id: str) -> Optional[str]:
        """Get primary node for a tile"""
        if tile_id not in self.tile_ownership:
            return None
        
        ownership = self.tile_ownership[tile_id]
        primary_node = ownership.primary_node
        
        # Verify primary is still healthy
        if primary_node in self.nodes:
            current_time = time.time()
            node = self.nodes[primary_node]
            if node.is_healthy(current_time, self.heartbeat_timeout):
                return primary_node
        
        # Primary is unhealthy, trigger election
        self._trigger_election(tile_id, reason=f"Primary {primary_node} unhealthy")
        return None
    
    def get_secondary_nodes(self, tile_id: str) -> List[str]:
        """Get secondary nodes for a tile"""
        if tile_id not in self.tile_ownership:
            return []
        
        ownership = self.tile_ownership[tile_id]
        healthy_secondaries = []
        current_time = time.time()
        
        for node_id in ownership.secondary_nodes:
            if (node_id in self.nodes and 
                self.nodes[node_id].is_healthy(current_time, self.heartbeat_timeout)):
                healthy_secondaries.append(node_id)
        
        return healthy_secondaries
    
    def run_maintenance(self):
        """Run periodic maintenance tasks"""
        current_time = time.time()
        
        # Check for failed nodes
        failed_nodes = []
        for node_id, node in self.nodes.items():
            if not node.is_healthy(current_time, self.heartbeat_timeout):
                failed_nodes.append(node_id)
        
        # Handle failed nodes
        for node_id in failed_nodes:
            print(f"💀 Node {node_id} failed (last heartbeat: {current_time - self.nodes[node_id].last_heartbeat:.1f}s ago)")
            node = self.nodes[node_id]
            node.status = NodeStatus.INACTIVE
            
            # Trigger elections for owned tiles
            for tile_id in list(node.tiles_owned):
                self._trigger_election(tile_id, reason=f"Primary node {node_id} failed")
        
        # Process pending elections
        for tile_id in list(self.pending_elections):
            self.elect_primary(tile_id)
        
        # Rebalance if needed
        self._trigger_rebalancing()
        
        # Save state
        self._save_registry()
    
    def get_node_stats(self) -> Dict:
        """Get registry statistics"""
        current_time = time.time()
        active_nodes = sum(1 for node in self.nodes.values() 
                          if node.is_healthy(current_time, self.heartbeat_timeout))
        
        total_tiles = len(self.tile_ownership)
        avg_tiles_per_node = total_tiles / max(1, active_nodes)
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'total_tiles': total_tiles,
            'avg_tiles_per_node': avg_tiles_per_node,
            'pending_elections': len(self.pending_elections),
            'total_elections': len(self.election_history)
        }
    
    def _load_registry(self):
        """Load registry from disk"""
        if not self.registry_file.exists():
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            # Load nodes
            for node_data in data.get('nodes', []):
                node = NodeInfo(
                    node_id=node_data['node_id'],
                    host=node_data['host'],
                    port=node_data['port'],
                    stake_amount=node_data['stake_amount'],
                    avg_latency_ms=node_data.get('avg_latency_ms', 50.0),
                    last_heartbeat=node_data.get('last_heartbeat', 0.0),
                    status=NodeStatus(node_data.get('status', 'inactive')),
                    reputation_score=node_data.get('reputation_score', 1.0),
                    tiles_owned=set(node_data.get('tiles_owned', [])),
                    tiles_secondary=set(node_data.get('tiles_secondary', [])),
                    region=node_data.get('region', 'unknown'),
                    gpu_memory_gb=node_data.get('gpu_memory_gb', 0.0),
                    bandwidth_mbps=node_data.get('bandwidth_mbps', 0.0)
                )
                self.nodes[node.node_id] = node
            
            # Load tile ownership
            for ownership_data in data.get('tile_ownership', []):
                ownership = TileOwnership.from_dict(ownership_data)
                self.tile_ownership[ownership.tile_id] = ownership
            
            # Load election history
            self.election_history = data.get('election_history', [])
            
            print(f"📂 Loaded registry: {len(self.nodes)} nodes, {len(self.tile_ownership)} tiles")
            
        except Exception as e:
            print(f"⚠️ Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'nodes': [
                    {
                        'node_id': node.node_id,
                        'host': node.host,
                        'port': node.port,
                        'stake_amount': node.stake_amount,
                        'avg_latency_ms': node.avg_latency_ms,
                        'last_heartbeat': node.last_heartbeat,
                        'status': node.status.value,
                        'reputation_score': node.reputation_score,
                        'tiles_owned': list(node.tiles_owned),
                        'tiles_secondary': list(node.tiles_secondary),
                        'region': node.region,
                        'gpu_memory_gb': node.gpu_memory_gb,
                        'bandwidth_mbps': node.bandwidth_mbps
                    }
                    for node in self.nodes.values()
                ],
                'tile_ownership': [
                    ownership.to_dict() 
                    for ownership in self.tile_ownership.values()
                ],
                'election_history': self.election_history[-100:]  # Keep last 100 elections
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save registry: {e}")

# Export main classes
__all__ = ['TileOwnershipRegistry', 'NodeInfo', 'TileOwnership', 'NodeStatus']