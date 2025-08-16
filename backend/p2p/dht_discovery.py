"""Kademlia-like DHT for peer discovery"""
import asyncio
import hashlib
import time
import random
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class DHTNode:
    """Node in the DHT"""
    node_id: bytes  # 160-bit node ID
    address: str
    port: int
    last_seen: float
    role: str = "FULL"
    
    def distance(self, other_id: bytes) -> int:
        """XOR distance metric"""
        return int.from_bytes(
            bytes(a ^ b for a, b in zip(self.node_id, other_id)), 
            'big'
        )
    
    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id.hex(),
            'address': self.address,
            'port': self.port,
            'role': self.role
        }


class KBucket:
    """K-bucket for storing nodes at a specific distance"""
    
    def __init__(self, k: int = 20):
        self.k = k
        self.nodes: List[DHTNode] = []
        self.replacement_cache: List[DHTNode] = []
        
    def add_node(self, node: DHTNode) -> bool:
        """Add node to bucket"""
        # Check if already exists
        for i, existing in enumerate(self.nodes):
            if existing.node_id == node.node_id:
                # Move to end (most recently seen)
                self.nodes.pop(i)
                self.nodes.append(node)
                return True
        
        # Add if space available
        if len(self.nodes) < self.k:
            self.nodes.append(node)
            return True
        
        # Add to replacement cache
        if len(self.replacement_cache) < self.k:
            self.replacement_cache.append(node)
        
        return False
    
    def remove_node(self, node_id: bytes):
        """Remove node from bucket"""
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        
        # Promote from replacement cache
        if self.replacement_cache and len(self.nodes) < self.k:
            self.nodes.append(self.replacement_cache.pop(0))
    
    def get_nodes(self, exclude: Set[bytes] = None) -> List[DHTNode]:
        """Get nodes excluding specified IDs"""
        if exclude:
            return [n for n in self.nodes if n.node_id not in exclude]
        return self.nodes.copy()


class DHT:
    """Kademlia-like Distributed Hash Table"""
    
    def __init__(self, node_id: bytes = None, bootstrap_nodes: List[str] = None):
        """
        Initialize DHT
        
        Args:
            node_id: This node's ID (160 bits)
            bootstrap_nodes: List of bootstrap node addresses
        """
        # Generate node ID if not provided
        if node_id is None:
            node_id = hashlib.sha1(str(random.random()).encode()).digest()
        
        self.node_id = node_id
        self.buckets: List[KBucket] = [KBucket() for _ in range(160)]
        self.data_store: Dict[bytes, bytes] = {}
        self.bootstrap_nodes = bootstrap_nodes or []
        self.alpha = 3  # Concurrency parameter
        self.refresh_interval = 3600  # Bucket refresh interval
        self.last_refresh: Dict[int, float] = defaultdict(float)
        
        # Persistence
        self.state_file = Path("data/dht_state.json")
        self.load_state()
    
    def get_bucket_index(self, node_id: bytes) -> int:
        """Get bucket index for node ID"""
        if node_id == self.node_id:
            return -1
        
        # Find first differing bit
        distance = self.distance(node_id)
        if distance == 0:
            return -1
        
        # Count leading zeros
        return 159 - distance.bit_length() + 1
    
    def distance(self, node_id: bytes) -> int:
        """Calculate XOR distance"""
        return int.from_bytes(
            bytes(a ^ b for a, b in zip(self.node_id, node_id)),
            'big'
        )
    
    def add_node(self, node: DHTNode):
        """Add node to routing table"""
        bucket_idx = self.get_bucket_index(node.node_id)
        if bucket_idx >= 0:
            self.buckets[bucket_idx].add_node(node)
            self.save_state()
    
    def remove_node(self, node_id: bytes):
        """Remove node from routing table"""
        bucket_idx = self.get_bucket_index(node_id)
        if bucket_idx >= 0:
            self.buckets[bucket_idx].remove_node(node_id)
    
    async def find_node(self, target_id: bytes, k: int = 20) -> List[DHTNode]:
        """
        Find k closest nodes to target ID
        
        Args:
            target_id: Target node ID
            k: Number of nodes to find
            
        Returns:
            List of k closest nodes
        """
        # Get initial candidates from local buckets
        candidates = self.get_closest_nodes(target_id, k * 2)
        
        # Keep track of queried nodes
        queried = {self.node_id}
        closest = []
        
        while candidates:
            # Query alpha nodes in parallel
            batch = candidates[:self.alpha]
            candidates = candidates[self.alpha:]
            
            tasks = []
            for node in batch:
                if node.node_id not in queried:
                    queried.add(node.node_id)
                    tasks.append(self.query_find_node(node, target_id))
            
            # Wait for responses
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, list):
                    for node in result:
                        if node.node_id not in queried:
                            candidates.append(node)
            
            # Update closest
            all_nodes = closest + candidates
            all_nodes.sort(key=lambda n: self.distance(n.node_id))
            closest = all_nodes[:k]
            
            # Check if we're done
            if len(closest) >= k or not candidates:
                break
        
        return closest[:k]
    
    async def query_find_node(self, node: DHTNode, target_id: bytes) -> List[DHTNode]:
        """
        Query a node for closest nodes to target
        
        Args:
            node: Node to query
            target_id: Target ID
            
        Returns:
            List of nodes from response
        """
        # This would make actual network request
        # For now, return empty (to be implemented with network layer)
        return []
    
    def get_closest_nodes(self, target_id: bytes, k: int = 20) -> List[DHTNode]:
        """
        Get k closest nodes from local routing table
        
        Args:
            target_id: Target node ID
            k: Number of nodes
            
        Returns:
            List of closest nodes
        """
        all_nodes = []
        
        for bucket in self.buckets:
            all_nodes.extend(bucket.get_nodes())
        
        # Sort by distance to target
        all_nodes.sort(key=lambda n: 
                      int.from_bytes(
                          bytes(a ^ b for a, b in zip(n.node_id, target_id)),
                          'big'
                      ))
        
        return all_nodes[:k]
    
    async def store(self, key: bytes, value: bytes) -> bool:
        """
        Store key-value pair in DHT
        
        Args:
            key: Key (160 bits)
            value: Value to store
            
        Returns:
            Success status
        """
        # Find nodes closest to key
        nodes = await self.find_node(key)
        
        # Store at k closest nodes
        tasks = []
        for node in nodes:
            tasks.append(self.store_at_node(node, key, value))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes
        successes = sum(1 for r in results if r is True)
        
        return successes > len(nodes) // 2
    
    async def store_at_node(self, node: DHTNode, key: bytes, value: bytes) -> bool:
        """Store key-value at specific node"""
        # This would make actual network request
        # For now, store locally if it's us
        if node.node_id == self.node_id:
            self.data_store[key] = value
            return True
        return False
    
    async def get(self, key: bytes) -> Optional[bytes]:
        """
        Retrieve value from DHT
        
        Args:
            key: Key to lookup
            
        Returns:
            Value if found
        """
        # Check local store first
        if key in self.data_store:
            return self.data_store[key]
        
        # Find nodes storing key
        nodes = await self.find_node(key)
        
        # Query nodes for value
        for node in nodes:
            value = await self.get_from_node(node, key)
            if value is not None:
                return value
        
        return None
    
    async def get_from_node(self, node: DHTNode, key: bytes) -> Optional[bytes]:
        """Get value from specific node"""
        # This would make actual network request
        if node.node_id == self.node_id:
            return self.data_store.get(key)
        return None
    
    async def bootstrap(self):
        """Bootstrap by connecting to known nodes"""
        for addr in self.bootstrap_nodes:
            try:
                # Parse address
                host, port = addr.rsplit(':', 1)
                
                # Create bootstrap node (ID unknown)
                bootstrap = DHTNode(
                    node_id=b'\x00' * 20,  # Placeholder
                    address=host,
                    port=int(port),
                    last_seen=time.time()
                )
                
                # Query for our own ID to populate routing table
                nodes = await self.query_find_node(bootstrap, self.node_id)
                
                for node in nodes:
                    self.add_node(node)
                    
            except Exception as e:
                print(f"Failed to bootstrap from {addr}: {e}")
    
    async def refresh_buckets(self):
        """Refresh stale buckets"""
        now = time.time()
        
        for i, bucket in enumerate(self.buckets):
            # Check if bucket needs refresh
            if now - self.last_refresh[i] > self.refresh_interval:
                # Generate random ID in bucket range
                random_id = self.random_id_in_bucket(i)
                
                # Perform lookup to refresh
                await self.find_node(random_id)
                
                self.last_refresh[i] = now
    
    def random_id_in_bucket(self, bucket_idx: int) -> bytes:
        """Generate random ID that would go in specified bucket"""
        # Generate ID with specific distance
        distance = 2 ** (159 - bucket_idx)
        
        # XOR with our ID to get target
        random_bytes = random.randbytes(20)
        target = int.from_bytes(self.node_id, 'big') ^ distance
        
        return target.to_bytes(20, 'big')
    
    def save_state(self):
        """Save DHT state to disk"""
        state = {
            'node_id': self.node_id.hex(),
            'nodes': []
        }
        
        for bucket in self.buckets:
            for node in bucket.nodes:
                state['nodes'].append(node.to_dict())
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def load_state(self):
        """Load DHT state from disk"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            for node_data in state.get('nodes', []):
                node = DHTNode(
                    node_id=bytes.fromhex(node_data['node_id']),
                    address=node_data['address'],
                    port=node_data['port'],
                    last_seen=time.time(),
                    role=node_data.get('role', 'FULL')
                )
                self.add_node(node)
                
        except Exception as e:
            print(f"Failed to load DHT state: {e}")