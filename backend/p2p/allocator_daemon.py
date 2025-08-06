"""
High-Availability Allocator Daemon
Central coordinator with Redis Stream and P2P Gossip fallback
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, will use Gossip-only mode")

logger = logging.getLogger(__name__)

class AllocationStatus(Enum):
    """Status of expert allocation"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    FAILED = "failed"
    REBALANCING = "rebalancing"

@dataclass
class AllocationRequest:
    """Expert allocation request"""
    request_id: str
    node_id: str
    requested_experts: List[str]
    gpu_tier: int
    timestamp: float
    status: AllocationStatus = AllocationStatus.PENDING
    assigned_experts: List[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict):
        data['status'] = AllocationStatus(data['status'])
        return cls(**data)

@dataclass
class AllocationState:
    """Current allocation state for gossip"""
    version: int
    timestamp: float
    allocations: Dict[str, List[str]]  # node_id -> experts
    hash: str
    
    def calculate_hash(self) -> str:
        """Calculate hash of allocation state"""
        data = json.dumps(self.allocations, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class AllocatorDaemon:
    """High-availability allocator with failover"""
    
    def __init__(
        self,
        node_id: str,
        redis_url: Optional[str] = None,
        is_primary: bool = False
    ):
        self.node_id = node_id
        self.is_primary = is_primary
        self.redis_url = redis_url
        self.redis_client = None
        
        # Allocation state
        self.current_allocations: Dict[str, List[str]] = {}
        self.pending_requests: Dict[str, AllocationRequest] = {}
        self.allocation_version = 0
        
        # Gossip state
        self.peer_states: Dict[str, AllocationState] = {}
        self.peers: Set[str] = set()
        
        # Redis streams
        self.request_stream = "allocation:requests"
        self.response_stream = "allocation:responses"
        self.state_stream = "allocation:state"
    
    async def connect_redis(self) -> bool:
        """Connect to Redis if available"""
        if not REDIS_AVAILABLE or not self.redis_url:
            logger.info("Running in Gossip-only mode (no Redis)")
            return False
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to Gossip")
            self.redis_client = None
            return False
    
    async def request_allocation(
        self,
        node_id: str,
        requested_experts: List[str],
        gpu_tier: int
    ) -> str:
        """Submit allocation request"""
        request = AllocationRequest(
            request_id=f"{node_id}_{int(time.time()*1000)}",
            node_id=node_id,
            requested_experts=requested_experts,
            gpu_tier=gpu_tier,
            timestamp=time.time()
        )
        
        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.xadd(
                    self.request_stream,
                    request.to_dict()
                )
                logger.info(f"Allocation request {request.request_id} sent via Redis")
                return request.request_id
            except Exception as e:
                logger.warning(f"Redis write failed: {e}, using Gossip")
        
        # Fallback to gossip
        self.pending_requests[request.request_id] = request
        await self.gossip_request(request)
        return request.request_id
    
    async def process_allocation_requests(self):
        """Process allocation requests (primary coordinator only)"""
        if not self.is_primary:
            return
        
        while True:
            try:
                # Try Redis stream first
                if self.redis_client:
                    messages = await self.redis_client.xread(
                        {self.request_stream: '$'},
                        block=1000
                    )
                    
                    for stream, stream_messages in messages:
                        for msg_id, data in stream_messages:
                            request = AllocationRequest.from_dict(data)
                            await self.handle_allocation_request(request)
                            
                            # Acknowledge
                            await self.redis_client.xack(
                                self.request_stream,
                                "allocator-group",
                                msg_id
                            )
                else:
                    # Process gossip requests
                    for request_id, request in list(self.pending_requests.items()):
                        if request.status == AllocationStatus.PENDING:
                            await self.handle_allocation_request(request)
                            del self.pending_requests[request_id]
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing requests: {e}")
                await asyncio.sleep(1)
    
    async def handle_allocation_request(self, request: AllocationRequest):
        """Handle a single allocation request"""
        logger.info(f"Processing allocation request {request.request_id}")
        
        # Simple allocation logic (can be enhanced)
        available_experts = self.get_available_experts()
        assigned = []
        
        for expert in request.requested_experts:
            if expert in available_experts:
                assigned.append(expert)
                # Update allocation state
                if request.node_id not in self.current_allocations:
                    self.current_allocations[request.node_id] = []
                self.current_allocations[request.node_id].append(expert)
        
        # Update request status
        request.status = AllocationStatus.ASSIGNED
        request.assigned_experts = assigned
        
        # Publish response
        await self.publish_allocation_response(request)
        
        # Update global state
        await self.publish_allocation_state()
    
    async def publish_allocation_response(self, request: AllocationRequest):
        """Publish allocation response"""
        if self.redis_client:
            try:
                await self.redis_client.xadd(
                    self.response_stream,
                    request.to_dict()
                )
                return
            except Exception as e:
                logger.warning(f"Redis publish failed: {e}")
        
        # Gossip the response
        await self.gossip_response(request)
    
    async def publish_allocation_state(self):
        """Publish current allocation state"""
        self.allocation_version += 1
        
        state = AllocationState(
            version=self.allocation_version,
            timestamp=time.time(),
            allocations=self.current_allocations,
            hash=""
        )
        state.hash = state.calculate_hash()
        
        if self.redis_client:
            try:
                await self.redis_client.set(
                    f"{self.state_stream}:latest",
                    json.dumps(asdict(state)),
                    ex=300  # 5 minute TTL
                )
                logger.debug(f"Published state v{state.version} to Redis")
                return
            except Exception as e:
                logger.warning(f"Redis state publish failed: {e}")
        
        # Gossip the state
        await self.gossip_state(state)
    
    def get_available_experts(self) -> Set[str]:
        """Get list of available experts not yet allocated"""
        all_experts = set(f"layer{i}.expert{j}" for i in range(4) for j in range(8))
        allocated = set()
        
        for experts in self.current_allocations.values():
            allocated.update(experts)
        
        return all_experts - allocated
    
    # Gossip Protocol Implementation
    
    async def gossip_request(self, request: AllocationRequest):
        """Gossip allocation request to peers"""
        message = {
            'type': 'allocation_request',
            'data': request.to_dict(),
            'sender': self.node_id,
            'ttl': 3
        }
        await self.broadcast_to_peers(message)
    
    async def gossip_response(self, request: AllocationRequest):
        """Gossip allocation response to peers"""
        message = {
            'type': 'allocation_response',
            'data': request.to_dict(),
            'sender': self.node_id,
            'ttl': 3
        }
        await self.broadcast_to_peers(message)
    
    async def gossip_state(self, state: AllocationState):
        """Gossip allocation state to peers"""
        message = {
            'type': 'allocation_state',
            'data': asdict(state),
            'sender': self.node_id,
            'ttl': 3
        }
        await self.broadcast_to_peers(message)
    
    async def broadcast_to_peers(self, message: dict):
        """Broadcast message to random subset of peers"""
        if not self.peers:
            return
        
        # Select random subset (fanout = 3)
        fanout = min(3, len(self.peers))
        selected_peers = random.sample(list(self.peers), fanout)
        
        for peer_id in selected_peers:
            await self.send_to_peer(peer_id, message)
    
    async def send_to_peer(self, peer_id: str, message: dict):
        """Send message to specific peer (stub - implement with actual networking)"""
        # In production, this would use actual P2P networking
        logger.debug(f"Gossip to {peer_id}: {message['type']}")
    
    async def handle_gossip_message(self, message: dict):
        """Handle incoming gossip message"""
        msg_type = message.get('type')
        ttl = message.get('ttl', 0)
        
        if ttl <= 0:
            return  # Message expired
        
        # Decrement TTL and forward
        message['ttl'] = ttl - 1
        
        if msg_type == 'allocation_request':
            request = AllocationRequest.from_dict(message['data'])
            if self.is_primary:
                await self.handle_allocation_request(request)
            else:
                # Forward to other peers
                await self.broadcast_to_peers(message)
        
        elif msg_type == 'allocation_state':
            state = AllocationState(**message['data'])
            self.handle_state_update(state)
        
        elif msg_type == 'allocation_response':
            # Handle response
            response = AllocationRequest.from_dict(message['data'])
            logger.info(f"Received allocation response for {response.request_id}")
    
    def handle_state_update(self, state: AllocationState):
        """Handle allocation state update from peer"""
        sender_id = state.hash[:8]  # Use hash prefix as sender ID
        
        # Check if state is newer
        if sender_id not in self.peer_states or \
           state.version > self.peer_states[sender_id].version:
            
            self.peer_states[sender_id] = state
            logger.info(f"Updated peer state from {sender_id} (v{state.version})")
            
            # If we're not primary and have no Redis, adopt this state
            if not self.is_primary and not self.redis_client:
                if state.version > self.allocation_version:
                    self.current_allocations = state.allocations.copy()
                    self.allocation_version = state.version
                    logger.info(f"Adopted allocation state v{state.version}")
    
    async def monitor_primary_health(self):
        """Monitor primary coordinator health and take over if needed"""
        if self.is_primary:
            return
        
        last_state_update = time.time()
        
        while True:
            try:
                # Check Redis for primary heartbeat
                if self.redis_client:
                    heartbeat = await self.redis_client.get("allocator:primary:heartbeat")
                    if heartbeat:
                        last_heartbeat = float(heartbeat)
                        if time.time() - last_heartbeat < 30:
                            # Primary is alive
                            await asyncio.sleep(10)
                            continue
                
                # Check gossip state updates
                latest_state_time = max(
                    (s.timestamp for s in self.peer_states.values()),
                    default=0
                )
                
                if time.time() - latest_state_time > 60:
                    # No state updates for 60s, consider taking over
                    logger.warning("Primary coordinator seems down, attempting takeover")
                    await self.attempt_primary_takeover()
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            
            await asyncio.sleep(10)
    
    async def attempt_primary_takeover(self):
        """Attempt to become primary coordinator"""
        # Simple leader election using Redis SET NX or gossip consensus
        if self.redis_client:
            try:
                # Try to acquire primary lock
                acquired = await self.redis_client.set(
                    "allocator:primary:lock",
                    self.node_id,
                    nx=True,
                    ex=60
                )
                
                if acquired:
                    self.is_primary = True
                    logger.info(f"Node {self.node_id} became primary coordinator")
                    await self.process_allocation_requests()
            except Exception as e:
                logger.error(f"Takeover failed: {e}")
        else:
            # Gossip-based leader election (simplified)
            # In production, use proper consensus algorithm like Raft
            self.is_primary = True
            logger.info(f"Node {self.node_id} became primary (gossip mode)")
    
    async def run(self):
        """Run the allocator daemon"""
        # Connect to Redis if available
        await self.connect_redis()
        
        # Start background tasks
        tasks = []
        
        if self.is_primary:
            tasks.append(asyncio.create_task(self.process_allocation_requests()))
            tasks.append(asyncio.create_task(self.primary_heartbeat()))
        else:
            tasks.append(asyncio.create_task(self.monitor_primary_health()))
        
        # Wait for tasks
        await asyncio.gather(*tasks)
    
    async def primary_heartbeat(self):
        """Send heartbeat if primary"""
        while self.is_primary:
            if self.redis_client:
                try:
                    await self.redis_client.set(
                        "allocator:primary:heartbeat",
                        str(time.time()),
                        ex=30
                    )
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
            
            await asyncio.sleep(10)


# Example usage
if __name__ == "__main__":
    import sys
    
    node_id = sys.argv[1] if len(sys.argv) > 1 else "allocator-1"
    is_primary = "--primary" in sys.argv
    redis_url = "redis://localhost:6379" if "--redis" in sys.argv else None
    
    daemon = AllocatorDaemon(
        node_id=node_id,
        redis_url=redis_url,
        is_primary=is_primary
    )
    
    print(f"Starting allocator daemon: {node_id}")
    print(f"  Primary: {is_primary}")
    print(f"  Redis: {redis_url or 'Disabled (Gossip-only)'}")
    
    asyncio.run(daemon.run())