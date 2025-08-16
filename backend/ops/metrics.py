"""Prometheus metrics for P2P network monitoring"""
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
from typing import Dict, Any


# Create registry
registry = CollectorRegistry()

# Network metrics
peer_count = Gauge(
    'blyan_p2p_peers_total',
    'Total number of connected peers',
    ['role'],
    registry=registry
)

peer_connections = Counter(
    'blyan_p2p_connections_total',
    'Total peer connections',
    ['direction', 'status'],
    registry=registry
)

bytes_transferred = Counter(
    'blyan_p2p_bytes_total',
    'Total bytes transferred',
    ['direction', 'message_type'],
    registry=registry
)

message_count = Counter(
    'blyan_p2p_messages_total',
    'Total messages processed',
    ['type', 'status'],
    registry=registry
)

# Chain metrics
chain_height = Gauge(
    'blyan_chain_height',
    'Current blockchain height',
    ['chain'],
    registry=registry
)

chain_tips = Gauge(
    'blyan_chain_tips_total',
    'Number of chain tips (forks)',
    registry=registry
)

blocks_processed = Counter(
    'blyan_blocks_total',
    'Total blocks processed',
    ['status'],
    registry=registry
)

block_processing_time = Histogram(
    'blyan_block_processing_seconds',
    'Time to process blocks',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

reorg_count = Counter(
    'blyan_reorg_total',
    'Total chain reorganizations',
    registry=registry
)

reorg_depth = Histogram(
    'blyan_reorg_depth',
    'Depth of chain reorganizations',
    buckets=[1, 2, 5, 10, 20, 50, 100],
    registry=registry
)

# Sync metrics
sync_progress = Gauge(
    'blyan_sync_progress_ratio',
    'Sync progress (0-1)',
    registry=registry
)

sync_speed = Gauge(
    'blyan_sync_blocks_per_second',
    'Current sync speed',
    registry=registry
)

# DHT metrics
dht_nodes = Gauge(
    'blyan_dht_nodes_total',
    'Total nodes in DHT routing table',
    ['bucket'],
    registry=registry
)

dht_lookups = Counter(
    'blyan_dht_lookups_total',
    'Total DHT lookups',
    ['status'],
    registry=registry
)

dht_store_operations = Counter(
    'blyan_dht_store_total',
    'Total DHT store operations',
    ['status'],
    registry=registry
)

# Security metrics
banned_peers = Gauge(
    'blyan_banned_peers_total',
    'Total banned peers',
    ['reason'],
    registry=registry
)

rate_limit_hits = Counter(
    'blyan_rate_limit_hits_total',
    'Rate limit violations',
    ['peer'],
    registry=registry
)

invalid_messages = Counter(
    'blyan_invalid_messages_total',
    'Invalid messages received',
    ['reason'],
    registry=registry
)

# Gossip metrics
gossip_messages = Counter(
    'blyan_gossip_messages_total',
    'Gossip messages',
    ['type', 'status'],
    registry=registry
)

gossip_queue_size = Gauge(
    'blyan_gossip_queue_size',
    'Current gossip queue size',
    registry=registry
)

# Performance metrics
message_latency = Summary(
    'blyan_message_latency_seconds',
    'Message round-trip latency',
    ['message_type'],
    registry=registry
)

memory_usage = Gauge(
    'blyan_memory_usage_bytes',
    'Memory usage',
    ['component'],
    registry=registry
)


class MetricsCollector:
    """Collects and exposes metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_block_time = time.time()
        self.blocks_in_window = 0
        
    def update_peer_count(self, full: int = 0, light: int = 0, seed: int = 0):
        """Update peer count metrics"""
        peer_count.labels(role='full').set(full)
        peer_count.labels(role='light').set(light)
        peer_count.labels(role='seed').set(seed)
    
    def record_connection(self, direction: str, success: bool):
        """Record connection attempt"""
        status = 'success' if success else 'failed'
        peer_connections.labels(direction=direction, status=status).inc()
    
    def record_bytes(self, direction: str, msg_type: str, size: int):
        """Record bytes transferred"""
        bytes_transferred.labels(direction=direction, message_type=msg_type).inc(size)
    
    def record_message(self, msg_type: str, success: bool):
        """Record message processing"""
        status = 'success' if success else 'failed'
        message_count.labels(type=msg_type, status=status).inc()
    
    def update_chain_height(self, height: int, chain: str = 'main'):
        """Update blockchain height"""
        chain_height.labels(chain=chain).set(height)
    
    def record_block(self, success: bool, processing_time: float):
        """Record block processing"""
        status = 'success' if success else 'failed'
        blocks_processed.labels(status=status).inc()
        
        if success:
            block_processing_time.observe(processing_time)
            
            # Update sync speed
            self.blocks_in_window += 1
            window = time.time() - self.last_block_time
            if window > 10:  # 10 second window
                speed = self.blocks_in_window / window
                sync_speed.set(speed)
                self.blocks_in_window = 0
                self.last_block_time = time.time()
    
    def record_reorg(self, depth: int):
        """Record chain reorganization"""
        reorg_count.inc()
        reorg_depth.observe(depth)
    
    def update_sync_progress(self, current: int, target: int):
        """Update sync progress"""
        if target > 0:
            progress = min(1.0, current / target)
            sync_progress.set(progress)
    
    def record_dht_lookup(self, success: bool):
        """Record DHT lookup"""
        status = 'success' if success else 'failed'
        dht_lookups.labels(status=status).inc()
    
    def record_ban(self, reason: str):
        """Record peer ban"""
        banned_peers.labels(reason=reason).inc()
    
    def record_rate_limit(self, peer_id: str = 'unknown'):
        """Record rate limit hit"""
        rate_limit_hits.labels(peer=peer_id[:8]).inc()
    
    def record_invalid_message(self, reason: str):
        """Record invalid message"""
        invalid_messages.labels(reason=reason).inc()
    
    def record_gossip(self, msg_type: str, success: bool):
        """Record gossip message"""
        status = 'sent' if success else 'dropped'
        gossip_messages.labels(type=msg_type, status=status).inc()
    
    def update_gossip_queue(self, size: int):
        """Update gossip queue size"""
        gossip_queue_size.set(size)
    
    def record_latency(self, msg_type: str, latency: float):
        """Record message latency"""
        message_latency.labels(message_type=msg_type).observe(latency)
    
    def update_memory(self, component: str, bytes_used: int):
        """Update memory usage"""
        memory_usage.labels(component=component).set(bytes_used)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics"""
        return generate_latest(registry)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats as dict"""
        return {
            'uptime': time.time() - self.start_time,
            'peers': {
                'full': peer_count.labels(role='full')._value.get(),
                'light': peer_count.labels(role='light')._value.get(),
                'seed': peer_count.labels(role='seed')._value.get(),
            },
            'chain': {
                'height': chain_height.labels(chain='main')._value.get(),
                'tips': chain_tips._value.get(),
            },
            'sync': {
                'progress': sync_progress._value.get(),
                'speed': sync_speed._value.get(),
            }
        }


# Global collector instance
metrics_collector = MetricsCollector()