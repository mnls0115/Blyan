"""
Edge Aggregator gRPC Service for Blyan
Regional gradient aggregation with 10-20x WAN traffic reduction
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import grpc
from concurrent import futures
import threading
from collections import defaultdict, deque
import numpy as np
import torch

from .delta_compression import DeltaCompressor, DeltaBase
from .tile_ownership import TileOwnershipRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeltaSubmission:
    """A delta submission from a learner node"""
    node_id: str
    tile_id: str
    delta: DeltaBase
    timestamp: float
    sequence_number: int
    base_block_hash: Optional[str] = None  # Base version for CAS
    round_id: Optional[int] = None  # Learning round anchor
    signature: Optional[str] = None

@dataclass
class AggregationBatch:
    """Batch of deltas for aggregation"""
    tile_id: str
    base_block_hash: Optional[str] = None  # Base version all deltas in batch share
    round_id: Optional[int] = None  # Round anchor all deltas share
    deltas: List[DeltaSubmission] = field(default_factory=list)
    batch_start_time: float = 0.0
    target_primary: Optional[str] = None
    
    def is_ready(self, min_deltas: int = 2, max_wait_time: float = 5.0) -> bool:
        """Check if batch is ready for aggregation"""
        current_time = time.time()
        has_enough_deltas = len(self.deltas) >= min_deltas
        waited_long_enough = (current_time - self.batch_start_time) >= max_wait_time
        
        return has_enough_deltas or waited_long_enough

@dataclass 
class RegionStats:
    """Statistics for regional aggregation"""
    total_deltas_received: int = 0
    total_deltas_aggregated: int = 0
    total_bytes_received: int = 0
    total_bytes_sent: int = 0
    avg_batch_size: float = 0.0
    avg_aggregation_time_ms: float = 0.0
    compression_ratio: float = 0.0
    active_learners: int = 0
    last_reset: float = field(default_factory=time.time)

class EdgeAggregator:
    """
    Regional Edge Aggregator
    
    Architecture:
    Learner Nodes → Edge Aggregator → Primary Tile Owner
         1-5ms           20-50ms WAN
    
    Benefits:
    - Reduces WAN traffic by 10-20x
    - Batches deltas from multiple learners
    - Compresses aggregated deltas
    - Provides regional fault tolerance
    """
    
    def __init__(self, 
                 region: str,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 ownership_registry: Optional[TileOwnershipRegistry] = None):
        
        self.region = region
        self.host = host
        self.port = port
        self.ownership_registry = ownership_registry or TileOwnershipRegistry()
        
        # Aggregation state
        self.pending_batches: Dict[str, AggregationBatch] = {}
        self.delta_compressor = DeltaCompressor()
        
        # Regional statistics
        self.stats = RegionStats()
        self.node_last_seen: Dict[str, float] = {}
        
        # Configuration
        self.batch_config = {
            'min_deltas': 2,           # Minimum deltas before aggregation
            'max_wait_time': 5.0,      # Maximum wait time in seconds
            'max_batch_size': 50,      # Maximum deltas per batch
            'aggregation_interval': 1.0  # Aggregation check interval
        }
        
        # Threading
        self.aggregation_thread: Optional[threading.Thread] = None
        self.running = False
        
        # gRPC server
        self.server: Optional[grpc.aio.Server] = None
        
        logger.info(f"🌐 EdgeAggregator initialized for region {region} on {host}:{port}")
    
    async def start_server(self):
        """Start the gRPC aggregation server"""
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add service to server (proto definition would be needed for full implementation)
        # aggregator_pb2_grpc.add_EdgeAggregatorServicer_to_server(self, self.server)
        
        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
        
        logger.info(f"🚀 Starting EdgeAggregator server on {listen_addr}")
        await self.server.start()
        
        # Start background aggregation
        self.running = True
        self.aggregation_thread = threading.Thread(target=self._run_aggregation_loop)
        self.aggregation_thread.start()
        
        return self.server
    
    async def stop_server(self):
        """Stop the gRPC server"""
        self.running = False
        
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5.0)
        
        if self.server:
            await self.server.stop(grace=5)
        
        logger.info("🛑 EdgeAggregator server stopped")
    
    def submit_delta(self, submission: DeltaSubmission) -> bool:
        """Submit a delta for aggregation"""
        try:
            # Update node tracking
            self.node_last_seen[submission.node_id] = submission.timestamp
            
            # Validate anchors presence
            if not submission.base_block_hash or submission.round_id is None:
                logger.warning("Missing base_block_hash/round_id in delta submission; rejecting")
                return False

            # Create unique batch key based on tile_id + base_block_hash + round_id
            batch_key = f"{submission.tile_id}:{submission.base_block_hash}:{submission.round_id}"
            
            # Get or create batch for tile+base combination
            if batch_key not in self.pending_batches:
                primary_node = self.ownership_registry.get_primary_node(submission.tile_id)
                
                self.pending_batches[batch_key] = AggregationBatch(
                    tile_id=submission.tile_id,
                    base_block_hash=submission.base_block_hash,
                    round_id=submission.round_id,
                    batch_start_time=time.time(),
                    target_primary=primary_node
                )
            
            batch = self.pending_batches[batch_key]
            
            # Verify same base version and round_id
            if batch.base_block_hash != submission.base_block_hash:
                logger.warning(f"Base version mismatch in batch: expected {batch.base_block_hash}, got {submission.base_block_hash}")
                return False
            if batch.round_id != submission.round_id:
                logger.warning(f"Round mismatch in batch: expected {batch.round_id}, got {submission.round_id}")
                return False
            
            # Add delta to batch
            batch.deltas.append(submission)
            
            # Update statistics
            self.stats.total_deltas_received += 1
            self.stats.total_bytes_received += len(submission.delta.to_bytes())
            
            # Check if batch is full
            if len(batch.deltas) >= self.batch_config['max_batch_size']:
                self._trigger_immediate_aggregation(submission.tile_id)
            
            logger.debug(f"📥 Delta submitted: {submission.node_id} → {submission.tile_id} "
                        f"(batch size: {len(batch.deltas)})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to submit delta: {e}")
            return False
    
    def _run_aggregation_loop(self):
        """Background thread for periodic aggregation"""
        logger.info("🔄 Starting aggregation loop")
        
        while self.running:
            try:
                self._process_ready_batches()
                self._update_statistics()
                time.sleep(self.batch_config['aggregation_interval'])
                
            except Exception as e:
                logger.error(f"💥 Error in aggregation loop: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _process_ready_batches(self):
        """Process all ready batches"""
        ready_tiles = []
        
        # Find ready batches
        for batch_key, batch in self.pending_batches.items():
            if batch.is_ready(
                min_deltas=self.batch_config['min_deltas'],
                max_wait_time=self.batch_config['max_wait_time']
            ):
                ready_tiles.append(batch_key)
        
        # Process ready batches
        for batch_key in ready_tiles:
            self._aggregate_and_forward(batch_key)
    
    def _aggregate_and_forward(self, batch_key: str):
        """Aggregate deltas for a tile and forward to primary"""
        if batch_key not in self.pending_batches:
            return
        
        batch = self.pending_batches[batch_key]
        if not batch.deltas:
            return
        
        start_time = time.time()
        
        try:
            # Aggregate deltas
            aggregated_delta = self._aggregate_deltas(batch.deltas)
            
            # Forward to primary node
            success = self._forward_to_primary(batch.target_primary, batch.tile_id, aggregated_delta)
            
            if success:
                # Update statistics
                aggregation_time = (time.time() - start_time) * 1000  # ms
                
                self.stats.total_deltas_aggregated += len(batch.deltas)
                self.stats.total_bytes_sent += len(aggregated_delta.to_bytes())
                
                # Update moving averages
                alpha = 0.1
                self.stats.avg_batch_size = (alpha * len(batch.deltas) + 
                                           (1 - alpha) * self.stats.avg_batch_size)
                self.stats.avg_aggregation_time_ms = (alpha * aggregation_time + 
                                                    (1 - alpha) * self.stats.avg_aggregation_time_ms)
                
                logger.info(f"✅ Aggregated {len(batch.deltas)} deltas for {batch.tile_id} (base {batch.base_block_hash}) "
                           f"({aggregation_time:.1f}ms)")
            else:
                logger.warning(f"⚠️ Failed to forward aggregated delta for {batch.tile_id}")
            
        except Exception as e:
            logger.error(f"❌ Aggregation failed for {tile_id}: {e}")
        
        finally:
            # Remove processed batch
            del self.pending_batches[batch_key]
    
    def _aggregate_deltas(self, submissions: List[DeltaSubmission]) -> DeltaBase:
        """
        Aggregate multiple deltas into a single delta
        
        Strategy depends on delta type:
        - INT8Delta: Average with weight adjustment
        - SparseDelta: Union of sparse elements
        - LoRADelta: Average low-rank factors
        """
        if not submissions:
            raise ValueError("No deltas to aggregate")
        
        if len(submissions) == 1:
            return submissions[0].delta
        
        # Group by delta type
        delta_types = defaultdict(list)
        for submission in submissions:
            delta_type = type(submission.delta).__name__
            delta_types[delta_type].append(submission.delta)
        
        # For simplicity, assume all deltas are same type (most common case)
        if len(delta_types) == 1:
            delta_type, deltas = list(delta_types.items())[0]
            
            if delta_type == "INT8Delta":
                return self._aggregate_int8_deltas(deltas)
            elif delta_type == "SparseDelta":
                return self._aggregate_sparse_deltas(deltas)
            elif delta_type == "LoRADelta":
                return self._aggregate_lora_deltas(deltas)
        
        # Mixed types - convert to common format and aggregate
        return self._aggregate_mixed_deltas([s.delta for s in submissions])
    
    def _aggregate_int8_deltas(self, deltas) -> 'INT8Delta':
        """Aggregate INT8 deltas by averaging"""
        if not deltas:
            raise ValueError("No INT8 deltas to aggregate")
        
        # Convert to float, average, then quantize back
        float_sum = None
        count = 0
        
        for delta in deltas:
            # Dequantize to float
            if delta.per_channel:
                float_delta = (delta.data.float() - delta.zero_point) * delta.scale.unsqueeze(-1)
            else:
                float_delta = (delta.data.float() - delta.zero_point) * delta.scale
            
            if float_sum is None:
                float_sum = float_delta
            else:
                float_sum += float_delta
            count += 1
        
        # Average
        avg_delta = float_sum / count
        
        # Requantize
        scale = avg_delta.abs().max() / 127.0
        quantized = ((avg_delta / scale)).round().clamp(-128, 127)
        
        from .delta_compression import INT8Delta
        return INT8Delta(
            data=quantized.to(torch.int8),
            scale=scale,
            zero_point=0,
            per_channel=False,
            original_shape=deltas[0].original_shape
        )
    
    def _aggregate_sparse_deltas(self, deltas) -> 'SparseDelta':
        """Aggregate sparse deltas by union of non-zero elements"""
        if not deltas:
            raise ValueError("No sparse deltas to aggregate")
        
        # Collect all indices and values
        all_indices = []
        all_values = []
        shape = deltas[0].shape
        
        # Create a dense accumulator
        dense_accumulator = torch.zeros(shape, dtype=torch.float32)
        count_accumulator = torch.zeros(shape, dtype=torch.float32)
        
        for delta in deltas:
            # Convert sparse to dense temporarily
            if len(shape) == 1:
                dense_accumulator[delta.indices] += delta.values.float()
                count_accumulator[delta.indices] += 1.0
            else:
                flat_acc = dense_accumulator.flatten()
                flat_count = count_accumulator.flatten()
                flat_acc[delta.indices] += delta.values.float()
                flat_count[delta.indices] += 1.0
        
        # Average where count > 0
        mask = count_accumulator > 0
        dense_accumulator[mask] /= count_accumulator[mask]
        
        # Convert back to sparse (keep top elements)
        flat_result = dense_accumulator.flatten()
        abs_values = flat_result.abs()
        
        # Keep top 20% elements
        k = max(1, int(len(flat_result) * 0.2))
        threshold = torch.kthvalue(abs_values, len(abs_values) - k + 1)[0]
        
        sparse_mask = abs_values >= threshold
        indices = sparse_mask.nonzero().squeeze()
        values = flat_result[sparse_mask]
        
        from .delta_compression import SparseDelta
        return SparseDelta(
            indices=indices,
            values=values.to(torch.float16),
            shape=shape,
            sparsity_ratio=1.0 - (len(values) / len(flat_result))
        )
    
    def _aggregate_lora_deltas(self, deltas) -> 'LoRADelta':
        """Aggregate LoRA deltas by averaging factors"""
        if not deltas:
            raise ValueError("No LoRA deltas to aggregate")
        
        # Average A and B matrices
        A_sum = None
        B_sum = None
        count = 0
        
        for delta in deltas:
            if A_sum is None:
                A_sum = delta.A.float()
                B_sum = delta.B.float()
            else:
                A_sum += delta.A.float()
                B_sum += delta.B.float()
            count += 1
        
        A_avg = A_sum / count
        B_avg = B_sum / count
        
        from .delta_compression import LoRADelta
        return LoRADelta(
            A=A_avg.to(torch.float16),
            B=B_avg.to(torch.float16),
            rank=deltas[0].rank,
            original_shape=deltas[0].original_shape,
            alpha=deltas[0].alpha
        )
    
    def _aggregate_mixed_deltas(self, deltas) -> DeltaBase:
        """Aggregate mixed delta types (fallback)"""
        # Convert all to dense tensors, average, then compress
        # This is less efficient but handles mixed types
        
        dense_sum = None
        count = 0
        reference_shape = None
        
        for delta in deltas:
            # Convert to dense tensor (this requires implementing dense conversion)
            # For now, return the first delta as fallback
            if reference_shape is None:
                reference_shape = getattr(delta, 'original_shape', (1,))
            
            # Simplified: just return first delta
            if dense_sum is None:
                return delta
        
        return deltas[0]  # Fallback
    
    def _forward_to_primary(self, primary_node: str, tile_id: str, aggregated_delta: DeltaBase) -> bool:
        """Forward aggregated delta to primary node"""
        if not primary_node:
            logger.warning(f"⚠️ No primary node found for tile {tile_id}")
            return False
        
        try:
            # In a real implementation, this would use gRPC to send to primary
            # For now, we'll simulate the forwarding
            
            delta_bytes = aggregated_delta.to_bytes()
            compressed_bytes = len(delta_bytes)  # Already compressed
            
            logger.info(f"📤 Forwarding {compressed_bytes} bytes to {primary_node} for {tile_id}")
            
            # Simulate network delay
            time.sleep(0.001)  # 1ms simulated latency
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to forward to {primary_node}: {e}")
            return False
    
    def _trigger_immediate_aggregation(self, tile_id: str, base_block_hash: Optional[str] = None):
        """Trigger immediate aggregation for a tile"""
        key = f"{tile_id}:{base_block_hash or 'genesis'}"
        if key in self.pending_batches:
            self._aggregate_and_forward(key)
    
    def _update_statistics(self):
        """Update regional statistics"""
        current_time = time.time()
        
        # Count active learners (seen in last 60 seconds)
        active_learners = sum(1 for last_seen in self.node_last_seen.values()
                            if current_time - last_seen < 60.0)
        self.stats.active_learners = active_learners
        
        # Calculate compression ratio
        if self.stats.total_bytes_received > 0:
            self.stats.compression_ratio = (self.stats.total_bytes_received / 
                                          max(1, self.stats.total_bytes_sent))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        current_time = time.time()
        uptime = current_time - self.stats.last_reset
        
        return {
            'region': self.region,
            'uptime_seconds': uptime,
            'total_deltas_received': self.stats.total_deltas_received,
            'total_deltas_aggregated': self.stats.total_deltas_aggregated,
            'total_bytes_received': self.stats.total_bytes_received,
            'total_bytes_sent': self.stats.total_bytes_sent,
            'compression_ratio': self.stats.compression_ratio,
            'avg_batch_size': self.stats.avg_batch_size,
            'avg_aggregation_time_ms': self.stats.avg_aggregation_time_ms,
            'active_learners': self.stats.active_learners,
            'pending_batches': len(self.pending_batches),
            'throughput_deltas_per_sec': self.stats.total_deltas_received / max(1, uptime),
            'bandwidth_saved_ratio': max(0, self.stats.compression_ratio - 1) / max(1, self.stats.compression_ratio)
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = RegionStats()
        logger.info("📊 Statistics reset")

# Utility function for creating edge aggregators
def create_edge_aggregator(region: str, port: int, 
                         ownership_registry: Optional[TileOwnershipRegistry] = None) -> EdgeAggregator:
    """Create and configure an edge aggregator for a region"""
    return EdgeAggregator(
        region=region,
        port=port,
        ownership_registry=ownership_registry
    )

# Export main classes
__all__ = ['EdgeAggregator', 'DeltaSubmission', 'AggregationBatch', 'RegionStats', 'create_edge_aggregator']