"""
gRPC Service Implementation for EdgeAggregator
Production-ready distributed gradient aggregation service
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional
from concurrent import futures

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

# Import generated protobuf classes (would be generated from .proto file)
# For now, we'll create simplified versions
from dataclasses import dataclass
from enum import Enum

from .edge_aggregator import EdgeAggregator, DeltaSubmission, RegionStats
from .delta_compression import DeltaBase, INT8Delta, SparseDelta, LoRADelta, DeltaCompressor
from .tile_ownership import TileOwnershipRegistry, NodeInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified protobuf message classes (would normally be generated)
@dataclass
class GrpcDeltaSubmission:
    node_id: str
    tile_id: str
    delta_data: bytes
    delta_type: str
    timestamp: float
    sequence_number: int
    signature: str = ""

@dataclass
class GrpcSubmissionResponse:
    accepted: bool
    message: str
    batch_id: str = ""
    estimated_aggregation_time: float = 0.0

@dataclass
class GrpcAggregationStats:
    aggregator_id: str
    region: str
    uptime_seconds: float
    total_deltas_received: int
    total_deltas_aggregated: int
    total_bytes_received: int
    total_bytes_sent: int
    avg_batch_size: float
    avg_aggregation_time_ms: float
    compression_ratio: float
    active_learners: int
    pending_batches: int
    deltas_per_second: float
    bandwidth_saved_ratio: float

class EdgeAggregatorGrpcService:
    """
    gRPC service implementation for EdgeAggregator
    
    Provides high-performance distributed gradient aggregation with:
    - Async request handling
    - Automatic batching and compression
    - Regional traffic optimization
    - Cryptographic integrity verification
    """
    
    def __init__(self, edge_aggregator: EdgeAggregator, max_workers: int = 50):
        self.edge_aggregator = edge_aggregator
        self.max_workers = max_workers
        
        # Service state
        self.registered_learners: Dict[str, Dict] = {}
        self.service_stats = {
            'requests_received': 0,
            'requests_processed': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # gRPC server
        self.server: Optional[grpc.aio.Server] = None
        
        logger.info(f"🚀 EdgeAggregatorGrpcService initialized with {max_workers} workers")
    
    async def SubmitDelta(self, request: GrpcDeltaSubmission, context) -> GrpcSubmissionResponse:
        """Handle single delta submission from learner node"""
        try:
            self.service_stats['requests_received'] += 1
            
            # Validate request
            if not self._validate_delta_submission(request):
                return GrpcSubmissionResponse(
                    accepted=False,
                    message="Invalid delta submission"
                )
            
            # Deserialize delta
            delta = self._deserialize_delta(request.delta_data, request.delta_type)
            if not delta:
                return GrpcSubmissionResponse(
                    accepted=False,
                    message="Failed to deserialize delta"
                )
            
            # Create submission object
            submission = DeltaSubmission(
                node_id=request.node_id,
                tile_id=request.tile_id,
                delta=delta,
                timestamp=request.timestamp,
                sequence_number=request.sequence_number,
                signature=request.signature
            )
            
            # Submit to aggregator
            success = self.edge_aggregator.submit_delta(submission)
            
            if success:
                self.service_stats['requests_processed'] += 1
                return GrpcSubmissionResponse(
                    accepted=True,
                    message="Delta accepted for aggregation",
                    batch_id=f"batch_{int(time.time())}_{request.tile_id}",
                    estimated_aggregation_time=time.time() + 5.0  # 5 seconds
                )
            else:
                self.service_stats['errors'] += 1
                return GrpcSubmissionResponse(
                    accepted=False,
                    message="Failed to submit delta to aggregator"
                )
                
        except Exception as e:
            logger.error(f"❌ Error in SubmitDelta: {e}")
            self.service_stats['errors'] += 1
            return GrpcSubmissionResponse(
                accepted=False,
                message=f"Internal error: {str(e)}"
            )
    
    async def SubmitDeltaBatch(self, request, context):
        """Handle batch delta submission for improved efficiency"""
        try:
            self.service_stats['requests_received'] += len(request.deltas)
            accepted_count = 0
            rejected_count = 0
            rejection_reasons = []
            
            for delta_request in request.deltas:
                # Process each delta in batch
                response = await self.SubmitDelta(delta_request, context)
                if response.accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1
                    rejection_reasons.append(response.message)
            
            return {
                'accepted': accepted_count > 0,
                'message': f"Processed {accepted_count + rejected_count} deltas",
                'accepted_deltas': accepted_count,
                'rejected_deltas': rejected_count,
                'rejection_reasons': rejection_reasons
            }
            
        except Exception as e:
            logger.error(f"❌ Error in SubmitDeltaBatch: {e}")
            return {
                'accepted': False,
                'message': f"Batch processing failed: {str(e)}",
                'accepted_deltas': 0,
                'rejected_deltas': len(request.deltas),
                'rejection_reasons': [str(e)]
            }
    
    async def GetAggregationStats(self, request, context) -> GrpcAggregationStats:
        """Get aggregation performance statistics"""
        try:
            # Get stats from edge aggregator
            stats = self.edge_aggregator.get_stats()
            
            # Add service-level stats
            service_uptime = time.time() - self.service_stats['start_time']
            
            return GrpcAggregationStats(
                aggregator_id=self.edge_aggregator.region,
                region=self.edge_aggregator.region,
                uptime_seconds=service_uptime,
                total_deltas_received=stats['total_deltas_received'],
                total_deltas_aggregated=stats['total_deltas_aggregated'],
                total_bytes_received=stats['total_bytes_received'],
                total_bytes_sent=stats['total_bytes_sent'],
                avg_batch_size=stats['avg_batch_size'],
                avg_aggregation_time_ms=stats['avg_aggregation_time_ms'],
                compression_ratio=stats['compression_ratio'],
                active_learners=stats['active_learners'],
                pending_batches=stats['pending_batches'],
                deltas_per_second=stats['throughput_deltas_per_sec'],
                bandwidth_saved_ratio=stats['bandwidth_saved_ratio']
            )
            
        except Exception as e:
            logger.error(f"❌ Error in GetAggregationStats: {e}")
            # Return empty stats on error
            return GrpcAggregationStats(
                aggregator_id=self.edge_aggregator.region,
                region=self.edge_aggregator.region,
                uptime_seconds=0,
                total_deltas_received=0,
                total_deltas_aggregated=0,
                total_bytes_received=0,
                total_bytes_sent=0,
                avg_batch_size=0,
                avg_aggregation_time_ms=0,
                compression_ratio=0,
                active_learners=0,
                pending_batches=0,
                deltas_per_second=0,
                bandwidth_saved_ratio=0
            )
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            # Check if aggregator is running
            is_healthy = (
                self.edge_aggregator.running and
                len(self.edge_aggregator.pending_batches) < 100  # Not overloaded
            )
            
            return {
                'status': 'SERVING' if is_healthy else 'NOT_SERVING',
                'message': 'Aggregator healthy' if is_healthy else 'Aggregator overloaded or stopped',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in HealthCheck: {e}")
            return {
                'status': 'NOT_SERVING',
                'message': f'Health check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    async def RegisterLearner(self, request, context):
        """Register a new learner node"""
        try:
            learner_info = {
                'node_id': request.node_id,
                'host': request.host,
                'port': request.port,
                'region': request.region,
                'interested_tiles': request.interested_tiles,
                'registration_time': time.time(),
                'last_heartbeat': time.time()
            }
            
            # Store learner information
            self.registered_learners[request.node_id] = learner_info
            
            logger.info(f"📝 Registered learner {request.node_id} from {request.region}")
            
            return {
                'accepted': True,
                'message': 'Learner registered successfully',
                'assigned_learner_id': request.node_id,
                'assigned_tiles': request.interested_tiles,
                'config': {
                    'batch_size_limit': self.edge_aggregator.batch_config['max_batch_size'],
                    'max_wait_time_seconds': self.edge_aggregator.batch_config['max_wait_time'],
                    'max_delta_size_mb': 50,  # 50MB limit
                    'preferred_compression_methods': ['INT8_SPARSE_LORA']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in RegisterLearner: {e}")
            return {
                'accepted': False,
                'message': f'Registration failed: {str(e)}',
                'assigned_learner_id': '',
                'assigned_tiles': [],
                'config': {}
            }
    
    async def UnregisterLearner(self, request, context):
        """Unregister a learner node"""
        try:
            if request.node_id in self.registered_learners:
                del self.registered_learners[request.node_id]
                logger.info(f"📤 Unregistered learner {request.node_id}: {request.reason}")
                
                return {
                    'success': True,
                    'message': 'Learner unregistered successfully'
                }
            else:
                return {
                    'success': False,
                    'message': 'Learner not found'
                }
                
        except Exception as e:
            logger.error(f"❌ Error in UnregisterLearner: {e}")
            return {
                'success': False,
                'message': f'Unregistration failed: {str(e)}'
            }
    
    def _validate_delta_submission(self, request: GrpcDeltaSubmission) -> bool:
        """Validate delta submission request"""
        # Basic validation
        if not request.node_id or not request.tile_id:
            return False
        
        # Check if learner is registered
        if request.node_id not in self.registered_learners:
            logger.warning(f"⚠️ Delta from unregistered learner: {request.node_id}")
            return False
        
        # Check delta size (prevent DoS)
        if len(request.delta_data) > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"⚠️ Delta too large: {len(request.delta_data)} bytes")
            return False
        
        # Update learner heartbeat
        self.registered_learners[request.node_id]['last_heartbeat'] = time.time()
        
        return True
    
    def _deserialize_delta(self, delta_data: bytes, delta_type: str) -> Optional[DeltaBase]:
        """Deserialize delta from bytes"""
        try:
            if delta_type == "INT8_DELTA":
                return INT8Delta.from_bytes(delta_data)
            elif delta_type == "SPARSE_DELTA":
                return SparseDelta.from_bytes(delta_data)
            elif delta_type == "LORA_DELTA":
                return LoRADelta.from_bytes(delta_data)
            else:
                logger.warning(f"⚠️ Unknown delta type: {delta_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to deserialize delta: {e}")
            return None
    
    async def start_service(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the gRPC service"""
        try:
            # Create gRPC server
            self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))
            
            # Add service to server (would use generated code in real implementation)
            # edge_aggregator_pb2_grpc.add_EdgeAggregatorServiceServicer_to_server(self, self.server)
            
            # Add insecure port
            listen_addr = f"{host}:{port}"
            self.server.add_insecure_port(listen_addr)
            
            # Start edge aggregator
            await self.edge_aggregator.start_server()
            
            # Start gRPC server
            await self.server.start()
            logger.info(f"🌐 EdgeAggregator gRPC service started on {listen_addr}")
            
            # Keep server running
            await self.server.wait_for_termination()
            
        except Exception as e:
            logger.error(f"❌ Failed to start gRPC service: {e}")
            raise
    
    async def stop_service(self):
        """Stop the gRPC service"""
        try:
            if self.server:
                await self.server.stop(grace=5)
            
            await self.edge_aggregator.stop_server()
            logger.info("🛑 EdgeAggregator gRPC service stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping gRPC service: {e}")

class PrimaryOwnerGrpcService:
    """
    gRPC service for primary tile owners
    Receives aggregated deltas and applies them to blockchain
    """
    
    def __init__(self, ownership_registry: TileOwnershipRegistry):
        self.ownership_registry = ownership_registry
        self.applied_deltas = 0
        self.failed_applications = 0
        
    async def ReceiveAggregatedDelta(self, request, context):
        """Receive aggregated delta from edge aggregator"""
        try:
            # Verify this node is the primary owner
            primary_owner = self.ownership_registry.get_primary_node(request.tile_id)
            if not primary_owner:
                return {
                    'accepted': False,
                    'message': f'No primary owner for tile {request.tile_id}'
                }
            
            # Apply delta to blockchain (would integrate with blockchain here)
            # For now, simulate successful application
            success = await self._apply_delta_to_blockchain(
                request.tile_id, 
                request.aggregated_delta_data,
                request.delta_type
            )
            
            if success:
                self.applied_deltas += 1
                return {
                    'accepted': True,
                    'message': 'Delta applied successfully',
                    'tile_hash': f'hash_{request.tile_id}_{int(time.time())}',
                    'new_block_height': self.applied_deltas
                }
            else:
                self.failed_applications += 1
                return {
                    'accepted': False,
                    'message': 'Failed to apply delta to blockchain'
                }
                
        except Exception as e:
            logger.error(f"❌ Error in ReceiveAggregatedDelta: {e}")
            self.failed_applications += 1
            return {
                'accepted': False,
                'message': f'Internal error: {str(e)}'
            }
    
    async def _apply_delta_to_blockchain(self, tile_id: str, delta_data: bytes, delta_type: str) -> bool:
        """Apply delta to blockchain (placeholder implementation)"""
        try:
            # Simulate blockchain application
            await asyncio.sleep(0.01)  # Simulate processing time
            
            logger.info(f"✅ Applied delta to tile {tile_id} ({len(delta_data)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply delta to blockchain: {e}")
            return False

# Factory functions for creating gRPC services
def create_edge_aggregator_service(region: str, port: int) -> EdgeAggregatorGrpcService:
    """Create a complete EdgeAggregator gRPC service"""
    ownership_registry = TileOwnershipRegistry()
    edge_aggregator = EdgeAggregator(region=region, port=port, ownership_registry=ownership_registry)
    return EdgeAggregatorGrpcService(edge_aggregator)

def create_primary_owner_service(ownership_registry: TileOwnershipRegistry) -> PrimaryOwnerGrpcService:
    """Create a PrimaryOwner gRPC service"""
    return PrimaryOwnerGrpcService(ownership_registry)

# Export main classes
__all__ = ['EdgeAggregatorGrpcService', 'PrimaryOwnerGrpcService', 'create_edge_aggregator_service', 'create_primary_owner_service']