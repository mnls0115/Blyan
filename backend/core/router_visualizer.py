"""
Router Traffic Visualizer for Blyan
Real-time visualization of expert routing patterns and traffic flow
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

from .dynamic_router import DynamicRouter, MultiLayerDynamicRouter
from .router_self_tuning import SelfTuningRouter

@dataclass
class RoutingFlow:
    """Represents traffic flow between tokens and experts."""
    source_token_id: int
    target_expert_id: int
    layer_id: str
    weight: float
    timestamp: float
    routing_decision_id: str

@dataclass
class ExpertTrafficStats:
    """Traffic statistics for an expert."""
    expert_id: int
    layer_id: str
    total_requests: int = 0
    total_weight: float = 0.0
    avg_weight: float = 0.0
    recent_requests: deque = field(default_factory=lambda: deque(maxlen=100))
    peak_load: float = 0.0
    last_accessed: float = 0.0
    
    def update(self, weight: float, timestamp: float):
        """Update traffic stats with new request."""
        self.total_requests += 1
        self.total_weight += weight
        self.avg_weight = self.total_weight / self.total_requests
        self.recent_requests.append({'weight': weight, 'timestamp': timestamp})
        self.peak_load = max(self.peak_load, weight)
        self.last_accessed = timestamp

@dataclass
class LayerTrafficPattern:
    """Traffic pattern analysis for a layer."""
    layer_id: str
    expert_distribution: Dict[int, float] = field(default_factory=dict)
    routing_entropy: float = 0.0
    load_balance_score: float = 0.0
    dominant_experts: List[int] = field(default_factory=list)
    traffic_variance: float = 0.0
    total_routing_decisions: int = 0

class RouterTrafficVisualizer:
    """
    Visualizes and analyzes router traffic patterns.
    
    Features:
    - Real-time traffic flow tracking
    - Expert utilization analysis
    - Load balancing metrics
    - Traffic pattern detection
    - Export capabilities for external visualization
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./data/router_visualization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Traffic tracking
        self.routing_flows: List[RoutingFlow] = []
        self.expert_stats: Dict[str, Dict[int, ExpertTrafficStats]] = defaultdict(dict)
        self.layer_patterns: Dict[str, LayerTrafficPattern] = {}
        
        # Real-time metrics
        self.current_flows: Dict[str, List[RoutingFlow]] = defaultdict(list)  # layer_id -> flows
        self.flow_history_window = 1000  # Keep last 1000 flows per layer
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.last_analysis_time = 0.0
        self.analysis_interval = 5.0  # Analyze every 5 seconds
        
        print("📊 Router Traffic Visualizer initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def record_routing_decision(self, 
                               layer_id: str,
                               expert_indices: List[int],
                               expert_weights: List[float],
                               token_count: int = 1,
                               routing_metadata: Optional[Dict[str, Any]] = None):
        """Record a routing decision for visualization."""
        timestamp = time.time()
        decision_id = f"{layer_id}_{int(timestamp * 1000000)}"  # Microsecond precision
        
        # Create routing flows
        flows = []
        for token_id in range(token_count):
            for expert_id, weight in zip(expert_indices, expert_weights):
                flow = RoutingFlow(
                    source_token_id=token_id,
                    target_expert_id=expert_id,
                    layer_id=layer_id,
                    weight=float(weight),
                    timestamp=timestamp,
                    routing_decision_id=decision_id
                )
                flows.append(flow)
        
        # Store flows
        self.routing_flows.extend(flows)
        self.current_flows[layer_id].extend(flows)
        
        # Maintain window size
        if len(self.current_flows[layer_id]) > self.flow_history_window:
            excess = len(self.current_flows[layer_id]) - self.flow_history_window
            self.current_flows[layer_id] = self.current_flows[layer_id][excess:]
        
        # Update expert statistics
        for expert_id, weight in zip(expert_indices, expert_weights):
            if expert_id not in self.expert_stats[layer_id]:
                self.expert_stats[layer_id][expert_id] = ExpertTrafficStats(
                    expert_id=expert_id,
                    layer_id=layer_id
                )
            
            self.expert_stats[layer_id][expert_id].update(weight, timestamp)
        
        # Trigger analysis if needed
        if timestamp - self.last_analysis_time > self.analysis_interval:
            self._analyze_traffic_patterns()
            self.last_analysis_time = timestamp
    
    def _analyze_traffic_patterns(self):
        """Analyze current traffic patterns."""
        for layer_id, flows in self.current_flows.items():
            if not flows:
                continue
            
            # Calculate expert distribution
            expert_weights = defaultdict(float)
            total_weight = 0.0
            
            for flow in flows:
                expert_weights[flow.target_expert_id] += flow.weight
                total_weight += flow.weight
            
            # Normalize to get distribution
            if total_weight > 0:
                expert_distribution = {
                    expert_id: weight / total_weight 
                    for expert_id, weight in expert_weights.items()
                }
            else:
                expert_distribution = {}
            
            # Calculate entropy (measure of load distribution)
            entropy = 0.0
            if expert_distribution:
                for prob in expert_distribution.values():
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
            
            # Calculate load balance score (1.0 = perfectly balanced)
            num_experts = len(expert_distribution)
            if num_experts > 1:
                max_entropy = np.log2(num_experts)
                load_balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                load_balance_score = 1.0
            
            # Find dominant experts (top 20% by traffic)
            sorted_experts = sorted(expert_distribution.items(), 
                                  key=lambda x: x[1], reverse=True)
            top_20_percent = max(1, len(sorted_experts) // 5)
            dominant_experts = [expert_id for expert_id, _ in sorted_experts[:top_20_percent]]
            
            # Calculate traffic variance
            if expert_distribution:
                weights = list(expert_distribution.values())
                traffic_variance = np.var(weights)
            else:
                traffic_variance = 0.0
            
            # Update layer pattern
            self.layer_patterns[layer_id] = LayerTrafficPattern(
                layer_id=layer_id,
                expert_distribution=expert_distribution,
                routing_entropy=entropy,
                load_balance_score=load_balance_score,
                dominant_experts=dominant_experts,
                traffic_variance=traffic_variance,
                total_routing_decisions=len(flows)
            )
        
        # Update analysis cache
        self.analysis_cache.update({
            'last_analysis': time.time(),
            'total_flows': len(self.routing_flows),
            'active_layers': len(self.current_flows),
            'total_experts_seen': sum(len(stats) for stats in self.expert_stats.values())
        })
    
    def get_real_time_traffic_data(self) -> Dict[str, Any]:
        """Get real-time traffic data for live visualization."""
        current_time = time.time()
        
        # Recent flows (last 30 seconds)
        recent_threshold = current_time - 30.0
        recent_flows = []
        
        for layer_id, flows in self.current_flows.items():
            layer_recent_flows = [
                {
                    'layer_id': flow.layer_id,
                    'expert_id': flow.target_expert_id,
                    'weight': flow.weight,
                    'timestamp': flow.timestamp,
                    'age_seconds': current_time - flow.timestamp
                }
                for flow in flows if flow.timestamp > recent_threshold
            ]
            recent_flows.extend(layer_recent_flows)
        
        # Current expert loads
        expert_loads = {}
        for layer_id, layer_stats in self.expert_stats.items():
            expert_loads[layer_id] = {}
            for expert_id, stats in layer_stats.items():
                # Calculate current load (requests in last 10 seconds)
                recent_requests = [
                    req for req in stats.recent_requests 
                    if current_time - req['timestamp'] < 10.0
                ]
                
                current_load = sum(req['weight'] for req in recent_requests)
                expert_loads[layer_id][expert_id] = {
                    'current_load': current_load,
                    'request_count': len(recent_requests),
                    'avg_weight': stats.avg_weight,
                    'total_requests': stats.total_requests,
                    'last_accessed': stats.last_accessed
                }
        
        return {
            'timestamp': current_time,
            'recent_flows': recent_flows,
            'expert_loads': expert_loads,
            'layer_patterns': {
                layer_id: {
                    'entropy': pattern.routing_entropy,
                    'load_balance_score': pattern.load_balance_score,
                    'dominant_experts': pattern.dominant_experts,
                    'expert_count': len(pattern.expert_distribution)
                }
                for layer_id, pattern in self.layer_patterns.items()
            }
        }
    
    def generate_traffic_heatmap_data(self, layer_id: str, 
                                    time_window: float = 300.0) -> Dict[str, Any]:
        """Generate heatmap data for a specific layer."""
        if layer_id not in self.current_flows:
            return {'error': f'No data for layer {layer_id}'}
        
        current_time = time.time()
        window_start = current_time - time_window
        
        # Filter flows to time window
        relevant_flows = [
            flow for flow in self.current_flows[layer_id]
            if flow.timestamp >= window_start
        ]
        
        if not relevant_flows:
            return {'error': 'No flows in time window'}
        
        # Create time buckets (e.g., 30-second intervals)
        bucket_size = 30.0  # seconds
        num_buckets = int(time_window / bucket_size)
        
        # Get all experts that appear in this time window
        all_experts = sorted(set(flow.target_expert_id for flow in relevant_flows))
        
        # Initialize heatmap matrix
        heatmap_matrix = np.zeros((len(all_experts), num_buckets))
        time_labels = []
        
        for bucket_idx in range(num_buckets):
            bucket_start = window_start + bucket_idx * bucket_size
            bucket_end = bucket_start + bucket_size
            time_labels.append(bucket_start)
            
            # Sum weights for each expert in this time bucket
            bucket_flows = [
                flow for flow in relevant_flows
                if bucket_start <= flow.timestamp < bucket_end
            ]
            
            expert_weights = defaultdict(float)
            for flow in bucket_flows:
                expert_weights[flow.target_expert_id] += flow.weight
            
            # Fill matrix
            for expert_idx, expert_id in enumerate(all_experts):
                heatmap_matrix[expert_idx, bucket_idx] = expert_weights[expert_id]
        
        return {
            'layer_id': layer_id,
            'time_window': time_window,
            'bucket_size': bucket_size,
            'expert_ids': all_experts,
            'time_labels': time_labels,
            'heatmap_matrix': heatmap_matrix.tolist(),
            'max_weight': float(np.max(heatmap_matrix)),
            'total_flows': len(relevant_flows)
        }
    
    def detect_traffic_anomalies(self, layer_id: str) -> List[Dict[str, Any]]:
        """Detect anomalous traffic patterns."""
        if layer_id not in self.layer_patterns:
            return []
        
        pattern = self.layer_patterns[layer_id]
        anomalies = []
        
        # Anomaly: Very low entropy (all traffic to few experts)
        if pattern.routing_entropy < 1.0 and len(pattern.expert_distribution) > 4:
            anomalies.append({
                'type': 'low_entropy',
                'severity': 'medium',
                'description': f'Low routing entropy ({pattern.routing_entropy:.2f}) - traffic concentrated on few experts',
                'affected_experts': pattern.dominant_experts,
                'metric_value': pattern.routing_entropy
            })
        
        # Anomaly: Very high traffic variance
        if pattern.traffic_variance > 0.1:
            anomalies.append({
                'type': 'high_variance',
                'severity': 'low',
                'description': f'High traffic variance ({pattern.traffic_variance:.3f}) - uneven load distribution',
                'metric_value': pattern.traffic_variance
            })
        
        # Anomaly: Single expert dominance (>70% of traffic)
        if pattern.expert_distribution:
            max_share = max(pattern.expert_distribution.values())
            if max_share > 0.7:
                dominant_expert = max(pattern.expert_distribution.items(), key=lambda x: x[1])[0]
                anomalies.append({
                    'type': 'expert_dominance',
                    'severity': 'high',
                    'description': f'Expert {dominant_expert} handles {max_share:.1%} of traffic',
                    'affected_experts': [dominant_expert],
                    'metric_value': max_share
                })
        
        # Anomaly: Expert starvation (experts with <1% traffic)
        if pattern.expert_distribution:
            starved_experts = [
                expert_id for expert_id, share in pattern.expert_distribution.items()
                if share < 0.01 and pattern.total_routing_decisions > 100
            ]
            
            if starved_experts:
                anomalies.append({
                    'type': 'expert_starvation',
                    'severity': 'medium',
                    'description': f'{len(starved_experts)} experts receiving <1% of traffic',
                    'affected_experts': starved_experts,
                    'metric_value': len(starved_experts)
                })
        
        return anomalies
    
    def export_visualization_data(self, format: str = 'json') -> str:
        """Export traffic data for external visualization tools."""
        # Prepare comprehensive data
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'total_flows': len(self.routing_flows),
                'active_layers': list(self.current_flows.keys()),
                'analysis_window': self.flow_history_window,
                'format_version': '1.0'
            },
            'layer_patterns': {
                layer_id: {
                    'expert_distribution': pattern.expert_distribution,
                    'routing_entropy': pattern.routing_entropy,
                    'load_balance_score': pattern.load_balance_score,
                    'dominant_experts': pattern.dominant_experts,
                    'traffic_variance': pattern.traffic_variance,
                    'total_decisions': pattern.total_routing_decisions
                }
                for layer_id, pattern in self.layer_patterns.items()
            },
            'expert_statistics': {
                layer_id: {
                    str(expert_id): {
                        'total_requests': stats.total_requests,
                        'avg_weight': stats.avg_weight,
                        'peak_load': stats.peak_load,
                        'last_accessed': stats.last_accessed,
                        'recent_request_count': len(stats.recent_requests)
                    }
                    for expert_id, stats in layer_stats.items()
                }
                for layer_id, layer_stats in self.expert_stats.items()
            },
            'anomalies': {
                layer_id: self.detect_traffic_anomalies(layer_id)
                for layer_id in self.layer_patterns.keys()
            }
        }
        
        # Export based on format
        if format.lower() == 'json':
            export_file = self.output_dir / f"traffic_data_{int(time.time())}.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📊 Traffic data exported to {export_file}")
            return str(export_file)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_traffic_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive traffic summary report."""
        current_time = time.time()
        
        # Global statistics
        total_flows = len(self.routing_flows)
        active_layers = len(self.current_flows)
        total_experts = sum(len(stats) for stats in self.expert_stats.values())
        
        # Per-layer summary
        layer_summaries = {}
        for layer_id, pattern in self.layer_patterns.items():
            layer_summaries[layer_id] = {
                'total_experts': len(pattern.expert_distribution),
                'routing_entropy': pattern.routing_entropy,
                'load_balance_score': pattern.load_balance_score,
                'dominant_expert_share': max(pattern.expert_distribution.values()) if pattern.expert_distribution else 0.0,
                'traffic_decisions': pattern.total_routing_decisions,
                'anomaly_count': len(self.detect_traffic_anomalies(layer_id))
            }
        
        # Overall health metrics
        avg_entropy = np.mean([p.routing_entropy for p in self.layer_patterns.values()]) if self.layer_patterns else 0.0
        avg_balance_score = np.mean([p.load_balance_score for p in self.layer_patterns.values()]) if self.layer_patterns else 0.0
        
        return {
            'summary': {
                'report_timestamp': current_time,
                'total_flows_recorded': total_flows,
                'active_layers': active_layers,
                'total_experts_tracked': total_experts,
                'average_entropy': avg_entropy,
                'average_load_balance': avg_balance_score
            },
            'layer_summaries': layer_summaries,
            'global_health': {
                'routing_efficiency': avg_balance_score,
                'traffic_distribution': avg_entropy,
                'anomaly_layers': [
                    layer_id for layer_id in self.layer_patterns.keys()
                    if len(self.detect_traffic_anomalies(layer_id)) > 0
                ]
            },
            'recommendations': self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations for routing optimization."""
        recommendations = []
        
        for layer_id, pattern in self.layer_patterns.items():
            # Low entropy recommendation
            if pattern.routing_entropy < 1.5 and len(pattern.expert_distribution) > 4:
                recommendations.append(
                    f"Layer {layer_id}: Consider increasing routing temperature to improve expert diversity"
                )
            
            # High variance recommendation  
            if pattern.traffic_variance > 0.08:
                recommendations.append(
                    f"Layer {layer_id}: High traffic variance detected - consider load balancing adjustments"
                )
            
            # Expert dominance recommendation
            if pattern.expert_distribution:
                max_share = max(pattern.expert_distribution.values())
                if max_share > 0.6:
                    dominant_expert = max(pattern.expert_distribution.items(), key=lambda x: x[1])[0]
                    recommendations.append(
                        f"Layer {layer_id}: Expert {dominant_expert} is handling {max_share:.1%} of traffic - consider capacity scaling"
                    )
        
        if not recommendations:
            recommendations.append("Traffic patterns appear well-balanced across all layers")
        
        return recommendations

# Integration with existing router systems
def integrate_with_dynamic_router(visualizer: RouterTrafficVisualizer, 
                                router: DynamicRouter) -> DynamicRouter:
    """Integrate visualizer with dynamic router."""
    original_select = router.select_experts
    
    def wrapped_select_experts(hidden_states, top_k=2, context=None):
        expert_indices, expert_weights = original_select(hidden_states, top_k, context)
        
        # Record the routing decision
        visualizer.record_routing_decision(
            layer_id=router.layer_id,
            expert_indices=expert_indices[0] if isinstance(expert_indices[0], list) else expert_indices,
            expert_weights=expert_weights[0].tolist() if hasattr(expert_weights[0], 'tolist') else expert_weights,
            token_count=hidden_states.shape[0] if hasattr(hidden_states, 'shape') else 1,
            routing_metadata={'router_type': 'dynamic', 'context': context}
        )
        
        return expert_indices, expert_weights
    
    router.select_experts = wrapped_select_experts
    return router

# Export main classes
__all__ = ['RouterTrafficVisualizer', 'RoutingFlow', 'ExpertTrafficStats', 'LayerTrafficPattern', 'integrate_with_dynamic_router']