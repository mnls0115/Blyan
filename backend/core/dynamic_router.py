"""
Dynamic Router System for Blyan
Performance-based expert selection with gate re-weighting
"""

import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .pol import EnhancedPoLValidator, PoLScore

@dataclass
class ExpertPerformanceMetrics:
    """Performance metrics for a single expert."""
    expert_name: str
    layer_id: str
    
    # Usage statistics
    selection_count: int = 0
    total_inference_time: float = 0.0
    avg_inference_time: float = 0.0
    
    # Quality metrics (from PoL)
    avg_improvement_score: float = 0.0
    success_rate: float = 0.0
    trust_score: float = 0.5
    
    # Recent performance (sliding window)
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_selection_probability: float = 0.0
    
    # Dynamic weighting
    performance_weight: float = 1.0
    base_gate_score: float = 0.0
    adjusted_gate_score: float = 0.0
    
    def update_metrics(self, inference_time: float, pol_score: Optional[PoLScore] = None):
        """Update metrics with new data point."""
        self.selection_count += 1
        self.total_inference_time += inference_time
        self.avg_inference_time = self.total_inference_time / self.selection_count
        
        if pol_score:
            self.recent_scores.append(pol_score.improvement_score)
            
            # Update averages
            if self.recent_scores:
                self.avg_improvement_score = np.mean(self.recent_scores)
                success_count = sum(1 for score in self.recent_scores if score > 0)
                self.success_rate = success_count / len(self.recent_scores)
        
        # Update performance weight based on recent performance
        self._update_performance_weight()
    
    def _update_performance_weight(self):
        """Calculate performance-based weight adjustment."""
        # Base weight from success rate
        success_weight = self.success_rate
        
        # Speed weight (faster = better, but with diminishing returns)
        speed_weight = 1.0
        if self.avg_inference_time > 0:
            speed_weight = min(2.0, 100.0 / self.avg_inference_time)  # Cap at 2x
        
        # Quality weight from improvement scores
        quality_weight = max(0.1, min(2.0, 1.0 + self.avg_improvement_score))
        
        # Combined weight
        self.performance_weight = success_weight * 0.4 + speed_weight * 0.3 + quality_weight * 0.3
        self.performance_weight = max(0.1, min(3.0, self.performance_weight))  # Bounded

@dataclass
class RouterState:
    """Current state of the dynamic router."""
    layer_id: str
    expert_metrics: Dict[str, ExpertPerformanceMetrics]
    base_gate_weights: torch.Tensor
    adjusted_gate_weights: torch.Tensor
    adaptation_rate: float = 0.1
    last_update: float = 0.0
    total_routing_decisions: int = 0

class DynamicRouter:
    """
    Dynamic router with performance-based expert selection.
    
    Features:
    - Real-time gate weight adjustment based on expert performance
    - PoL-based quality scoring
    - Speed and success rate optimization
    - Gradual adaptation to prevent instability
    """
    
    def __init__(self, 
                 layer_id: str,
                 initial_gate_weights: torch.Tensor,
                 pol_validator: Optional[EnhancedPoLValidator] = None,
                 adaptation_rate: float = 0.1,
                 performance_window: int = 100):
        
        self.layer_id = layer_id
        self.pol_validator = pol_validator
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        # Initialize router state
        num_experts = initial_gate_weights.shape[-1]
        self.router_state = RouterState(
            layer_id=layer_id,
            expert_metrics={},
            base_gate_weights=initial_gate_weights.clone(),
            adjusted_gate_weights=initial_gate_weights.clone(),
            adaptation_rate=adaptation_rate,
            last_update=time.time()
        )
        
        # Initialize expert metrics for each expert
        for i in range(num_experts):
            expert_name = f"expert_{i}"
            self.router_state.expert_metrics[expert_name] = ExpertPerformanceMetrics(
                expert_name=expert_name,
                layer_id=layer_id
            )
        
        # Routing history for analysis
        self.routing_history: List[Dict[str, Any]] = []
        
        print(f"🧠 Dynamic router initialized for layer {layer_id} with {num_experts} experts")
    
    def select_experts(self, 
                      hidden_states: torch.Tensor, 
                      top_k: int = 2,
                      context: Optional[Dict[str, Any]] = None) -> Tuple[List[int], torch.Tensor]:
        """
        Select top-k experts using dynamic gate weights.
        
        Args:
            hidden_states: Input hidden states
            top_k: Number of experts to select
            context: Additional context for routing decision
            
        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        start_time = time.time()
        
        # Get current adjusted gate weights
        current_gate_weights = self.router_state.adjusted_gate_weights
        
        # Compute router logits
        # In practice, this would be actual neural network computation
        batch_size = hidden_states.shape[0]
        router_logits = self._compute_router_logits(hidden_states, current_gate_weights)
        
        # Apply temperature scaling based on confidence
        temperature = self._calculate_routing_temperature()
        router_logits = router_logits / temperature
        
        # Get top-k experts
        scores, expert_indices = torch.topk(router_logits, top_k, dim=-1)
        expert_weights = F.softmax(scores, dim=-1)
        
        # Record routing decision
        routing_decision = {
            'timestamp': time.time(),
            'layer_id': self.layer_id,
            'selected_experts': expert_indices.tolist(),
            'expert_weights': expert_weights.tolist(),
            'routing_time_ms': (time.time() - start_time) * 1000,
            'temperature': temperature,
            'context': context or {}
        }
        self.routing_history.append(routing_decision)
        
        # Keep only recent history
        if len(self.routing_history) > 1000:
            self.routing_history.pop(0)
        
        self.router_state.total_routing_decisions += 1
        
        return expert_indices.tolist(), expert_weights
    
    def _compute_router_logits(self, hidden_states: torch.Tensor, 
                              gate_weights: torch.Tensor) -> torch.Tensor:
        """Compute router logits from hidden states and gate weights."""
        # Simplified routing computation
        # In practice, this would be a learned linear layer
        
        # Use mean pooling of hidden states as routing features
        routing_features = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Simple linear transformation (mock)
        # Real implementation would use actual learned parameters
        if len(routing_features.shape) == 1:
            routing_features = routing_features.unsqueeze(0)
        
        # Create mock routing logits based on gate weights
        batch_size = routing_features.shape[0]
        num_experts = gate_weights.shape[-1]
        
        # Simple routing: use gate weights directly with some noise
        base_logits = gate_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Add some variation based on input
        input_influence = torch.randn_like(base_logits) * 0.1
        router_logits = base_logits + input_influence
        
        return router_logits
    
    def _calculate_routing_temperature(self) -> float:
        """Calculate temperature for routing based on system confidence."""
        # Higher temperature (more exploration) when:
        # - System is new (few routing decisions)
        # - Recent performance is poor
        # - Expert performance varies widely
        
        base_temperature = 1.0
        
        # Experience factor
        if self.router_state.total_routing_decisions < 100:
            experience_factor = 1.5  # More exploration for new routers
        else:
            experience_factor = 1.0
        
        # Performance variation factor
        expert_weights = [m.performance_weight for m in self.router_state.expert_metrics.values()]
        if expert_weights:
            weight_std = np.std(expert_weights)
            variation_factor = 1.0 + weight_std * 0.5
        else:
            variation_factor = 1.0
        
        return base_temperature * experience_factor * variation_factor
    
    def update_expert_performance(self, 
                                 expert_indices: List[int], 
                                 inference_times: List[float],
                                 pol_scores: Optional[List[PoLScore]] = None):
        """Update expert performance metrics after inference."""
        for i, expert_idx in enumerate(expert_indices):
            expert_name = f"expert_{expert_idx}"
            
            if expert_name in self.router_state.expert_metrics:
                inference_time = inference_times[i] if i < len(inference_times) else 0.0
                pol_score = pol_scores[i] if pol_scores and i < len(pol_scores) else None
                
                self.router_state.expert_metrics[expert_name].update_metrics(
                    inference_time, pol_score
                )
        
        # Trigger gate weight adaptation
        self._adapt_gate_weights()
    
    def _adapt_gate_weights(self):
        """Adapt gate weights based on expert performance."""
        current_time = time.time()
        
        # Only adapt if enough time has passed (stability)
        if current_time - self.router_state.last_update < 1.0:  # 1 second minimum
            return
        
        # Calculate new gate weights based on performance
        num_experts = len(self.router_state.expert_metrics)
        new_weights = torch.zeros(num_experts)
        
        for i, (expert_name, metrics) in enumerate(self.router_state.expert_metrics.items()):
            # Base weight from original gate
            base_weight = self.router_state.base_gate_weights[i] if i < len(self.router_state.base_gate_weights) else 1.0
            
            # Performance adjustment
            performance_adjustment = metrics.performance_weight
            
            # Combine base and performance
            new_weights[i] = base_weight * performance_adjustment
        
        # Normalize to prevent extreme values
        new_weights = F.softmax(new_weights, dim=0) * num_experts  # Keep average at 1.0
        
        # Gradual adaptation to prevent instability
        alpha = self.adaptation_rate
        self.router_state.adjusted_gate_weights = (
            alpha * new_weights + 
            (1 - alpha) * self.router_state.adjusted_gate_weights
        )
        
        self.router_state.last_update = current_time
        
        # Update individual expert gate scores
        for i, (expert_name, metrics) in enumerate(self.router_state.expert_metrics.items()):
            metrics.adjusted_gate_score = self.router_state.adjusted_gate_weights[i].item()
            metrics.base_gate_score = self.router_state.base_gate_weights[i].item()
    
    def get_expert_rankings(self) -> List[Tuple[str, float]]:
        """Get current expert rankings by performance."""
        rankings = []
        
        for expert_name, metrics in self.router_state.expert_metrics.items():
            score = (
                metrics.performance_weight * 0.5 + 
                metrics.success_rate * 0.3 + 
                (1.0 / max(0.001, metrics.avg_inference_time)) * 0.2
            )
            rankings.append((expert_name, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_selections = sum(m.selection_count for m in self.router_state.expert_metrics.values())
        
        expert_stats = {}
        for expert_name, metrics in self.router_state.expert_metrics.items():
            selection_probability = metrics.selection_count / max(1, total_selections)
            
            expert_stats[expert_name] = {
                'selection_count': metrics.selection_count,
                'selection_probability': selection_probability,
                'avg_inference_time': metrics.avg_inference_time,
                'success_rate': metrics.success_rate,
                'performance_weight': metrics.performance_weight,
                'base_gate_score': metrics.base_gate_score,
                'adjusted_gate_score': metrics.adjusted_gate_score,
                'avg_improvement_score': metrics.avg_improvement_score
            }
        
        return {
            'layer_id': self.layer_id,
            'total_routing_decisions': self.router_state.total_routing_decisions,
            'total_expert_selections': total_selections,
            'adaptation_rate': self.adaptation_rate,
            'last_adaptation': self.router_state.last_update,
            'expert_stats': expert_stats,
            'current_temperature': self._calculate_routing_temperature(),
            'gate_weight_entropy': self._calculate_gate_entropy()
        }
    
    def _calculate_gate_entropy(self) -> float:
        """Calculate entropy of current gate weights (diversity measure)."""
        weights = F.softmax(self.router_state.adjusted_gate_weights, dim=0)
        log_weights = torch.log(weights + 1e-8)
        entropy = -torch.sum(weights * log_weights).item()
        return entropy
    
    def export_routing_analysis(self) -> Dict[str, Any]:
        """Export detailed routing analysis for debugging."""
        expert_rankings = self.get_expert_rankings()
        routing_stats = self.get_routing_stats()
        
        # Analyze routing patterns over time
        recent_routing = self.routing_history[-50:] if len(self.routing_history) >= 50 else self.routing_history
        
        routing_patterns = defaultdict(int)
        for decision in recent_routing:
            for expert_list in decision['selected_experts']:
                if isinstance(expert_list, list):
                    pattern = tuple(sorted(expert_list))
                else:
                    pattern = (expert_list,)
                routing_patterns[pattern] += 1
        
        return {
            'routing_stats': routing_stats,
            'expert_rankings': expert_rankings,
            'routing_patterns': dict(routing_patterns),
            'adaptation_history': {
                'base_weights': self.router_state.base_gate_weights.tolist(),
                'current_weights': self.router_state.adjusted_gate_weights.tolist(),
                'weight_changes': (self.router_state.adjusted_gate_weights - self.router_state.base_gate_weights).tolist()
            },
            'recent_decisions': recent_routing
        }
    
    def reset_adaptation(self):
        """Reset router to original weights (useful for debugging)."""
        self.router_state.adjusted_gate_weights = self.router_state.base_gate_weights.clone()
        
        # Reset expert metrics
        for metrics in self.router_state.expert_metrics.values():
            metrics.performance_weight = 1.0
            metrics.recent_scores.clear()
        
        print(f"🔄 Router adaptation reset for layer {self.layer_id}")

class MultiLayerDynamicRouter:
    """Manages multiple dynamic routers across all layers."""
    
    def __init__(self, pol_validator: Optional[EnhancedPoLValidator] = None):
        self.pol_validator = pol_validator
        self.layer_routers: Dict[str, DynamicRouter] = {}
        self.global_stats = {
            'total_routing_decisions': 0,
            'total_adaptations': 0,
            'start_time': time.time()
        }
    
    def add_layer_router(self, layer_id: str, initial_gate_weights: torch.Tensor) -> DynamicRouter:
        """Add a dynamic router for a layer."""
        router = DynamicRouter(
            layer_id=layer_id,
            initial_gate_weights=initial_gate_weights,
            pol_validator=self.pol_validator
        )
        
        self.layer_routers[layer_id] = router
        return router
    
    def route_layer(self, layer_id: str, hidden_states: torch.Tensor, 
                   top_k: int = 2) -> Tuple[List[int], torch.Tensor]:
        """Route for a specific layer."""
        if layer_id not in self.layer_routers:
            raise ValueError(f"No router found for layer {layer_id}")
        
        result = self.layer_routers[layer_id].select_experts(hidden_states, top_k)
        self.global_stats['total_routing_decisions'] += 1
        return result
    
    def update_layer_performance(self, layer_id: str, expert_indices: List[int], 
                               inference_times: List[float], 
                               pol_scores: Optional[List[PoLScore]] = None):
        """Update performance for a specific layer."""
        if layer_id in self.layer_routers:
            self.layer_routers[layer_id].update_expert_performance(
                expert_indices, inference_times, pol_scores
            )
    
    def get_global_routing_report(self) -> Dict[str, Any]:
        """Get comprehensive report across all layers."""
        layer_reports = {}
        for layer_id, router in self.layer_routers.items():
            layer_reports[layer_id] = router.get_routing_stats()
        
        return {
            'global_stats': self.global_stats,
            'layer_reports': layer_reports,
            'total_layers': len(self.layer_routers),
            'uptime_seconds': time.time() - self.global_stats['start_time']
        }

# Export main classes
__all__ = ['DynamicRouter', 'MultiLayerDynamicRouter', 'ExpertPerformanceMetrics', 'RouterState']