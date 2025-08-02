"""Expert Group Optimization for AI-Block Distributed Inference

This module implements the expert group-based optimization strategy where:
1. Nodes store frequently co-used expert combinations
2. Router pre-selects expert groups needed for entire prompts
3. Inference is delegated to nodes with complete expert groups
4. Hot expert combinations are replicated across multiple nodes
"""

from __future__ import annotations

import json
import time
import hashlib
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter

from backend.model.moe_infer import ExpertUsageTracker


@dataclass
class ExpertGroup:
    """Represents a group of experts that are frequently used together."""
    experts: Set[str]
    usage_count: int = 0
    co_occurrence_score: float = 0.0
    average_latency: float = 0.0
    last_used: float = 0.0
    
    @property
    def group_id(self) -> str:
        """Generate a unique ID for this expert group."""
        sorted_experts = sorted(self.experts)
        return hashlib.md5('|'.join(sorted_experts).encode()).hexdigest()[:12]
    
    def __hash__(self):
        return hash(frozenset(self.experts))
    
    def __eq__(self, other):
        if not isinstance(other, ExpertGroup):
            return False
        return self.experts == other.experts


@dataclass
class NodeCapability:
    """Enhanced node capability tracking expert groups."""
    node_id: str
    host: str
    port: int
    expert_groups: List[ExpertGroup]
    individual_experts: Set[str]  # For fallback
    load_factor: float = 0.0
    last_heartbeat: float = 0.0
    region: str = "default"
    
    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def can_handle_experts(self, required_experts: Set[str]) -> bool:
        """Check if this node can handle all required experts."""
        # First check if any expert group contains all required experts
        for group in self.expert_groups:
            if required_experts.issubset(group.experts):
                return True
        
        # Fallback: check individual experts
        return required_experts.issubset(self.individual_experts)
    
    def get_matching_group(self, required_experts: Set[str]) -> Optional[ExpertGroup]:
        """Get the expert group that best matches the required experts."""
        best_group = None
        best_coverage = 0
        
        for group in self.expert_groups:
            coverage = len(required_experts.intersection(group.experts))
            if coverage > best_coverage and required_experts.issubset(group.experts):
                best_coverage = coverage
                best_group = group
        
        return best_group


class ExpertGroupAnalyzer:
    """Analyzes expert usage patterns to identify frequently co-used groups."""
    
    def __init__(self, usage_tracker: ExpertUsageTracker, min_co_occurrence: int = 5):
        self.usage_tracker = usage_tracker
        self.min_co_occurrence = min_co_occurrence
        self.co_occurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        self.request_history: List[Set[str]] = []
        
    def record_expert_request(self, expert_names: Set[str]):
        """Record a request with multiple experts."""
        self.request_history.append(expert_names)
        
        # Update co-occurrence matrix
        expert_list = list(expert_names)
        for i, expert_a in enumerate(expert_list):
            for expert_b in expert_list[i+1:]:
                pair = tuple(sorted([expert_a, expert_b]))
                self.co_occurrence_matrix[pair] += 1
    
    def identify_expert_groups(self, min_group_size: int = 2, max_group_size: int = 6) -> List[ExpertGroup]:
        """Identify frequently co-used expert groups."""
        if len(self.request_history) < self.min_co_occurrence:
            return []
        
        # Count frequency of each expert combination
        group_counter = Counter()
        
        for request_experts in self.request_history:
            if len(request_experts) < min_group_size:
                continue
            
            # Generate all possible subgroups
            expert_list = list(request_experts)
            for size in range(min_group_size, min(len(expert_list) + 1, max_group_size + 1)):
                from itertools import combinations
                for combo in combinations(expert_list, size):
                    group_key = frozenset(combo)
                    group_counter[group_key] += 1
        
        # Filter groups by minimum co-occurrence
        expert_groups = []
        for expert_set, count in group_counter.items():
            if count >= self.min_co_occurrence:
                # Calculate co-occurrence score
                total_requests = len(self.request_history)
                co_occurrence_score = count / total_requests
                
                group = ExpertGroup(
                    experts=set(expert_set),
                    usage_count=count,
                    co_occurrence_score=co_occurrence_score,
                    last_used=time.time()
                )
                expert_groups.append(group)
        
        # Sort by usage frequency and co-occurrence score
        expert_groups.sort(key=lambda g: (g.usage_count, g.co_occurrence_score), reverse=True)
        return expert_groups


class ExpertGroupIndex:
    """Index of nodes and their expert group capabilities."""
    
    def __init__(self):
        self.nodes: Dict[str, NodeCapability] = {}
        self.group_to_nodes: Dict[str, List[str]] = defaultdict(list)  # group_id -> [node_ids]
        self.expert_to_nodes: Dict[str, List[str]] = defaultdict(list)  # expert -> [node_ids]
        self.group_analyzer = None
        
    def set_analyzer(self, analyzer: ExpertGroupAnalyzer):
        """Set the expert group analyzer."""
        self.group_analyzer = analyzer
    
    def register_node(self, node: NodeCapability):
        """Register a node with its expert group capabilities."""
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        
        # Update group mappings
        for group in node.expert_groups:
            group_id = group.group_id
            if node.node_id not in self.group_to_nodes[group_id]:
                self.group_to_nodes[group_id].append(node.node_id)
        
        # Update individual expert mappings (for fallback)
        for expert in node.individual_experts:
            if node.node_id not in self.expert_to_nodes[expert]:
                self.expert_to_nodes[expert].append(node.node_id)
        
        print(f"✓ Registered node {node.node_id} with {len(node.expert_groups)} expert groups")
    
    def unregister_node(self, node_id: str):
        """Remove a node from the index."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Remove from group mappings
        for group in node.expert_groups:
            group_id = group.group_id
            self.group_to_nodes[group_id] = [
                nid for nid in self.group_to_nodes[group_id] if nid != node_id
            ]
            if not self.group_to_nodes[group_id]:
                del self.group_to_nodes[group_id]
        
        # Remove from expert mappings
        for expert in node.individual_experts:
            self.expert_to_nodes[expert] = [
                nid for nid in self.expert_to_nodes[expert] if nid != node_id
            ]
            if not self.expert_to_nodes[expert]:
                del self.expert_to_nodes[expert]
        
        del self.nodes[node_id]
        print(f"✗ Unregistered node {node_id}")
    
    def find_optimal_nodes(self, required_experts: Set[str]) -> List[Tuple[NodeCapability, Optional[ExpertGroup]]]:
        """Find nodes that can optimally handle the required experts."""
        candidates = []
        
        # Strategy 1: Find nodes with complete expert groups
        for node_id, node in self.nodes.items():
            if not self._is_node_healthy(node):
                continue
                
            matching_group = node.get_matching_group(required_experts)
            if matching_group:
                candidates.append((node, matching_group))
        
        # Strategy 2: Fallback to nodes with individual experts
        if not candidates:
            for node_id, node in self.nodes.items():
                if not self._is_node_healthy(node):
                    continue
                    
                if node.can_handle_experts(required_experts):
                    candidates.append((node, None))
        
        # Sort by preference: complete groups first, then by load
        def sort_key(item):
            node, group = item
            has_group = 1 if group else 0
            return (has_group, -node.load_factor, -node.last_heartbeat)
        
        candidates.sort(key=sort_key, reverse=True)
        return candidates
    
    def _is_node_healthy(self, node: NodeCapability) -> bool:
        """Check if node is healthy and responsive."""
        current_time = time.time()
        return (current_time - node.last_heartbeat) < 60.0  # 60 second timeout
    
    def get_hot_expert_groups(self, limit: int = 10) -> List[ExpertGroup]:
        """Get the most frequently used expert groups."""
        if not self.group_analyzer:
            return []
        
        return self.group_analyzer.identify_expert_groups()[:limit]
    
    def suggest_replication(self) -> List[Tuple[ExpertGroup, List[str]]]:
        """Suggest expert groups that should be replicated to more nodes."""
        suggestions = []
        hot_groups = self.get_hot_expert_groups(limit=5)
        
        for group in hot_groups:
            group_id = group.group_id
            current_nodes = self.group_to_nodes.get(group_id, [])
            
            # Suggest replication if group is popular but has few nodes
            if group.usage_count > 10 and len(current_nodes) < 3:
                # Find nodes that could host this group
                potential_nodes = []
                for node_id, node in self.nodes.items():
                    if node_id not in current_nodes and self._is_node_healthy(node):
                        # Check if node has capacity and relevant experts
                        if node.load_factor < 0.7:  # Not too loaded
                            potential_nodes.append(node_id)
                
                if potential_nodes:
                    suggestions.append((group, potential_nodes[:2]))  # Suggest 2 additional nodes
        
        return suggestions


class DistributedInferenceRouter:
    """Enhanced router that uses expert group optimization."""
    
    def __init__(self, group_index: ExpertGroupIndex, usage_tracker: ExpertUsageTracker):
        self.group_index = group_index
        self.usage_tracker = usage_tracker
        
        # Setup group analyzer
        self.group_analyzer = ExpertGroupAnalyzer(usage_tracker)
        self.group_index.set_analyzer(self.group_analyzer)
    
    def route_inference_request(
        self, 
        prompt: str, 
        required_experts: List[str],
        preferred_region: str = "default"
    ) -> Tuple[Optional[NodeCapability], Optional[ExpertGroup], Dict[str, Any]]:
        """Route inference request to the optimal node."""
        
        required_experts_set = set(required_experts)
        
        # Record this expert combination for analysis
        self.group_analyzer.record_expert_request(required_experts_set)
        
        # Find optimal nodes
        candidates = self.group_index.find_optimal_nodes(required_experts_set)
        
        if not candidates:
            return None, None, {"error": "No available nodes for required experts"}
        
        # Select best candidate
        selected_node, matching_group = candidates[0]
        
        # Prepare routing info
        routing_info = {
            "selected_node": selected_node.node_id,
            "endpoint": selected_node.endpoint,
            "uses_expert_group": matching_group is not None,
            "group_id": matching_group.group_id if matching_group else None,
            "load_factor": selected_node.load_factor,
            "total_candidates": len(candidates)
        }
        
        return selected_node, matching_group, routing_info
    
    def get_replication_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for expert group replication."""
        suggestions = self.group_index.suggest_replication()
        
        result = []
        for group, potential_nodes in suggestions:
            result.append({
                "group_id": group.group_id,
                "experts": list(group.experts),
                "usage_count": group.usage_count,
                "co_occurrence_score": group.co_occurrence_score,
                "suggested_nodes": potential_nodes
            })
        
        return result
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        total_nodes = len(self.group_index.nodes)
        total_groups = sum(len(node.expert_groups) for node in self.group_index.nodes.values())
        hot_groups = self.group_index.get_hot_expert_groups()
        
        return {
            "total_nodes": total_nodes,
            "total_expert_groups": total_groups,
            "hot_groups_count": len(hot_groups),
            "requests_analyzed": len(self.group_analyzer.request_history),
            "co_occurrence_patterns": len(self.group_analyzer.co_occurrence_matrix),
            "replication_suggestions": len(self.group_index.suggest_replication())
        }