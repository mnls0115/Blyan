"""
Dynamic Rebalance Daemon
Monitors expert usage and automatically rebalances network load
"""

import asyncio
import time
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class ExpertUsageStats:
    """Track usage statistics for experts"""
    expert_name: str
    usage_count: int
    avg_latency_ms: float
    last_used: float
    node_assignments: List[str]
    
    @property
    def is_hot(self) -> bool:
        """Expert is hot if used > 10 times in last 10 minutes"""
        return (self.usage_count > 10 and 
                time.time() - self.last_used < 600)
    
    @property
    def is_cold(self) -> bool:
        """Expert is cold if not used in last hour"""
        return time.time() - self.last_used > 3600

class RebalanceEngine:
    """Engine for dynamic expert rebalancing"""
    
    def __init__(self, rebalance_interval: int = 600):
        self.rebalance_interval = rebalance_interval  # 10 minutes default
        self.usage_stats: Dict[str, ExpertUsageStats] = {}
        self.node_loads: Dict[str, float] = {}  # node_id -> load percentage
        self.rebalance_history: List[dict] = []
        
    def update_usage(self, expert_name: str, node_id: str, latency_ms: float):
        """Update usage statistics for an expert"""
        if expert_name not in self.usage_stats:
            self.usage_stats[expert_name] = ExpertUsageStats(
                expert_name=expert_name,
                usage_count=0,
                avg_latency_ms=0,
                last_used=time.time(),
                node_assignments=[]
            )
        
        stats = self.usage_stats[expert_name]
        stats.usage_count += 1
        stats.last_used = time.time()
        
        # Update average latency (moving average)
        stats.avg_latency_ms = (
            (stats.avg_latency_ms * (stats.usage_count - 1) + latency_ms) / 
            stats.usage_count
        )
        
        if node_id not in stats.node_assignments:
            stats.node_assignments.append(node_id)
    
    def generate_usage_heatmap(self) -> Dict[str, float]:
        """
        Generate heatmap of expert usage
        Returns dict of expert_name -> heat_score (0-100)
        """
        heatmap = {}
        
        if not self.usage_stats:
            return heatmap
        
        # Find max usage for normalization
        max_usage = max(s.usage_count for s in self.usage_stats.values())
        
        for expert_name, stats in self.usage_stats.items():
            # Calculate heat score (0-100)
            recency_factor = 1.0 if stats.is_hot else 0.5 if not stats.is_cold else 0.1
            usage_factor = stats.usage_count / max_usage if max_usage > 0 else 0
            
            heat_score = min(100, (usage_factor * 70 + recency_factor * 30))
            heatmap[expert_name] = heat_score
        
        logger.info(f"Generated heatmap for {len(heatmap)} experts")
        return heatmap
    
    def identify_replication_candidates(self, top_percent: float = 5) -> List[str]:
        """
        Identify top N% hottest experts for replication
        """
        heatmap = self.generate_usage_heatmap()
        
        if not heatmap:
            return []
        
        # Sort by heat score
        sorted_experts = sorted(heatmap.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N%
        num_candidates = max(1, int(len(sorted_experts) * top_percent / 100))
        candidates = [expert for expert, score in sorted_experts[:num_candidates]]
        
        logger.info(f"Identified {len(candidates)} hot experts for replication: "
                   f"{candidates[:3]}{'...' if len(candidates) > 3 else ''}")
        
        return candidates
    
    def identify_eviction_candidates(self, threshold_usage: int = 1) -> List[str]:
        """
        Identify cold experts for eviction from cache
        """
        candidates = []
        
        for expert_name, stats in self.usage_stats.items():
            if stats.is_cold and stats.usage_count <= threshold_usage:
                candidates.append(expert_name)
        
        logger.info(f"Identified {len(candidates)} cold experts for eviction")
        return candidates
    
    def calculate_node_loads(self, node_experts: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate load percentage for each node based on expert usage
        """
        node_loads = {}
        
        for node_id, experts in node_experts.items():
            total_heat = 0
            for expert in experts:
                if expert in self.usage_stats:
                    heatmap = self.generate_usage_heatmap()
                    total_heat += heatmap.get(expert, 0)
            
            # Normalize to 0-100%
            node_loads[node_id] = min(100, total_heat / len(experts)) if experts else 0
        
        self.node_loads = node_loads
        return node_loads
    
    def generate_rebalance_plan(
        self, 
        node_experts: Dict[str, List[str]],
        node_capacities: Dict[str, int]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate a rebalance plan to optimize load distribution
        
        Returns:
            Dict with 'add' and 'remove' operations per node
        """
        plan = {}
        
        # Calculate current loads
        node_loads = self.calculate_node_loads(node_experts)
        
        # Identify hot experts for replication
        hot_experts = self.identify_replication_candidates()
        
        # Identify cold experts for eviction
        cold_experts = self.identify_eviction_candidates()
        
        # Find overloaded and underloaded nodes
        avg_load = sum(node_loads.values()) / len(node_loads) if node_loads else 50
        overloaded_nodes = {n: l for n, l in node_loads.items() if l > avg_load + 20}
        underloaded_nodes = {n: l for n, l in node_loads.items() if l < avg_load - 20}
        
        # Replicate hot experts to underloaded nodes
        for expert in hot_experts:
            if expert in self.usage_stats:
                current_nodes = self.usage_stats[expert].node_assignments
                
                for node_id in underloaded_nodes:
                    if node_id not in current_nodes:
                        # Check capacity
                        current_count = len(node_experts.get(node_id, []))
                        capacity = node_capacities.get(node_id, 10)
                        
                        if current_count < capacity:
                            if node_id not in plan:
                                plan[node_id] = {'add': [], 'remove': []}
                            plan[node_id]['add'].append(expert)
                            break
        
        # Remove cold experts from overloaded nodes
        for node_id in overloaded_nodes:
            if node_id not in plan:
                plan[node_id] = {'add': [], 'remove': []}
            
            node_cold_experts = [
                e for e in node_experts.get(node_id, [])
                if e in cold_experts
            ]
            
            # Remove up to 20% of cold experts
            num_to_remove = max(1, len(node_cold_experts) // 5)
            plan[node_id]['remove'].extend(node_cold_experts[:num_to_remove])
        
        # Log the plan
        total_adds = sum(len(p['add']) for p in plan.values())
        total_removes = sum(len(p['remove']) for p in plan.values())
        
        logger.info(f"Generated rebalance plan: {total_adds} additions, {total_removes} removals")
        
        # Save to history
        self.rebalance_history.append({
            'timestamp': time.time(),
            'plan': plan,
            'node_loads': node_loads,
            'hot_experts': hot_experts[:5],  # Top 5
            'cold_experts': cold_experts[:5]   # Bottom 5
        })
        
        return plan
    
    async def run_rebalance_daemon(self, get_state_callback, apply_plan_callback):
        """
        Run the rebalance daemon
        
        Args:
            get_state_callback: Async function to get current node/expert state
            apply_plan_callback: Async function to apply rebalance plan
        """
        logger.info(f"Starting rebalance daemon (interval: {self.rebalance_interval}s)")
        
        while True:
            try:
                # Get current state
                state = await get_state_callback()
                node_experts = state['node_experts']
                node_capacities = state['node_capacities']
                
                # Generate rebalance plan
                plan = self.generate_rebalance_plan(node_experts, node_capacities)
                
                if plan:
                    # Apply the plan
                    logger.info("Applying rebalance plan...")
                    await apply_plan_callback(plan)
                    logger.info("Rebalance plan applied successfully")
                else:
                    logger.debug("No rebalancing needed")
                
            except Exception as e:
                logger.error(f"Rebalance error: {e}")
            
            # Wait for next interval
            await asyncio.sleep(self.rebalance_interval)
    
    def get_status(self) -> dict:
        """Get current rebalance engine status"""
        heatmap = self.generate_usage_heatmap()
        
        return {
            'total_experts': len(self.usage_stats),
            'hot_experts': len([e for e, s in self.usage_stats.items() if s.is_hot]),
            'cold_experts': len([e for e, s in self.usage_stats.items() if s.is_cold]),
            'avg_node_load': sum(self.node_loads.values()) / len(self.node_loads) if self.node_loads else 0,
            'max_node_load': max(self.node_loads.values()) if self.node_loads else 0,
            'top_5_hottest': sorted(heatmap.items(), key=lambda x: x[1], reverse=True)[:5],
            'last_rebalance': self.rebalance_history[-1]['timestamp'] if self.rebalance_history else None
        }


# CLI tool for testing
if __name__ == "__main__":
    import random
    
    engine = RebalanceEngine(rebalance_interval=10)
    
    # Simulate usage data
    experts = [f"layer{i}.expert{j}" for i in range(4) for j in range(8)]
    nodes = [f"node{i}" for i in range(5)]
    
    # Generate random usage
    for _ in range(100):
        expert = random.choice(experts)
        node = random.choice(nodes)
        latency = random.uniform(10, 100)
        engine.update_usage(expert, node, latency)
    
    # Make some experts hot
    hot_experts = random.sample(experts, 5)
    for expert in hot_experts:
        for _ in range(20):
            engine.update_usage(expert, random.choice(nodes), random.uniform(5, 50))
    
    # Generate heatmap
    heatmap = engine.generate_usage_heatmap()
    print("\n📊 Usage Heatmap (Top 10):")
    for expert, score in sorted(heatmap.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {expert}: {'█' * int(score/10)} {score:.1f}")
    
    # Get replication candidates
    hot = engine.identify_replication_candidates()
    print(f"\n🔥 Hot experts for replication: {hot}")
    
    # Get eviction candidates
    cold = engine.identify_eviction_candidates()
    print(f"\n❄️  Cold experts for eviction: {cold[:5]}...")
    
    # Generate rebalance plan
    node_experts = {node: random.sample(experts, 10) for node in nodes}
    node_capacities = {node: 15 for node in nodes}
    
    plan = engine.generate_rebalance_plan(node_experts, node_capacities)
    print(f"\n📋 Rebalance Plan:")
    for node, ops in plan.items():
        if ops['add'] or ops['remove']:
            print(f"  {node}:")
            if ops['add']:
                print(f"    + Add: {ops['add']}")
            if ops['remove']:
                print(f"    - Remove: {ops['remove']}")
    
    # Get status
    status = engine.get_status()
    print(f"\n📈 Status: {json.dumps(status, indent=2, default=str)}")