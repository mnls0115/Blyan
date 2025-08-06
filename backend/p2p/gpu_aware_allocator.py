"""
GPU-Aware Expert Allocator
Dynamically assigns experts based on GPU capabilities
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class GPUTier:
    """GPU performance tier classification"""
    tier: str
    min_vram_gb: int
    max_experts: int
    max_layers: int
    expert_size_limit_mb: int

# GPU Tier definitions
GPU_TIERS = {
    "hobbyist": GPUTier("hobbyist", 4, 2, 1, 500),      # GTX 1060, RTX 2060
    "prosumer": GPUTier("prosumer", 8, 4, 2, 1000),     # RTX 3070, RTX 4060
    "professional": GPUTier("professional", 16, 8, 4, 2000),  # RTX 3090, RTX 4080
    "enterprise": GPUTier("enterprise", 24, 16, 8, 4000),     # RTX 4090, A100
    "datacenter": GPUTier("datacenter", 40, 32, 16, 8000)     # A100 80GB, H100
}

class GPUAwareAllocator:
    """Allocates experts based on GPU capabilities"""
    
    def __init__(self):
        self.node_capabilities: Dict[str, GPUTier] = {}
        self.expert_sizes: Dict[str, int] = {}  # Expert name -> size in MB
        
    def classify_gpu(self, vram_gb: int, tflops: float = 0, pcie_gbps: float = 0) -> int:
        """
        Classify GPU into performance tier based on multi-axis scoring
        Returns tier 0-9 (higher is better)
        """
        # Multi-axis scoring: VRAM + FP16 TFLOPS + PCIe bandwidth
        score = vram_gb * 0.5 + tflops * 2 + pcie_gbps * 0.2
        tier = min(9, int(score // 10))  # 0-9 tiers
        
        logger.info(f"GPU Score: {score:.2f} → Tier {tier} "
                   f"(VRAM:{vram_gb}GB, TFLOPS:{tflops}, PCIe:{pcie_gbps}GB/s)")
        return tier
    
    def get_tier_limits(self, tier: int) -> dict:
        """Get resource limits for a specific tier (0-9)"""
        # 10-tier system with 4GB increments
        base_vram = 4 + (tier * 4)  # 4GB to 40GB
        return {
            "max_experts": 2 ** tier,  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            "max_layers": min(16, tier + 1),
            "expert_size_limit_mb": 250 * (tier + 1),  # 250MB to 2500MB
            "max_vram_gb": base_vram
        }
    
    def calculate_expert_assignment(
        self, 
        node_id: str, 
        vram_gb: int,
        available_experts: List[str],
        expert_sizes: Dict[str, int]
    ) -> List[str]:
        """
        Calculate which experts should be assigned to a node
        based on its GPU capabilities
        """
        tier = self.classify_gpu(vram_gb)
        self.node_capabilities[node_id] = tier
        
        # Sort experts by layer and size
        layer_experts = {}
        for expert in available_experts:
            if "layer" in expert:
                layer_num = int(expert.split("layer")[1].split(".")[0])
                if layer_num not in layer_experts:
                    layer_experts[layer_num] = []
                layer_experts[layer_num].append(expert)
        
        assigned_experts = []
        total_size_mb = 0
        layers_assigned = 0
        
        # Assign experts based on tier limits
        for layer_num in sorted(layer_experts.keys()):
            if layers_assigned >= tier.max_layers:
                break
                
            for expert in layer_experts[layer_num]:
                expert_size = expert_sizes.get(expert, 100)  # Default 100MB
                
                # Check constraints
                if (len(assigned_experts) >= tier.max_experts or
                    total_size_mb + expert_size > vram_gb * 1000 * 0.8 or  # Use 80% of VRAM
                    expert_size > tier.expert_size_limit_mb):
                    continue
                    
                assigned_experts.append(expert)
                total_size_mb += expert_size
            
            if assigned_experts:
                layers_assigned += 1
        
        logger.info(f"Node {node_id} (Tier: {tier.tier}, VRAM: {vram_gb}GB) "
                   f"assigned {len(assigned_experts)} experts, "
                   f"total size: {total_size_mb}MB")
        
        return assigned_experts
    
    def suggest_expert_sharding(
        self, 
        expert_name: str, 
        expert_size_mb: int,
        target_nodes: List[str]
    ) -> Dict[str, List[str]]:
        """
        Suggest how to shard a large expert across multiple nodes
        """
        sharding_plan = {}
        
        # Calculate shard size based on smallest node capability
        min_tier = min(
            self.node_capabilities[node] 
            for node in target_nodes 
            if node in self.node_capabilities
        )
        
        shard_size_mb = min_tier.expert_size_limit_mb
        num_shards = (expert_size_mb + shard_size_mb - 1) // shard_size_mb
        
        # Distribute shards across nodes
        for i, node in enumerate(target_nodes[:num_shards]):
            shard_name = f"{expert_name}.shard{i}"
            if node not in sharding_plan:
                sharding_plan[node] = []
            sharding_plan[node].append(shard_name)
        
        logger.info(f"Expert {expert_name} ({expert_size_mb}MB) "
                   f"sharded into {num_shards} pieces across {len(sharding_plan)} nodes")
        
        return sharding_plan
    
    def rebalance_network(
        self,
        all_nodes: Dict[str, dict],
        all_experts: List[str],
        usage_stats: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """
        Rebalance expert distribution across the network
        based on usage patterns and node capabilities
        """
        new_assignments = {}
        
        # Sort experts by usage (hot to cold)
        sorted_experts = sorted(
            all_experts, 
            key=lambda e: usage_stats.get(e, 0), 
            reverse=True
        )
        
        # Sort nodes by capability (strongest to weakest)
        sorted_nodes = sorted(
            all_nodes.keys(),
            key=lambda n: self.node_capabilities.get(n, GPU_TIERS["hobbyist"]).max_experts,
            reverse=True
        )
        
        # Assign hot experts to strongest nodes
        for expert in sorted_experts:
            best_node = None
            best_score = -1
            
            for node in sorted_nodes:
                if node not in new_assignments:
                    new_assignments[node] = []
                
                tier = self.node_capabilities.get(node, GPU_TIERS["hobbyist"])
                
                # Calculate placement score
                score = 0
                if len(new_assignments[node]) < tier.max_experts:
                    score += tier.max_experts - len(new_assignments[node])
                    score += usage_stats.get(expert, 0) * 0.1
                    
                if score > best_score:
                    best_score = score
                    best_node = node
            
            if best_node:
                new_assignments[best_node].append(expert)
        
        return new_assignments


# Example usage
if __name__ == "__main__":
    allocator = GPUAwareAllocator()
    
    # Example: RTX 3090 with 24GB VRAM
    experts = allocator.calculate_expert_assignment(
        node_id="node1",
        vram_gb=24,
        available_experts=[f"layer{i}.expert{j}" for i in range(4) for j in range(8)],
        expert_sizes={f"layer{i}.expert{j}": 500 for i in range(4) for j in range(8)}
    )
    
    print(f"Assigned {len(experts)} experts to RTX 3090 node")
    
    # Example: Sharding large expert
    sharding = allocator.suggest_expert_sharding(
        expert_name="layer0.expert0",
        expert_size_mb=5000,
        target_nodes=["node1", "node2", "node3"]
    )
    
    print(f"Sharding plan: {sharding}")