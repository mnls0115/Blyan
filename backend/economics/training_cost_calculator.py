#!/usr/bin/env python3
"""
Training Cost Calculator for Blyan Network
Calculates actual GPU costs for model training
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from decimal import Decimal
import math

@dataclass
class GPUSpec:
    """GPU specifications for cost calculation."""
    name: str
    tflops_fp16: float  # FP16 TFLOPS
    power_watts: int  # Power consumption
    cost_per_hour: float  # USD per hour (cloud spot price)
    memory_gb: int

# Common GPUs and their specs
GPU_SPECS = {
    "A100-40GB": GPUSpec("A100-40GB", 240, 300, 1.80, 40),
    "A100-80GB": GPUSpec("A100-80GB", 240, 400, 2.50, 80),
    "A6000": GPUSpec("A6000", 150, 300, 1.20, 48),
    "RTX4090": GPUSpec("RTX4090", 82, 450, 0.80, 24),
    "V100": GPUSpec("V100", 120, 250, 1.00, 32),
    "T4": GPUSpec("T4", 65, 70, 0.35, 16),
}

class TrainingCostCalculator:
    """
    Calculates training costs and required BLY rewards.
    """
    
    def __init__(self, gpu_type: str = "A100-40GB"):
        self.gpu = GPU_SPECS.get(gpu_type, GPU_SPECS["A100-40GB"])
        self.efficiency = 0.35  # 35% GPU utilization (realistic for distributed)
        
    def calculate_training_cost(
        self,
        model_params: int,  # Number of parameters (in billions)
        tokens: int,  # Training tokens (in billions)
        batch_size: int = 2048,
        gradient_accumulation: int = 4,
        training_type: str = "pretrain"  # "pretrain" or "finetune"
    ) -> Dict[str, any]:
        """
        Calculate training cost for a model.
        
        Args:
            model_params: Model size in billions of parameters
            tokens: Training tokens in billions
            batch_size: Batch size per GPU
            gradient_accumulation: Gradient accumulation steps
            training_type: "pretrain" (6x FLOPs) or "finetune" (2x FLOPs)
            
        Returns:
            Cost breakdown and BLY requirements
        """
        # Calculate FLOPs required
        # Pretrain: 6 * params * tokens (forward + backward + optimizer)
        # Finetune: 2 * params * tokens (forward + backward only)
        flops_multiplier = 6 if training_type == "pretrain" else 2
        total_flops = flops_multiplier * model_params * 1e9 * tokens * 1e9
        
        # Calculate GPU hours needed
        gpu_tflops_effective = self.gpu.tflops_fp16 * self.efficiency
        gpu_flops_per_hour = gpu_tflops_effective * 1e12 * 3600
        gpu_hours = total_flops / gpu_flops_per_hour
        
        # Calculate costs
        hardware_cost = gpu_hours * self.gpu.cost_per_hour
        electricity_cost = (self.gpu.power_watts / 1000) * gpu_hours * 0.12  # $0.12/kWh
        total_cost_usd = hardware_cost + electricity_cost
        
        # Memory requirements
        model_memory_gb = self._estimate_model_memory(model_params)
        gpus_needed = math.ceil(model_memory_gb / self.gpu.memory_gb)
        
        # Distributed training time
        distributed_hours = gpu_hours / gpus_needed
        
        return {
            "model_size_B": model_params,
            "tokens_B": tokens,
            "training_type": training_type,
            "total_flops": total_flops,
            "gpu_hours": gpu_hours,
            "hardware_cost_usd": hardware_cost,
            "electricity_cost_usd": electricity_cost,
            "total_cost_usd": total_cost_usd,
            "gpus_needed": gpus_needed,
            "distributed_hours": distributed_hours,
            "memory_required_gb": model_memory_gb,
            "gpu_type": self.gpu.name
        }
        
    def calculate_finetune_cost(
        self,
        model_params: int,
        dataset_size_gb: float = 10,  # Fine-tuning dataset size
        epochs: float = 0.25  # Fraction of epochs
    ) -> Dict[str, any]:
        """
        Calculate fine-tuning cost (more realistic for blockchain use).
        
        Args:
            model_params: Model size in billions
            dataset_size_gb: Dataset size in GB
            epochs: Number of epochs (0.25 = 1/4 epoch)
            
        Returns:
            Fine-tuning cost breakdown
        """
        # Estimate tokens from dataset (1GB ≈ 0.25B tokens for text)
        tokens_billions = dataset_size_gb * 0.25 * epochs
        
        # Calculate with finetune multiplier (2x instead of 6x)
        costs = self.calculate_training_cost(
            model_params=model_params,
            tokens=tokens_billions,
            training_type="finetune"
        )
        
        costs["dataset_size_gb"] = dataset_size_gb
        costs["epochs"] = epochs
        return costs
    
    def calculate_bly_rewards(
        self,
        model_params: int,
        tokens: int,
        bly_price: float = 0.10,
        training_type: str = "pretrain"
    ) -> Dict[str, any]:
        """
        Calculate BLY rewards needed for training.
        
        Args:
            model_params: Model size in billions
            tokens: Training tokens in billions  
            bly_price: Current BLY price in USD
            training_type: "pretrain" or "finetune"
            
        Returns:
            BLY reward requirements
        """
        # Get training costs
        costs = self.calculate_training_cost(model_params, tokens, training_type=training_type)
        
        # Add 20% margin for network overhead
        total_cost_with_margin = costs["total_cost_usd"] * 1.2
        
        # Convert to BLY
        bly_required = total_cost_with_margin / bly_price
        
        # Calculate per-epoch rewards (assuming 10 epochs)
        epochs = 10
        bly_per_epoch = bly_required / epochs
        
        # Calculate per-hour rewards
        bly_per_hour = bly_required / costs["distributed_hours"]
        
        return {
            "total_bly_required": bly_required,
            "bly_per_epoch": bly_per_epoch,
            "bly_per_hour": bly_per_hour,
            "usd_cost": total_cost_with_margin,
            "bly_price": bly_price,
            "training_hours": costs["distributed_hours"],
            "gpus_needed": costs["gpus_needed"]
        }
        
    def estimate_node_rewards(
        self,
        model_params: int,
        contribution_percent: float,
        bly_price: float = 0.10
    ) -> Dict[str, any]:
        """
        Estimate rewards for a single node's contribution.
        
        Args:
            model_params: Model size being trained
            contribution_percent: Node's contribution (0-100)
            bly_price: Current BLY price
            
        Returns:
            Expected rewards for the node
        """
        # Standard training run (300B tokens)
        rewards = self.calculate_bly_rewards(model_params, 300, bly_price)
        
        # Calculate node's share
        node_share = rewards["total_bly_required"] * (contribution_percent / 100)
        
        # Daily rewards (assuming 30-day training)
        daily_rewards = node_share / 30
        
        return {
            "total_rewards_bly": node_share,
            "daily_rewards_bly": daily_rewards,
            "hourly_rewards_bly": node_share / (rewards["training_hours"] * rewards["gpus_needed"]),
            "contribution_percent": contribution_percent,
            "total_usd_value": node_share * bly_price
        }
        
    def _estimate_model_memory(self, params_billions: float) -> float:
        """Estimate GPU memory needed for model."""
        # Rule of thumb: 20 bytes per parameter for training
        # (model weights, gradients, optimizer states, activations)
        bytes_per_param = 20
        memory_gb = (params_billions * 1e9 * bytes_per_param) / (1024**3)
        return memory_gb
        
    def get_common_models_cost(self, bly_price: float = 0.10) -> Dict[str, any]:
        """Get training costs for common model sizes."""
        models = {
            "GPT-2 (1.5B)": (1.5, 40),
            "GPT-3 Small (6.7B)": (6.7, 150),
            "GPT-3 Medium (13B)": (13, 300),
            "GPT-20B": (20, 300),
            "LLaMA-30B": (30, 500),
            "GPT-3 (175B)": (175, 300),
            "GPT-4 Scale (1T)": (1000, 1000)
        }
        
        results = {}
        for name, (params, tokens) in models.items():
            costs = self.calculate_training_cost(params, tokens)
            rewards = self.calculate_bly_rewards(params, tokens, bly_price)
            
            results[name] = {
                "params_B": params,
                "tokens_B": tokens,
                "cost_usd": costs["total_cost_usd"],
                "gpu_hours": costs["gpu_hours"],
                "gpus_needed": costs["gpus_needed"],
                "bly_required": rewards["total_bly_required"],
                "training_days": costs["distributed_hours"] / 24
            }
            
        return results

def compare_gpu_efficiency():
    """Compare different GPU options for training."""
    model_params = 20  # 20B model
    tokens = 300  # 300B tokens
    
    results = {}
    for gpu_name in GPU_SPECS.keys():
        calc = TrainingCostCalculator(gpu_name)
        cost = calc.calculate_training_cost(model_params, tokens)
        results[gpu_name] = {
            "total_cost": cost["total_cost_usd"],
            "gpu_hours": cost["gpu_hours"],
            "gpus_needed": cost["gpus_needed"],
            "cost_per_tflop": cost["total_cost_usd"] / (cost["total_flops"] / 1e12)
        }
        
    return results

if __name__ == "__main__":
    # Example calculations
    calc = TrainingCostCalculator("A100-40GB")
    
    print("=== PRETRAIN vs FINETUNE Cost Comparison ===\n")
    
    # GPT-20B Pretrain (full training from scratch)
    print("📊 GPT-20B PRETRAIN (300B tokens):")
    pretrain_20b = calc.calculate_training_cost(20, 300, training_type="pretrain")
    pretrain_rewards = calc.calculate_bly_rewards(20, 300, 0.10, "pretrain")
    
    print(f"  Total cost: ${pretrain_20b['total_cost_usd']:,.2f}")
    print(f"  GPU hours: {pretrain_20b['gpu_hours']:,.0f}")
    print(f"  Training days: {pretrain_20b['distributed_hours']/24:.1f}")
    print(f"  BLY Required: {pretrain_rewards['total_bly_required']:,.0f}")
    
    # GPT-20B Finetune (realistic blockchain use case)
    print(f"\n📊 GPT-20B FINETUNE (0.25 epoch, 10GB dataset):")
    finetune_20b = calc.calculate_finetune_cost(20, dataset_size_gb=10, epochs=0.25)
    finetune_rewards = calc.calculate_bly_rewards(20, finetune_20b['tokens_B'], 0.10, "finetune")
    
    print(f"  Total cost: ${finetune_20b['total_cost_usd']:,.2f}")
    print(f"  GPU hours: {finetune_20b['gpu_hours']:,.0f}")
    print(f"  Training hours: {finetune_20b['distributed_hours']:.1f}")
    print(f"  BLY Required: {finetune_rewards['total_bly_required']:,.0f}")
    print(f"  💡 Cost reduction: {(1 - finetune_20b['total_cost_usd']/pretrain_20b['total_cost_usd'])*100:.1f}%")
    
    # GPT-120B comparison
    print(f"\n📊 GPT-120B PRETRAIN (300B tokens):")
    pretrain_120b = calc.calculate_training_cost(120, 300, training_type="pretrain")
    pretrain_120b_rewards = calc.calculate_bly_rewards(120, 300, 0.10, "pretrain")
    
    print(f"  Total cost: ${pretrain_120b['total_cost_usd']:,.2f}")
    print(f"  Training days: {pretrain_120b['distributed_hours']/24:.1f}")
    print(f"  BLY Required: {pretrain_120b_rewards['total_bly_required']:,.0f}")
    
    print(f"\n📊 GPT-120B FINETUNE (0.25 epoch, 10GB dataset):")
    finetune_120b = calc.calculate_finetune_cost(120, dataset_size_gb=10, epochs=0.25)
    finetune_120b_rewards = calc.calculate_bly_rewards(120, finetune_120b['tokens_B'], 0.10, "finetune")
    
    print(f"  Total cost: ${finetune_120b['total_cost_usd']:,.2f}")
    print(f"  Training hours: {finetune_120b['distributed_hours']:.1f}")
    print(f"  BLY Required: {finetune_120b_rewards['total_bly_required']:,.0f}")
    print(f"  💡 Cost reduction: {(1 - finetune_120b['total_cost_usd']/pretrain_120b['total_cost_usd'])*100:.1f}%")
    
    print("\n=== Common Models Training Costs ===")
    common = calc.get_common_models_cost(0.10)
    for model, data in common.items():
        print(f"{model}:")
        print(f"  Cost: ${data['cost_usd']:,.2f}")
        print(f"  BLY: {data['bly_required']:,.0f}")
        print(f"  Days: {data['training_days']:.1f}")
        print()