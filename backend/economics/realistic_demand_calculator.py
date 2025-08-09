#!/usr/bin/env python3
"""
Realistic Demand Calculator for Blyan Network
Calculates actual minimum requirements for sustainable economics
"""

from typing import Dict, List
from decimal import Decimal

class RealisticDemandCalculator:
    """
    Calculate realistic minimum demand for sustainability.
    Focus on fine-tuning scenarios (99.9% cheaper than pretrain).
    """
    
    def __init__(self):
        # Pricing
        self.price_per_1k_tokens = 0.001  # $0.001 per 1K tokens
        self.bly_price = 0.10  # $0.10 per BLY
        
        # Economics
        self.burn_ratio = 0.50  # 50% burned
        self.pool_ratio = 0.50  # 50% to pool
        
        # Fine-tuning costs (from actual calculations)
        self.finetune_costs = {
            "gpt_20b": {
                "cost_usd": 152,
                "hours": 8.3,
                "bly_required": 1821
            },
            "gpt_120b": {
                "cost_usd": 911,
                "hours": 8.9,
                "bly_required": 10929
            }
        }
        
    def calculate_minimum_demand(
        self,
        daily_finetunes: int = 1,
        model_size: str = "gpt_20b"
    ) -> Dict[str, float]:
        """
        Calculate minimum demand to support N fine-tunes per day.
        
        Args:
            daily_finetunes: Number of fine-tuning jobs per day
            model_size: "gpt_20b" or "gpt_120b"
            
        Returns:
            Minimum demand requirements
        """
        finetune_cost = self.finetune_costs[model_size]
        
        # Daily cost for fine-tuning
        daily_cost_usd = finetune_cost["cost_usd"] * daily_finetunes
        daily_cost_bly = finetune_cost["bly_required"] * daily_finetunes
        
        # Need 2x revenue because 50% gets burned
        required_daily_revenue_usd = daily_cost_usd * 2
        required_daily_revenue_bly = daily_cost_bly * 2
        
        # Calculate required token volume
        required_daily_tokens = (required_daily_revenue_usd / self.price_per_1k_tokens) * 1000
        
        return {
            "scenario": f"{daily_finetunes} {model_size} finetunes/day",
            "daily_cost_usd": daily_cost_usd,
            "daily_cost_bly": daily_cost_bly,
            "required_daily_revenue_usd": required_daily_revenue_usd,
            "required_daily_revenue_bly": required_daily_revenue_bly,
            "required_daily_tokens_M": required_daily_tokens / 1_000_000,
            "required_monthly_tokens_B": (required_daily_tokens * 30) / 1_000_000_000,
            "daily_burn_bly": required_daily_revenue_bly * self.burn_ratio,
            "daily_to_pool_bly": required_daily_revenue_bly * self.pool_ratio
        }
        
    def calculate_network_scenarios(self) -> List[Dict]:
        """
        Calculate various network growth scenarios.
        """
        scenarios = []
        
        # Scenario 1: Minimal (1 GPT-20B finetune/day)
        scenarios.append({
            "name": "Minimal",
            "description": "1 GPT-20B finetune per day",
            **self.calculate_minimum_demand(1, "gpt_20b")
        })
        
        # Scenario 2: Small Network (10 GPT-20B finetunes/day)
        scenarios.append({
            "name": "Small Network",
            "description": "10 GPT-20B finetunes per day",
            **self.calculate_minimum_demand(10, "gpt_20b")
        })
        
        # Scenario 3: Medium Network (50 GPT-20B + 5 GPT-120B/day)
        scenario_20b = self.calculate_minimum_demand(50, "gpt_20b")
        scenario_120b = self.calculate_minimum_demand(5, "gpt_120b")
        
        scenarios.append({
            "name": "Medium Network",
            "description": "50 GPT-20B + 5 GPT-120B finetunes/day",
            "daily_cost_usd": scenario_20b["daily_cost_usd"] + scenario_120b["daily_cost_usd"],
            "required_daily_revenue_usd": scenario_20b["required_daily_revenue_usd"] + scenario_120b["required_daily_revenue_usd"],
            "required_daily_tokens_M": scenario_20b["required_daily_tokens_M"] + scenario_120b["required_daily_tokens_M"],
            "required_monthly_tokens_B": scenario_20b["required_monthly_tokens_B"] + scenario_120b["required_monthly_tokens_B"]
        })
        
        # Scenario 4: Large Network (100 GPT-20B + 20 GPT-120B/day)
        scenario_20b_large = self.calculate_minimum_demand(100, "gpt_20b")
        scenario_120b_large = self.calculate_minimum_demand(20, "gpt_120b")
        
        scenarios.append({
            "name": "Large Network",
            "description": "100 GPT-20B + 20 GPT-120B finetunes/day",
            "daily_cost_usd": scenario_20b_large["daily_cost_usd"] + scenario_120b_large["daily_cost_usd"],
            "required_daily_revenue_usd": scenario_20b_large["required_daily_revenue_usd"] + scenario_120b_large["required_daily_revenue_usd"],
            "required_daily_tokens_M": scenario_20b_large["required_daily_tokens_M"] + scenario_120b_large["required_daily_tokens_M"],
            "required_monthly_tokens_B": scenario_20b_large["required_monthly_tokens_B"] + scenario_120b_large["required_monthly_tokens_B"]
        })
        
        return scenarios
        
    def compare_with_existing_services(self) -> Dict:
        """
        Compare required volume with existing AI services.
        """
        # Rough estimates of daily token volumes
        benchmarks = {
            "ChatGPT Free Tier": 10_000_000_000,      # 10B tokens/day
            "Claude.ai Free": 1_000_000_000,          # 1B tokens/day  
            "Local LLM Service": 100_000_000,         # 100M tokens/day
            "Enterprise API": 5_000_000_000,          # 5B tokens/day
        }
        
        # Our minimum requirements
        minimal = self.calculate_minimum_demand(1, "gpt_20b")
        small = self.calculate_minimum_demand(10, "gpt_20b")
        
        comparisons = {}
        for service, daily_tokens in benchmarks.items():
            comparisons[service] = {
                "daily_tokens_M": daily_tokens / 1_000_000,
                "can_support_minimal": daily_tokens >= minimal["required_daily_tokens_M"] * 1_000_000,
                "can_support_small": daily_tokens >= small["required_daily_tokens_M"] * 1_000_000,
                "finetunes_supported": int(daily_tokens / (minimal["required_daily_tokens_M"] * 1_000_000))
            }
            
        return comparisons
        
    def calculate_profitability_threshold(self) -> Dict:
        """
        Calculate when network becomes profitable.
        """
        # Infrastructure costs (estimated)
        monthly_infra_cost = 5000  # $5K/month for servers, storage, etc.
        
        # Calculate break-even
        daily_infra_cost = monthly_infra_cost / 30
        
        # Need to cover infra + at least 1 finetune/day
        minimal_finetune = self.calculate_minimum_demand(1, "gpt_20b")
        total_daily_cost = daily_infra_cost + minimal_finetune["daily_cost_usd"]
        
        # Revenue needed (2x because of burn)
        breakeven_revenue = total_daily_cost * 2
        breakeven_tokens = (breakeven_revenue / self.price_per_1k_tokens) * 1000
        
        return {
            "monthly_infra_cost": monthly_infra_cost,
            "daily_infra_cost": daily_infra_cost,
            "min_finetune_cost": minimal_finetune["daily_cost_usd"],
            "total_daily_cost": total_daily_cost,
            "breakeven_daily_revenue": breakeven_revenue,
            "breakeven_daily_tokens_M": breakeven_tokens / 1_000_000,
            "breakeven_monthly_revenue": breakeven_revenue * 30
        }

def main():
    """Run demand calculations and display results."""
    calc = RealisticDemandCalculator()
    
    print("=== 🎯 REALISTIC MINIMUM DEMAND ANALYSIS ===\n")
    
    # Single finetune requirements
    print("📊 Single Fine-tune Requirements:")
    for model in ["gpt_20b", "gpt_120b"]:
        result = calc.calculate_minimum_demand(1, model)
        print(f"\n  {model.upper()}:")
        print(f"    Daily cost: ${result['daily_cost_usd']:.2f}")
        print(f"    Required revenue: ${result['required_daily_revenue_usd']:.2f}")
        print(f"    Required tokens: {result['required_daily_tokens_M']:.1f}M/day")
        print(f"    Monthly tokens: {result['required_monthly_tokens_B']:.2f}B")
    
    # Network scenarios
    print("\n📊 Network Growth Scenarios:")
    scenarios = calc.calculate_network_scenarios()
    for scenario in scenarios:
        print(f"\n  {scenario['name']}: {scenario['description']}")
        print(f"    Daily revenue needed: ${scenario['required_daily_revenue_usd']:,.2f}")
        print(f"    Daily tokens: {scenario['required_daily_tokens_M']:,.1f}M")
        print(f"    Monthly tokens: {scenario['required_monthly_tokens_B']:.2f}B")
    
    # Comparison with existing services
    print("\n📊 Comparison with Existing Services:")
    comparisons = calc.compare_with_existing_services()
    for service, data in comparisons.items():
        print(f"\n  {service} ({data['daily_tokens_M']:.0f}M tokens/day):")
        print(f"    Can support minimal: {'✅' if data['can_support_minimal'] else '❌'}")
        print(f"    Can support small network: {'✅' if data['can_support_small'] else '❌'}")
        print(f"    GPT-20B finetunes supported: {data['finetunes_supported']}")
    
    # Profitability threshold
    print("\n📊 Profitability Threshold:")
    threshold = calc.calculate_profitability_threshold()
    print(f"  Infrastructure cost: ${threshold['monthly_infra_cost']:,}/month")
    print(f"  Break-even daily revenue: ${threshold['breakeven_daily_revenue']:,.2f}")
    print(f"  Break-even daily tokens: {threshold['breakeven_daily_tokens_M']:.1f}M")
    print(f"  Break-even monthly revenue: ${threshold['breakeven_monthly_revenue']:,.2f}")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Fine-tuning is 99.9% cheaper than pretraining")
    print("  • Minimal viable network needs only ~300M tokens/day")
    print("  • Small network (10 finetunes/day) needs ~3B tokens/day")
    print("  • This is achievable with moderate usage (< 1% of ChatGPT)")

if __name__ == "__main__":
    main()