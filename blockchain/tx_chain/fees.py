"""
EIP-1559 style fee mechanism
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class FeeConfig:
    min_base_fee: int = 1_000_000_000  # 1 gwei
    max_base_fee: int = 1_000_000_000_000  # 1000 gwei
    target_gas_used: int = 15_000_000  # 15M gas target
    max_gas_limit: int = 30_000_000  # 30M gas max
    base_fee_change_denominator: int = 8  # 12.5% max change
    elasticity_multiplier: int = 2

class EIP1559FeeMarket:
    """EIP-1559 fee calculation and management"""
    
    def __init__(self, config: FeeConfig = None):
        self.config = config or FeeConfig()
        self.current_base_fee = self.config.min_base_fee
        
    def calculate_next_base_fee(self, gas_used: int, gas_limit: int) -> int:
        """
        Calculate next block's base fee
        
        Formula:
        next_base_fee = base_fee * (1 + adjustment)
        adjustment = (gas_used - target) / target / denominator
        """
        # Handle empty blocks
        if gas_used == 0:
            gas_used = 1
            
        target = gas_limit // self.config.elasticity_multiplier
        
        if gas_used == target:
            return self.current_base_fee
        
        if gas_used > target:
            # Increase base fee
            delta = gas_used - target
            base_fee_delta = max(
                self.current_base_fee * delta // target // self.config.base_fee_change_denominator,
                1
            )
            new_base_fee = self.current_base_fee + base_fee_delta
        else:
            # Decrease base fee
            delta = target - gas_used
            base_fee_delta = self.current_base_fee * delta // target // self.config.base_fee_change_denominator
            new_base_fee = self.current_base_fee - base_fee_delta
        
        # Apply bounds
        new_base_fee = max(new_base_fee, self.config.min_base_fee)
        new_base_fee = min(new_base_fee, self.config.max_base_fee)
        
        return new_base_fee
    
    def update_base_fee(self, gas_used: int, gas_limit: int):
        """Update base fee for next block"""
        self.current_base_fee = self.calculate_next_base_fee(gas_used, gas_limit)
    
    def validate_transaction_fee(self, max_fee_per_gas: int, max_priority_fee: int) -> bool:
        """
        Validate transaction fee parameters
        
        Args:
            max_fee_per_gas: Maximum total fee per gas
            max_priority_fee: Maximum priority fee (tip) per gas
        """
        # Check max fee covers base fee
        if max_fee_per_gas < self.current_base_fee:
            return False
        
        # Check priority fee doesn't exceed max fee
        if max_priority_fee > max_fee_per_gas:
            return False
        
        return True
    
    def calculate_effective_fees(self, max_fee_per_gas: int, 
                                 max_priority_fee: int) -> tuple[int, int]:
        """
        Calculate effective base fee and priority fee
        
        Returns:
            (effective_base_fee, effective_priority_fee)
        """
        # Base fee is burned
        effective_base_fee = min(self.current_base_fee, max_fee_per_gas)
        
        # Priority fee goes to proposer
        effective_priority_fee = min(
            max_priority_fee,
            max_fee_per_gas - effective_base_fee
        )
        
        return effective_base_fee, effective_priority_fee
    
    def calculate_transaction_cost(self, gas_used: int, max_fee_per_gas: int,
                                   max_priority_fee: int) -> tuple[int, int, int]:
        """
        Calculate transaction cost breakdown
        
        Returns:
            (total_cost, burned_amount, proposer_reward)
        """
        base_fee, priority_fee = self.calculate_effective_fees(
            max_fee_per_gas, max_priority_fee
        )
        
        burned = gas_used * base_fee
        reward = gas_used * priority_fee
        total = burned + reward
        
        return total, burned, reward

class FeeAccumulator:
    """Track and distribute fees"""
    
    def __init__(self):
        self.total_burned = 0
        self.proposer_rewards = {}
        self.block_fees = {}
        
    def process_block_fees(self, block_height: int, proposer: str,
                           transactions: list, fee_market: EIP1559FeeMarket):
        """Process all fees in a block"""
        total_burned = 0
        total_reward = 0
        total_gas = 0
        
        for tx in transactions:
            gas_used = tx.get('gas_used', 21000)  # Default 21k for simple transfer
            max_fee = tx.get('max_fee_per_gas', fee_market.current_base_fee)
            max_priority = tx.get('max_priority_fee', 0)
            
            _, burned, reward = fee_market.calculate_transaction_cost(
                gas_used, max_fee, max_priority
            )
            
            total_burned += burned
            total_reward += reward
            total_gas += gas_used
        
        # Update accumulator
        self.total_burned += total_burned
        self.proposer_rewards[proposer] = self.proposer_rewards.get(proposer, 0) + total_reward
        self.block_fees[block_height] = {
            'burned': total_burned,
            'reward': total_reward,
            'gas_used': total_gas,
            'base_fee': fee_market.current_base_fee
        }
        
        # Update base fee for next block
        fee_market.update_base_fee(total_gas, fee_market.config.max_gas_limit)
        
        return total_burned, total_reward