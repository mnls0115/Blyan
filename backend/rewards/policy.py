#!/usr/bin/env python3
"""
Canonical Reward Policy Implementation
Single source of truth for all reward calculations in the Blyan network.
All formulas are pure functions with no side effects.
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Centralized configuration for reward calculations."""
    # Per-unit rates
    per_1k_tokens_bly: float = 1.0
    per_1pct_improvement_bly: float = 500.0
    per_validation_task_bly: float = 10.0
    per_gb_dataset_bly: float = 100.0
    
    # Quality ranges
    inference_quality_min: float = 0.5
    inference_quality_max: float = 1.5
    
    # Learning factors
    difficulty_min: float = 1.0
    difficulty_max: float = 3.0
    applicability_min: float = 0.8
    applicability_max: float = 1.2
    
    # Daily budget
    daily_budget_bly: float = 273_972.0
    
    @classmethod
    def from_yaml(cls, path: str = "config/reward_policy.yaml") -> "RewardConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file {path} not found, using defaults")
            return cls()
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            per_1k_tokens_bly=data['rates']['per_1k_tokens_bly'],
            per_1pct_improvement_bly=data['rates']['per_1pct_improvement_bly'],
            per_validation_task_bly=data['rates']['per_validation_task_bly'],
            per_gb_dataset_bly=data['rates']['per_gb_dataset_bly'],
            inference_quality_min=data['inference_quality_multiplier']['min'],
            inference_quality_max=data['inference_quality_multiplier']['max'],
            difficulty_min=data['learning_quality_factors']['difficulty_factor']['min'],
            difficulty_max=data['learning_quality_factors']['difficulty_factor']['max'],
            applicability_min=data['learning_quality_factors']['applicability_factor']['min'],
            applicability_max=data['learning_quality_factors']['applicability_factor']['max'],
            daily_budget_bly=data['daily_budget_bly']
        )


# Pure calculation functions - no side effects
def calc_inference_bly(
    tokens: int,
    quality: float,
    rate: float = 1.0,
    config: Optional[RewardConfig] = None
) -> float:
    """
    Calculate BLY rewards for inference work.
    
    Args:
        tokens: Number of tokens processed
        quality: Quality score (0.5 to 1.5)
        rate: Base rate per 1k tokens (default from config)
        config: Reward configuration
    
    Returns:
        BLY reward amount
    
    Properties:
        - Monotonic in tokens (more tokens = more reward)
        - Monotonic in quality (higher quality = more reward)
        - Bounded by quality multipliers
    """
    if config is None:
        config = RewardConfig()
    
    # Validate inputs
    if tokens < 0:
        raise ValueError(f"Tokens must be non-negative, got {tokens}")
    
    # Clamp quality to valid range
    quality = max(config.inference_quality_min, 
                 min(config.inference_quality_max, quality))
    
    # Calculate reward: (tokens / 1000) * rate * quality
    base_reward = (tokens / 1000.0) * rate * quality
    
    return round(base_reward, 4)


def calc_learning_bly(
    base_pool_day: float,
    improvement_pct: float,
    difficulty: float = 1.5,
    applicability: float = 1.0,
    hourslice_share: float = 1/24,
    config: Optional[RewardConfig] = None
) -> float:
    """
    Calculate BLY rewards for learning contributions.
    
    Args:
        base_pool_day: Daily learning budget pool
        improvement_pct: Percentage improvement achieved (e.g., 2.5 for 2.5%)
        difficulty: Difficulty factor (1.0 to 3.0)
        applicability: Applicability factor (0.8 to 1.2)
        hourslice_share: Fraction of daily pool for this hour
        config: Reward configuration
    
    Returns:
        BLY reward amount
    
    Properties:
        - Monotonic in improvement_pct
        - Scaled by difficulty and applicability
        - Bounded by factor ranges
    """
    if config is None:
        config = RewardConfig()
    
    # Validate inputs
    if improvement_pct < 0:
        raise ValueError(f"Improvement must be non-negative, got {improvement_pct}")
    
    # Clamp factors to valid ranges
    difficulty = max(config.difficulty_min,
                    min(config.difficulty_max, difficulty))
    applicability = max(config.applicability_min,
                       min(config.applicability_max, applicability))
    
    # Base reward per 1% improvement
    base_per_pct = config.per_1pct_improvement_bly
    
    # Calculate reward with multipliers
    reward = (
        base_per_pct * 
        improvement_pct * 
        difficulty * 
        applicability * 
        hourslice_share
    )
    
    # Apply daily pool constraint
    max_from_pool = base_pool_day * hourslice_share
    reward = min(reward, max_from_pool)
    
    return round(reward, 4)


def calc_validation_bly(
    n_tasks: int,
    per_task_rate: float = 10.0,
    complexity_multiplier: float = 1.0,
    config: Optional[RewardConfig] = None
) -> float:
    """
    Calculate BLY rewards for validation work.
    
    Args:
        n_tasks: Number of validation tasks completed
        per_task_rate: BLY per task
        complexity_multiplier: Multiplier for complex validations
        config: Reward configuration
    
    Returns:
        BLY reward amount
    """
    if config is None:
        config = RewardConfig()
    
    if n_tasks < 0:
        raise ValueError(f"Tasks must be non-negative, got {n_tasks}")
    
    reward = n_tasks * per_task_rate * complexity_multiplier
    return round(reward, 4)


def calc_dataset_bly(
    gb_size: float,
    quality_score: float,
    diversity_score: float = 1.0,
    cap: float = 10000.0,
    config: Optional[RewardConfig] = None
) -> float:
    """
    Calculate BLY rewards for dataset contributions.
    
    Args:
        gb_size: Size of dataset in GB
        quality_score: Quality score (0 to 1)
        diversity_score: Diversity/uniqueness score (0 to 2)
        cap: Maximum reward cap
        config: Reward configuration
    
    Returns:
        BLY reward amount
    """
    if config is None:
        config = RewardConfig()
    
    if gb_size < 0:
        raise ValueError(f"Size must be non-negative, got {gb_size}")
    
    # Clamp scores
    quality_score = max(0, min(1, quality_score))
    diversity_score = max(0, min(2, diversity_score))
    
    # Calculate base reward
    base_reward = gb_size * config.per_gb_dataset_bly
    
    # Apply multipliers
    reward = base_reward * quality_score * diversity_score
    
    # Apply cap
    reward = min(reward, cap)
    
    return round(reward, 4)


def estimate_daily_rewards(
    expected_tokens: int,
    expected_improvements: float,
    expected_validations: int,
    expected_dataset_gb: float,
    config: Optional[RewardConfig] = None
) -> Dict[str, float]:
    """
    Estimate daily rewards across all categories.
    
    Returns:
        Dictionary with estimated rewards by category
    """
    if config is None:
        config = RewardConfig()
    
    estimates = {
        'inference': calc_inference_bly(expected_tokens, quality=1.0, config=config),
        'learning': calc_learning_bly(
            config.daily_budget_bly * 0.35,  # 35% allocation
            expected_improvements,
            difficulty=1.5,
            applicability=1.0,
            hourslice_share=1.0,  # Full day
            config=config
        ),
        'validation': calc_validation_bly(expected_validations, config=config),
        'dataset': calc_dataset_bly(expected_dataset_gb, 0.9, 1.0, config=config),
    }
    
    estimates['total'] = sum(estimates.values())
    estimates['utilization'] = estimates['total'] / config.daily_budget_bly
    
    return estimates


# Property testing helpers
def verify_monotonicity():
    """Verify that reward functions are monotonic in their primary inputs."""
    config = RewardConfig()
    
    # Test inference monotonicity in tokens
    for i in range(0, 10000, 1000):
        r1 = calc_inference_bly(i, 1.0, config=config)
        r2 = calc_inference_bly(i + 1000, 1.0, config=config)
        assert r2 >= r1, f"Inference not monotonic: {i} -> {i+1000}"
    
    # Test learning monotonicity in improvement
    for i in range(0, 10):
        r1 = calc_learning_bly(10000, i, config=config)
        r2 = calc_learning_bly(10000, i + 1, config=config)
        assert r2 >= r1, f"Learning not monotonic: {i}% -> {i+1}%"
    
    logger.info("✅ Monotonicity verified")


def verify_bounds():
    """Verify that rewards stay within configured bounds."""
    config = RewardConfig()
    
    # Test quality bounds for inference
    r_min = calc_inference_bly(1000, 0.3, config=config)  # Below min
    r_mid = calc_inference_bly(1000, 1.0, config=config)
    r_max = calc_inference_bly(1000, 2.0, config=config)  # Above max
    
    assert r_min == calc_inference_bly(1000, config.inference_quality_min, config=config)
    assert r_max == calc_inference_bly(1000, config.inference_quality_max, config=config)
    
    # Test factor bounds for learning
    r_min_diff = calc_learning_bly(10000, 1, difficulty=0.5, config=config)
    r_max_diff = calc_learning_bly(10000, 1, difficulty=5.0, config=config)
    
    assert r_min_diff == calc_learning_bly(10000, 1, difficulty=config.difficulty_min, config=config)
    assert r_max_diff == calc_learning_bly(10000, 1, difficulty=config.difficulty_max, config=config)
    
    logger.info("✅ Bounds verified")


if __name__ == "__main__":
    # Run verification tests
    verify_monotonicity()
    verify_bounds()
    
    # Example calculations
    config = RewardConfig.from_yaml()
    
    print("\n📊 Example Reward Calculations:")
    print(f"Inference (100k tokens, quality 1.1): {calc_inference_bly(100000, 1.1, config=config)} BLY")
    print(f"Learning (2% improvement, difficulty 2.0): {calc_learning_bly(95000, 2, 2.0, config=config)} BLY")
    print(f"Validation (50 tasks): {calc_validation_bly(50, config=config)} BLY")
    print(f"Dataset (10 GB, quality 0.9): {calc_dataset_bly(10, 0.9, config=config)} BLY")