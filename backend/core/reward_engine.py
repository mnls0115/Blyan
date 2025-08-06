#!/usr/bin/env python3
"""
Blyan Network Reward Engine
Implements dynamic reward calculation for learning and inference contributions
with automatic inflation control to maintain < 10% annual rate.
"""

import time
import math
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RewardMetrics:
    """Metrics for reward calculation."""
    tokens_served: int = 0
    quality_score: float = 0.0
    latency_ms: float = 0.0
    delta_loss: float = 0.0
    baseline_loss: float = 1.0

class RewardEngine:
    """
    Dynamic reward engine with inflation control.
    Automatically adjusts α (learning) and β (inference) coefficients
    to maintain target inflation rate.
    """
    
    def __init__(self, config_path: str = "config/tokenomics.yaml"):
        """Initialize reward engine with tokenomics config."""
        self.config = self._load_config(config_path)
        self.cache = {}  # Simple in-memory cache
        self.ledger = RewardLedger()  # Track minted tokens
        self.metrics = NetworkMetrics()  # Demand prediction
        
        # Initialize coefficients
        self._update_coefficients()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load tokenomics configuration."""
        path = Path(config_path)
        if not path.exists():
            # Use defaults if config not found
            return {
                "total_cap": 1_000_000_000,
                "annual_inflation": 0.10,
                "learning": {
                    "base_reward_per_1p": 20_000,
                    "min_train_reward": 0.02
                },
                "inference": {
                    "beta_init": 0.002,
                    "latency_slo_ms": 400,
                    "latency_penalty_max": 0.3,
                    "quality_factor_curve": {
                        "q90": 0.8,
                        "q95": 1.0,
                        "q99": 1.1
                    }
                },
                "controls": {
                    "reward_adjustment_interval": 3600,
                    "min_beta": 0.000001,
                    "max_alpha_multiplier": 5.0
                }
            }
            
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def adjust_coefficients(self) -> Tuple[float, float]:
        """
        Dynamically adjust α and β coefficients based on inflation target.
        Called periodically to maintain < 10% annual inflation.
        """
        # Get current year's minted tokens
        minted_this_year = self.ledger.get_minted_this_year()
        
        # Calculate remaining budget for the year
        annual_budget = self.config["total_cap"] * self.config["annual_inflation"]
        remaining_budget = annual_budget - minted_this_year
        
        # Predict demand for rest of year
        days_remaining = self._days_remaining_in_year()
        predicted_token_demand = self.metrics.predict_token_demand(days_remaining)
        predicted_learning_events = self.metrics.predict_learning_events(days_remaining)
        
        # Calculate new β (inference coefficient)
        if predicted_token_demand > 0:
            beta = max(
                self.config["controls"]["min_beta"],
                remaining_budget * 0.6 / predicted_token_demand  # 60% for inference
            )
        else:
            beta = self.config["inference"]["beta_init"]
            
        # Calculate new α (learning coefficient)
        base_reward = self.config["learning"]["base_reward_per_1p"]
        if predicted_learning_events > 0:
            alpha = min(
                base_reward * self.config["controls"]["max_alpha_multiplier"],
                remaining_budget * 0.4 / predicted_learning_events  # 40% for learning
            )
        else:
            alpha = base_reward
            
        # Cache the coefficients
        self.cache['coefficients'] = (alpha, beta)
        self.cache['coefficients_updated'] = time.time()
        
        logger.info(f"Reward coefficients adjusted: α={alpha:.2f}, β={beta:.6f}")
        logger.info(f"Inflation status: {minted_this_year/annual_budget:.1%} of annual budget used")
        
        return alpha, beta
    
    def _update_coefficients(self):
        """Update coefficients if needed based on interval."""
        interval = self.config["controls"]["reward_adjustment_interval"]
        last_update = self.cache.get('coefficients_updated', 0)
        
        if time.time() - last_update > interval:
            self.adjust_coefficients()
    
    def reward_inference(
        self,
        tokens_served: int,
        quality_score: float,
        latency_ms: float
    ) -> float:
        """
        Calculate reward for inference contribution.
        
        Args:
            tokens_served: Number of tokens generated
            quality_score: Quality percentile (0-1, where 0.95 = p95)
            latency_ms: Response latency in milliseconds
            
        Returns:
            BLY reward amount
        """
        self._update_coefficients()
        alpha, beta = self.cache['coefficients']
        
        # Quality factor from curve
        q_curve = self.config["inference"]["quality_factor_curve"]
        if quality_score >= 0.99:
            quality_factor = q_curve["q99"]
        elif quality_score >= 0.95:
            quality_factor = q_curve["q95"]
        else:
            quality_factor = q_curve["q90"]
            
        # Latency penalty
        slo = self.config["inference"]["latency_slo_ms"]
        penalty_max = self.config["inference"]["latency_penalty_max"]
        
        if latency_ms > slo:
            latency_penalty = min(penalty_max, (latency_ms / slo - 1) * 0.1)
        else:
            latency_penalty = 0
            
        # Calculate reward
        reward = beta * tokens_served * quality_factor * (1 - latency_penalty)
        
        # Log the reward
        self.ledger.record_reward('inference', reward)
        
        return max(0, reward)
    
    def reward_training(
        self,
        delta_loss: float,
        baseline_loss: float = 1.0
    ) -> float:
        """
        Calculate reward for training/learning contribution.
        
        Args:
            delta_loss: Improvement in loss (baseline - new)
            baseline_loss: Baseline loss before training
            
        Returns:
            BLY reward amount
        """
        self._update_coefficients()
        alpha, _ = self.cache['coefficients']
        
        # Calculate improvement percentage
        if baseline_loss > 0:
            improvement_percent = (delta_loss / baseline_loss) * 100
        else:
            improvement_percent = 0
            
        # Calculate reward
        min_reward = self.config["learning"]["min_train_reward"]
        reward = max(min_reward, alpha * improvement_percent)
        
        # Apply quality thresholds for bonus
        thresholds = self.config["learning"]["quality_thresholds"]
        if improvement_percent >= 5.0:
            reward *= 2.0  # Excellent improvement
        elif improvement_percent >= 2.0:
            reward *= 1.5  # Good improvement
            
        # Log the reward
        self.ledger.record_reward('training', reward)
        
        return reward
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Get current reward statistics."""
        self._update_coefficients()
        alpha, beta = self.cache['coefficients']
        
        return {
            "coefficients": {
                "alpha": alpha,
                "beta": beta,
                "last_updated": self.cache.get('coefficients_updated', 0)
            },
            "inflation": {
                "annual_target": self.config["annual_inflation"],
                "current_rate": self.ledger.get_inflation_rate(),
                "minted_this_year": self.ledger.get_minted_this_year(),
                "minted_total": self.ledger.get_total_minted()
            },
            "distribution": self.ledger.get_distribution_stats()
        }
    
    def _days_remaining_in_year(self) -> int:
        """Calculate days remaining in current year."""
        from datetime import datetime
        now = datetime.now()
        year_end = datetime(now.year, 12, 31)
        return (year_end - now).days


class RewardLedger:
    """Tracks all reward distributions for inflation control."""
    
    def __init__(self):
        self.rewards = []
        self.yearly_totals = {}
        
    def record_reward(self, reward_type: str, amount: float):
        """Record a reward distribution."""
        from datetime import datetime
        now = datetime.now()
        
        record = {
            "type": reward_type,
            "amount": amount,
            "timestamp": now.timestamp(),
            "year": now.year
        }
        
        self.rewards.append(record)
        
        # Update yearly total
        if now.year not in self.yearly_totals:
            self.yearly_totals[now.year] = {"inference": 0, "training": 0}
        self.yearly_totals[now.year][reward_type] += amount
        
    def get_minted_this_year(self) -> float:
        """Get total tokens minted in current year."""
        from datetime import datetime
        current_year = datetime.now().year
        
        if current_year in self.yearly_totals:
            year_data = self.yearly_totals[current_year]
            return year_data.get("inference", 0) + year_data.get("training", 0)
        return 0
        
    def get_total_minted(self) -> float:
        """Get total tokens ever minted."""
        total = 0
        for year_data in self.yearly_totals.values():
            total += year_data.get("inference", 0) + year_data.get("training", 0)
        return total
        
    def get_inflation_rate(self) -> float:
        """Calculate current annualized inflation rate."""
        from datetime import datetime
        
        # Get this year's minting
        minted = self.get_minted_this_year()
        
        # Calculate portion of year elapsed
        now = datetime.now()
        year_start = datetime(now.year, 1, 1)
        days_elapsed = (now - year_start).days + 1
        days_in_year = 365 + (1 if now.year % 4 == 0 else 0)
        
        # Annualize the current rate
        if days_elapsed > 0:
            annualized = (minted / days_elapsed) * days_in_year
            return annualized / 1_000_000_000  # As percentage of total cap
        return 0
        
    def get_distribution_stats(self) -> Dict[str, float]:
        """Get distribution statistics by type."""
        from datetime import datetime
        current_year = datetime.now().year
        
        if current_year in self.yearly_totals:
            return self.yearly_totals[current_year]
        return {"inference": 0, "training": 0}


class NetworkMetrics:
    """Predicts network demand for coefficient adjustment."""
    
    def __init__(self):
        self.historical_data = []
        
    def predict_token_demand(self, days: int) -> float:
        """Predict token demand for next N days."""
        # Simple prediction based on recent average
        # In production, use proper time series forecasting
        recent_avg_daily = 1_000_000  # 1M tokens per day baseline
        return recent_avg_daily * days
        
    def predict_learning_events(self, days: int) -> float:
        """Predict learning events for next N days."""
        # Simple prediction
        # In production, track actual training submissions
        avg_daily_improvements = 50  # 50 improvement events per day
        return avg_daily_improvements * days


# Singleton instance
_reward_engine = None

def get_reward_engine() -> RewardEngine:
    """Get or create the reward engine singleton."""
    global _reward_engine
    if _reward_engine is None:
        _reward_engine = RewardEngine()
    return _reward_engine

if __name__ == "__main__":
    # Test the reward engine
    engine = get_reward_engine()
    
    # Test inference reward
    reward = engine.reward_inference(
        tokens_served=1000,
        quality_score=0.95,
        latency_ms=350
    )
    print(f"Inference reward for 1k tokens @ p95 quality: {reward:.3f} BLY")
    
    # Test training reward
    reward = engine.reward_training(
        delta_loss=0.005,  # 0.5% improvement
        baseline_loss=1.0
    )
    print(f"Training reward for 0.5% improvement: {reward:.1f} BLY")
    
    # Show stats
    stats = engine.get_reward_stats()
    print(f"\nReward Engine Stats:")
    print(f"  α coefficient: {stats['coefficients']['alpha']:.2f}")
    print(f"  β coefficient: {stats['coefficients']['beta']:.6f}")
    print(f"  Current inflation: {stats['inflation']['current_rate']:.2%}")