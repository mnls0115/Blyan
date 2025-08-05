#!/usr/bin/env python3
"""Predictive scheduler for proactive resource allocation based on demand patterns."""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json


@dataclass
class DemandPrediction:
    """Predicted resource demand for a time window."""
    timestamp: float
    predicted_arrival_rate: float
    confidence: float
    recommended_warm_slots: int
    recommended_learning_util: float


@dataclass
class DemandPattern:
    """Historical demand pattern."""
    hour_of_day: int
    day_of_week: int  # 0=Monday, 6=Sunday
    avg_arrival_rate: float
    peak_arrival_rate: float
    std_deviation: float
    sample_count: int


class SimplePredictiveScheduler:
    """Rule-based predictive scheduler using time patterns and recent trends."""
    
    def __init__(self, 
                 history_window_hours: int = 24 * 7,  # 1 week
                 pattern_update_interval: int = 3600,  # 1 hour
                 max_samples_per_pattern: int = 100):
        
        self.history_window_hours = history_window_hours
        self.pattern_update_interval = pattern_update_interval
        self.max_samples_per_pattern = max_samples_per_pattern
        
        # Historical data storage
        self.arrival_history: deque = deque(maxlen=10000)  # (timestamp, arrival_rate)
        self.demand_patterns: Dict[Tuple[int, int], DemandPattern] = {}  # (hour, day_of_week) -> pattern
        
        # Recent trend tracking (last 15 minutes)
        self.recent_samples: deque = deque(maxlen=15)  # Recent arrival rates
        self.last_pattern_update = 0
        
        # Peak hours configuration (can be learned)
        self.peak_hours = {9, 10, 11, 14, 15, 16, 19, 20, 21}  # Common peak hours
        self.weekend_factor = 0.7  # Weekend traffic typically 70% of weekday
        
    def record_arrival_rate(self, arrival_rate: float, timestamp: float = None):
        """Record current arrival rate for pattern learning."""
        if timestamp is None:
            timestamp = time.time()
            
        self.arrival_history.append((timestamp, arrival_rate))
        self.recent_samples.append(arrival_rate)
        
        # Update patterns periodically
        if timestamp - self.last_pattern_update > self.pattern_update_interval:
            self._update_demand_patterns()
            self.last_pattern_update = timestamp
    
    def predict_demand(self, 
                      forecast_minutes: int = 15, 
                      current_time: float = None) -> DemandPrediction:
        """Predict demand for the next forecast window."""
        if current_time is None:
            current_time = time.time()
            
        forecast_time = current_time + (forecast_minutes * 60)
        dt = datetime.fromtimestamp(forecast_time)
        
        # Get base prediction from historical patterns
        base_prediction = self._get_pattern_prediction(dt.hour, dt.weekday())
        
        # Adjust with recent trend
        trend_adjustment = self._get_trend_adjustment()
        
        # Combine predictions
        predicted_rate = base_prediction * (1 + trend_adjustment)
        
        # Calculate confidence based on pattern reliability and recent stability
        confidence = self._calculate_confidence(dt.hour, dt.weekday())
        
        # Recommend resource allocation
        warm_slots, learning_util = self._recommend_allocation(predicted_rate, dt.hour, dt.weekday())
        
        return DemandPrediction(
            timestamp=forecast_time,
            predicted_arrival_rate=predicted_rate,
            confidence=confidence,
            recommended_warm_slots=warm_slots,
            recommended_learning_util=learning_util
        )
    
    def _get_pattern_prediction(self, hour: int, day_of_week: int) -> float:
        """Get base prediction from historical patterns."""
        pattern_key = (hour, day_of_week)
        
        if pattern_key in self.demand_patterns:
            pattern = self.demand_patterns[pattern_key]
            return pattern.avg_arrival_rate
        
        # Fallback to heuristics if no historical data
        return self._heuristic_prediction(hour, day_of_week)
    
    def _heuristic_prediction(self, hour: int, day_of_week: int) -> float:
        """Heuristic-based prediction when no historical data available."""
        base_rate = 10.0  # Default baseline
        
        # Peak hour multiplier
        if hour in self.peak_hours:
            base_rate *= 2.5
        elif 6 <= hour <= 23:  # Business hours
            base_rate *= 1.5
        else:  # Night hours
            base_rate *= 0.3
            
        # Weekend factor
        if day_of_week >= 5:  # Saturday/Sunday
            base_rate *= self.weekend_factor
            
        return base_rate
    
    def _get_trend_adjustment(self) -> float:
        """Calculate trend adjustment based on recent samples."""
        if len(self.recent_samples) < 3:
            return 0.0
            
        recent_rates = list(self.recent_samples)
        
        # Simple linear trend over last few samples
        x = np.arange(len(recent_rates))
        if len(recent_rates) > 1 and np.std(recent_rates) > 0:
            slope = np.polyfit(x, recent_rates, 1)[0]
            # Normalize slope to reasonable adjustment range (-0.5 to +0.5)
            trend_adjustment = np.clip(slope / np.mean(recent_rates), -0.5, 0.5)
        else:
            trend_adjustment = 0.0
            
        return trend_adjustment
    
    def _calculate_confidence(self, hour: int, day_of_week: int) -> float:
        """Calculate confidence in prediction."""
        pattern_key = (hour, day_of_week)
        
        if pattern_key not in self.demand_patterns:
            return 0.5  # Medium confidence for heuristic predictions
            
        pattern = self.demand_patterns[pattern_key]
        
        # Confidence based on sample count and pattern stability
        sample_confidence = min(pattern.sample_count / 20.0, 1.0)  # Max confidence at 20+ samples
        
        # Stability confidence based on coefficient of variation
        if pattern.avg_arrival_rate > 0:
            cv = pattern.std_deviation / pattern.avg_arrival_rate
            stability_confidence = max(0.1, 1.0 - cv)  # Lower CV = higher confidence
        else:
            stability_confidence = 0.5
            
        # Recent trend stability
        recent_stability = 1.0
        if len(self.recent_samples) >= 5:
            recent_std = np.std(list(self.recent_samples))
            recent_mean = np.mean(list(self.recent_samples))
            if recent_mean > 0:
                recent_cv = recent_std / recent_mean
                recent_stability = max(0.3, 1.0 - recent_cv)
        
        # Combined confidence
        confidence = (sample_confidence * 0.4 + stability_confidence * 0.4 + recent_stability * 0.2)
        return min(confidence, 0.95)  # Cap at 95%
    
    def _recommend_allocation(self, predicted_rate: float, hour: int, day_of_week: int) -> Tuple[int, float]:
        """Recommend warm slots and learning utilization based on predicted demand."""
        
        # Warm slots recommendation
        if predicted_rate > 30:      # High demand
            warm_slots = 4
        elif predicted_rate > 15:    # Medium demand  
            warm_slots = 3
        elif predicted_rate > 5:     # Low-medium demand
            warm_slots = 2
        else:                        # Low demand
            warm_slots = 1
            
        # Learning utilization recommendation
        if predicted_rate > 25:      # Very high demand expected
            learning_util = 0.2      # Reserve most GPU for inference
        elif predicted_rate > 15:    # High demand
            learning_util = 0.4
        elif predicted_rate > 8:     # Medium demand
            learning_util = 0.6
        else:                        # Low demand
            learning_util = 0.8      # Learning can use most GPU
            
        # Night hours - favor learning
        if hour < 6 or hour > 23:
            learning_util = min(learning_util + 0.2, 0.9)
            
        # Weekend adjustment - typically lower demand
        if day_of_week >= 5:
            learning_util = min(learning_util + 0.1, 0.9)
            warm_slots = max(warm_slots - 1, 1)
            
        return warm_slots, learning_util
    
    def _update_demand_patterns(self):
        """Update demand patterns from historical data."""
        # Group samples by (hour, day_of_week)
        pattern_samples = defaultdict(list)
        
        cutoff_time = time.time() - (self.history_window_hours * 3600)
        
        for timestamp, arrival_rate in self.arrival_history:
            if timestamp < cutoff_time:
                continue
                
            dt = datetime.fromtimestamp(timestamp)
            pattern_key = (dt.hour, dt.weekday())
            pattern_samples[pattern_key].append(arrival_rate)
        
        # Update patterns
        for pattern_key, samples in pattern_samples.items():
            if len(samples) < 3:  # Need minimum samples
                continue
                
            # Limit samples to avoid memory growth
            if len(samples) > self.max_samples_per_pattern:
                samples = samples[-self.max_samples_per_pattern:]
                
            hour, day_of_week = pattern_key
            samples_array = np.array(samples)
            
            self.demand_patterns[pattern_key] = DemandPattern(
                hour_of_day=hour,
                day_of_week=day_of_week,
                avg_arrival_rate=float(np.mean(samples_array)),
                peak_arrival_rate=float(np.max(samples_array)), 
                std_deviation=float(np.std(samples_array)),
                sample_count=len(samples)
            )
    
    def get_pattern_insights(self) -> Dict[str, any]:
        """Get insights about learned demand patterns."""
        if not self.demand_patterns:
            return {"status": "no_patterns_learned"}
            
        # Find peak hours across the week
        peak_patterns = sorted(
            [(k, p.avg_arrival_rate) for k, p in self.demand_patterns.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Calculate average by hour of day
        hourly_avg = defaultdict(list)
        for (hour, day), pattern in self.demand_patterns.items():
            hourly_avg[hour].append(pattern.avg_arrival_rate)
            
        hourly_patterns = {
            hour: {
                "avg_rate": np.mean(rates),
                "peak_rate": max(rates),
                "sample_days": len(rates)
            }
            for hour, rates in hourly_avg.items()
        }
        
        return {
            "total_patterns": len(self.demand_patterns),
            "peak_time_patterns": [
                {
                    "hour": k[0], 
                    "day_of_week": k[1], 
                    "avg_rate": rate,
                    "day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k[1]]
                }
                for k, rate in peak_patterns
            ],
            "hourly_patterns": hourly_patterns,
            "recent_trend": {
                "last_15_min_avg": np.mean(list(self.recent_samples)) if self.recent_samples else 0,
                "samples": len(self.recent_samples)
            }
        }
    
    def save_patterns(self, filepath: str):
        """Save learned patterns to file."""
        data = {
            "patterns": {
                f"{h}_{d}": {
                    "hour": p.hour_of_day,
                    "day": p.day_of_week, 
                    "avg_rate": p.avg_arrival_rate,
                    "peak_rate": p.peak_arrival_rate,
                    "std_dev": p.std_deviation,
                    "samples": p.sample_count
                }
                for (h, d), p in self.demand_patterns.items()
            },
            "config": {
                "history_window_hours": self.history_window_hours,
                "peak_hours": list(self.peak_hours),
                "weekend_factor": self.weekend_factor
            },
            "saved_at": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_patterns(self, filepath: str) -> bool:
        """Load patterns from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Load patterns
            for key, pattern_data in data["patterns"].items():
                pattern_key = (pattern_data["hour"], pattern_data["day"])
                self.demand_patterns[pattern_key] = DemandPattern(
                    hour_of_day=pattern_data["hour"],
                    day_of_week=pattern_data["day"],
                    avg_arrival_rate=pattern_data["avg_rate"],
                    peak_arrival_rate=pattern_data["peak_rate"],
                    std_deviation=pattern_data["std_dev"],
                    sample_count=pattern_data["samples"]
                )
                
            # Load config if available
            if "config" in data:
                config = data["config"]
                self.peak_hours = set(config.get("peak_hours", self.peak_hours))
                self.weekend_factor = config.get("weekend_factor", self.weekend_factor)
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load patterns: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    scheduler = SimplePredictiveScheduler()
    
    # Simulate some historical data
    print("=== Simulating historical demand patterns ===")
    current_time = time.time()
    
    # Simulate a week of data with realistic patterns
    for day in range(7):
        for hour in range(24):
            timestamp = current_time - (7-day)*24*3600 + hour*3600
            
            # Simulate realistic demand patterns
            if 9 <= hour <= 11 or 14 <= hour <= 16 or 19 <= hour <= 21:  # Peak hours
                base_rate = 25 + np.random.normal(0, 5)
            elif 6 <= hour <= 23:  # Business hours
                base_rate = 15 + np.random.normal(0, 3) 
            else:  # Night
                base_rate = 3 + np.random.normal(0, 1)
                
            # Weekend factor
            if day >= 5:
                base_rate *= 0.7
                
            base_rate = max(0, base_rate)
            scheduler.record_arrival_rate(base_rate, timestamp)
    
    # Make predictions
    print("\n=== Making predictions ===")
    for forecast_min in [15, 30, 60]:
        prediction = scheduler.predict_demand(forecast_min)
        forecast_dt = datetime.fromtimestamp(prediction.timestamp)
        
        print(f"Forecast +{forecast_min}min ({forecast_dt.strftime('%H:%M %a')}):")
        print(f"  Predicted rate: {prediction.predicted_arrival_rate:.1f} req/s")
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Recommended warm slots: {prediction.recommended_warm_slots}")
        print(f"  Recommended learning util: {prediction.recommended_learning_util:.1%}")
        print()
    
    # Pattern insights
    print("=== Pattern insights ===")
    insights = scheduler.get_pattern_insights()
    print(f"Learned patterns: {insights['total_patterns']}")
    print("\nTop peak periods:")
    for pattern in insights["peak_time_patterns"][:5]:
        print(f"  {pattern['day_name']} {pattern['hour']:02d}:00 - {pattern['avg_rate']:.1f} req/s")