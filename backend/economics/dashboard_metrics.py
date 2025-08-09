#!/usr/bin/env python3
"""
Economic Dashboard Metrics for Blyan Network
Real-time monitoring and alerting for token economics
"""

import time
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Economic alert."""
    level: str  # "warning" or "critical"
    metric: str
    message: str
    timestamp: float
    value: float
    threshold: float

class EconomicDashboard:
    """
    Real-time economic metrics dashboard with alerting.
    """
    
    def __init__(self):
        # Alert thresholds
        self.thresholds = {
            "demand_ratio_min": 0.60,
            "pool_balance_min": 10000,
            "pending_queue_max": 100,
            "deficit_days_max": 7,
            "burn_rate_daily_min": 100  # Minimum daily burn
        }
        
        # Metrics storage (rolling window)
        self.metrics_history = {
            "demand_ratio": deque(maxlen=168),  # 7 days hourly
            "pool_balance": deque(maxlen=168),
            "pending_rewards": deque(maxlen=168),
            "burn_rate": deque(maxlen=168),
            "inference_volume": deque(maxlen=168),
            "training_jobs": deque(maxlen=168)
        }
        
        # State tracking
        self.deficit_days = 0
        self.last_surplus_date = time.time()
        self.active_alerts: List[Alert] = []
        
    def update_metrics(
        self,
        demand_ratio: float,
        pool_balance: float,
        pending_rewards: int,
        daily_burn: float,
        inference_tokens: int,
        training_jobs: int,
        training_enabled: bool
    ):
        """
        Update dashboard with latest metrics.
        """
        timestamp = time.time()
        
        # Store metrics
        self.metrics_history["demand_ratio"].append((timestamp, demand_ratio))
        self.metrics_history["pool_balance"].append((timestamp, pool_balance))
        self.metrics_history["pending_rewards"].append((timestamp, pending_rewards))
        self.metrics_history["burn_rate"].append((timestamp, daily_burn))
        self.metrics_history["inference_volume"].append((timestamp, inference_tokens))
        self.metrics_history["training_jobs"].append((timestamp, training_jobs))
        
        # Check for alerts
        self._check_alerts(
            demand_ratio=demand_ratio,
            pool_balance=pool_balance,
            pending_rewards=pending_rewards,
            daily_burn=daily_burn,
            training_enabled=training_enabled
        )
        
        # Update deficit tracking
        if pool_balance < self.thresholds["pool_balance_min"]:
            if time.time() - self.last_surplus_date > 86400:  # 1 day
                self.deficit_days += 1
        else:
            self.deficit_days = 0
            self.last_surplus_date = time.time()
            
    def _check_alerts(
        self,
        demand_ratio: float,
        pool_balance: float,
        pending_rewards: int,
        daily_burn: float,
        training_enabled: bool
    ):
        """Check metrics against thresholds and generate alerts."""
        self.active_alerts = []
        
        # Demand ratio alert
        if demand_ratio < self.thresholds["demand_ratio_min"]:
            self.active_alerts.append(Alert(
                level="critical" if demand_ratio < 0.3 else "warning",
                metric="demand_ratio",
                message=f"Demand ratio {demand_ratio:.2f} below minimum {self.thresholds['demand_ratio_min']}",
                timestamp=time.time(),
                value=demand_ratio,
                threshold=self.thresholds["demand_ratio_min"]
            ))
            
        # Pool balance alert
        if pool_balance < self.thresholds["pool_balance_min"]:
            self.active_alerts.append(Alert(
                level="critical" if pool_balance < 1000 else "warning",
                metric="pool_balance",
                message=f"Pool balance {pool_balance:.0f} BLY below minimum",
                timestamp=time.time(),
                value=pool_balance,
                threshold=self.thresholds["pool_balance_min"]
            ))
            
        # Pending queue alert
        if pending_rewards > self.thresholds["pending_queue_max"]:
            self.active_alerts.append(Alert(
                level="warning",
                metric="pending_queue",
                message=f"{pending_rewards} rewards pending (max: {self.thresholds['pending_queue_max']})",
                timestamp=time.time(),
                value=pending_rewards,
                threshold=self.thresholds["pending_queue_max"]
            ))
            
        # Deficit days alert
        if self.deficit_days > self.thresholds["deficit_days_max"]:
            self.active_alerts.append(Alert(
                level="critical",
                metric="deficit_days",
                message=f"Deficit for {self.deficit_days} consecutive days",
                timestamp=time.time(),
                value=self.deficit_days,
                threshold=self.thresholds["deficit_days_max"]
            ))
            
        # Low burn rate alert (indicates low usage)
        if daily_burn < self.thresholds["burn_rate_daily_min"]:
            self.active_alerts.append(Alert(
                level="warning",
                metric="burn_rate",
                message=f"Daily burn {daily_burn:.0f} BLY below healthy minimum",
                timestamp=time.time(),
                value=daily_burn,
                threshold=self.thresholds["burn_rate_daily_min"]
            ))
            
    def get_dashboard_data(self) -> Dict:
        """
        Get complete dashboard data for display.
        """
        # Calculate current metrics
        current_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                current_metrics[metric_name] = history[-1][1]  # Latest value
            else:
                current_metrics[metric_name] = 0
                
        # Calculate trends (last 24h vs previous 24h)
        trends = self._calculate_trends()
        
        # Health score (0-100)
        health_score = self._calculate_health_score(current_metrics)
        
        return {
            "timestamp": time.time(),
            "current_metrics": current_metrics,
            "trends": trends,
            "health_score": health_score,
            "active_alerts": [
                {
                    "level": alert.level,
                    "metric": alert.metric,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold
                }
                for alert in self.active_alerts
            ],
            "deficit_days": self.deficit_days,
            "charts": self._prepare_chart_data()
        }
        
    def _calculate_trends(self) -> Dict[str, float]:
        """Calculate 24h trends for each metric."""
        trends = {}
        current_time = time.time()
        day_ago = current_time - 86400
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 2:
                trends[metric_name] = 0
                continue
                
            # Get values from 24h ago and now
            recent_values = [v for t, v in history if t > day_ago]
            older_values = [v for t, v in history if t <= day_ago]
            
            if recent_values and older_values:
                recent_avg = sum(recent_values) / len(recent_values)
                older_avg = sum(older_values) / len(older_values)
                
                if older_avg > 0:
                    trends[metric_name] = ((recent_avg - older_avg) / older_avg) * 100
                else:
                    trends[metric_name] = 0
            else:
                trends[metric_name] = 0
                
        return trends
        
    def _calculate_health_score(self, metrics: Dict) -> float:
        """
        Calculate overall system health score (0-100).
        """
        score = 100.0
        
        # Deduct points for poor metrics
        demand_ratio = metrics.get("demand_ratio", 1.0)
        if demand_ratio < 0.6:
            score -= (0.6 - demand_ratio) * 50  # Up to -30 points
            
        pool_balance = metrics.get("pool_balance", 10000)
        if pool_balance < 10000:
            score -= (1 - pool_balance / 10000) * 20  # Up to -20 points
            
        pending = metrics.get("pending_rewards", 0)
        if pending > 100:
            score -= min(20, pending / 10)  # Up to -20 points
            
        if self.deficit_days > 0:
            score -= min(30, self.deficit_days * 5)  # Up to -30 points
            
        return max(0, score)
        
    def _prepare_chart_data(self) -> Dict:
        """Prepare data for charts."""
        charts = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                # Convert deque to list for slicing
                history_list = list(history)
                last_24 = history_list[-24:] if len(history_list) >= 24 else history_list
                charts[metric_name] = {
                    "labels": [time.strftime("%H:%M", time.localtime(t)) for t, _ in last_24],
                    "values": [v for _, v in last_24]
                }
                
        return charts
        
    def get_summary_report(self) -> str:
        """
        Generate a text summary report.
        """
        data = self.get_dashboard_data()
        metrics = data["current_metrics"]
        trends = data["trends"]
        
        report = []
        report.append("=== BLYAN NETWORK ECONOMIC DASHBOARD ===\n")
        
        # Health Score
        health = data["health_score"]
        health_emoji = "🟢" if health > 80 else "🟡" if health > 50 else "🔴"
        report.append(f"{health_emoji} System Health: {health:.0f}/100\n")
        
        # Key Metrics
        report.append("📊 Current Metrics:")
        report.append(f"  • Demand Ratio: {metrics.get('demand_ratio', 0):.2f} ({trends.get('demand_ratio', 0):+.1f}%)")
        report.append(f"  • Pool Balance: {metrics.get('pool_balance', 0):,.0f} BLY ({trends.get('pool_balance', 0):+.1f}%)")
        report.append(f"  • Pending Rewards: {metrics.get('pending_rewards', 0)} ({trends.get('pending_rewards', 0):+.1f}%)")
        report.append(f"  • Daily Burn: {metrics.get('burn_rate', 0):,.0f} BLY ({trends.get('burn_rate', 0):+.1f}%)")
        report.append(f"  • Inference Volume: {metrics.get('inference_volume', 0)/1e6:.1f}M tokens")
        report.append(f"  • Training Jobs: {metrics.get('training_jobs', 0)}")
        
        # Alerts
        if data["active_alerts"]:
            report.append("\n⚠️ Active Alerts:")
            for alert in data["active_alerts"]:
                icon = "🔴" if alert["level"] == "critical" else "🟡"
                report.append(f"  {icon} {alert['message']}")
        else:
            report.append("\n✅ No active alerts")
            
        # Deficit tracking
        if data["deficit_days"] > 0:
            report.append(f"\n⚠️ Deficit Days: {data['deficit_days']}")
            
        return "\n".join(report)

async def demo_dashboard():
    """Demo the dashboard with simulated data."""
    dashboard = EconomicDashboard()
    
    # Simulate various scenarios
    scenarios = [
        # Healthy state
        {
            "demand_ratio": 1.2,
            "pool_balance": 50000,
            "pending_rewards": 10,
            "daily_burn": 500,
            "inference_tokens": 100_000_000,
            "training_jobs": 5,
            "training_enabled": True
        },
        # Warning state
        {
            "demand_ratio": 0.5,
            "pool_balance": 8000,
            "pending_rewards": 150,
            "daily_burn": 80,
            "inference_tokens": 50_000_000,
            "training_jobs": 2,
            "training_enabled": False
        },
        # Critical state
        {
            "demand_ratio": 0.2,
            "pool_balance": 500,
            "pending_rewards": 300,
            "daily_burn": 10,
            "inference_tokens": 10_000_000,
            "training_jobs": 0,
            "training_enabled": False
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*50}")
        print(f"Scenario {i+1}: {['Healthy', 'Warning', 'Critical'][i]}")
        print('='*50)
        
        dashboard.update_metrics(**scenario)
        print(dashboard.get_summary_report())
        
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(demo_dashboard())