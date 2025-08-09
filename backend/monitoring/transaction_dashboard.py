#!/usr/bin/env python3
"""
Transaction Monitoring Dashboard
Real-time metrics for chat transaction health
"""

import time
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

class TransactionMetrics(BaseModel):
    """Transaction health metrics."""
    timestamp: datetime
    total_transactions: int
    successful: int
    failed: int
    success_rate: float
    avg_duration_ms: int
    p95_duration_ms: int
    active_transactions: int
    
    # Failure breakdown
    failure_reasons: Dict[str, int]
    
    # Detailed metrics
    ttl_expiry_rate: float
    token_mismatch_rate: float
    quota_exceeded_rate: float
    balance_insufficient_rate: float
    challenge_pass_rate: float

class RejectionAnalytics(BaseModel):
    """Analytics for request rejections."""
    period: str  # "1h", "24h", "7d"
    total_requests: int
    total_rejections: int
    rejection_rate: float
    
    # Rejection reasons
    reasons: Dict[str, int]
    
    # Time series data
    hourly_rejections: List[Dict[str, Any]]
    
    # Challenge metrics
    captcha_presented: int
    captcha_solved: int
    captcha_success_rate: float
    pow_presented: int
    pow_solved: int
    pow_success_rate: float

class TransactionMonitor:
    """Monitor transaction health and collect metrics."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # Metric keys
        self.METRICS_KEY = "tx_metrics"
        self.FAILURE_REASONS_KEY = "tx_failure_reasons"
        self.REJECTION_LOG_KEY = "tx_rejections"
        self.CHALLENGE_METRICS_KEY = "challenge_metrics"
        self.TTL_METRICS_KEY = "ttl_metrics"
    
    async def connect(self):
        """Initialize Redis connection."""
        if not self.redis_client:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )
    
    async def record_transaction(
        self,
        transaction_id: str,
        success: bool,
        duration_ms: int,
        failure_reason: str = None
    ):
        """Record transaction outcome."""
        await self.connect()
        
        # Update counters
        if success:
            await self.redis_client.hincrby(self.METRICS_KEY, "success_count", 1)
        else:
            await self.redis_client.hincrby(self.METRICS_KEY, "failure_count", 1)
            
            if failure_reason:
                await self.redis_client.hincrby(self.FAILURE_REASONS_KEY, failure_reason, 1)
        
        # Record duration
        await self.redis_client.lpush("tx_durations", duration_ms)
        await self.redis_client.ltrim("tx_durations", 0, 9999)  # Keep last 10k
        
        # Record timestamp for time series
        timestamp = int(time.time())
        await self.redis_client.zadd(
            "tx_timeline",
            {f"{transaction_id}:{success}": timestamp}
        )
    
    async def record_rejection(
        self,
        reason: str,
        challenge_type: str = None,
        challenge_solved: bool = False
    ):
        """Record request rejection."""
        await self.connect()
        
        timestamp = int(time.time())
        
        # Record rejection
        rejection_data = {
            "timestamp": timestamp,
            "reason": reason,
            "challenge_type": challenge_type,
            "challenge_solved": challenge_solved
        }
        
        await self.redis_client.lpush(
            self.REJECTION_LOG_KEY,
            json.dumps(rejection_data)
        )
        await self.redis_client.ltrim(self.REJECTION_LOG_KEY, 0, 99999)  # Keep last 100k
        
        # Update challenge metrics
        if challenge_type:
            await self.redis_client.hincrby(
                self.CHALLENGE_METRICS_KEY,
                f"{challenge_type}_presented",
                1
            )
            
            if challenge_solved:
                await self.redis_client.hincrby(
                    self.CHALLENGE_METRICS_KEY,
                    f"{challenge_type}_solved",
                    1
                )
    
    async def record_ttl_event(self, event_type: str):
        """Record TTL-related events."""
        await self.connect()
        
        # event_type: "quote_created", "quote_used", "quote_expired"
        await self.redis_client.hincrby(self.TTL_METRICS_KEY, event_type, 1)
    
    async def get_transaction_metrics(self) -> TransactionMetrics:
        """Get comprehensive transaction metrics."""
        await self.connect()
        
        # Get basic counters
        metrics = await self.redis_client.hgetall(self.METRICS_KEY)
        failure_reasons = await self.redis_client.hgetall(self.FAILURE_REASONS_KEY)
        
        # Calculate rates
        success = int(metrics.get("success_count", 0))
        failure = int(metrics.get("failure_count", 0))
        total = success + failure
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        # Get duration stats
        durations = await self.redis_client.lrange("tx_durations", 0, -1)
        durations = [int(d) for d in durations]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            sorted_durations = sorted(durations)
            p95_index = int(len(durations) * 0.95)
            p95_duration = sorted_durations[p95_index] if p95_index < len(durations) else sorted_durations[-1]
        else:
            avg_duration = 0
            p95_duration = 0
        
        # Calculate specific failure rates
        ttl_failures = int(failure_reasons.get("quote_expired", 0))
        token_failures = int(failure_reasons.get("token_mismatch", 0))
        quota_failures = int(failure_reasons.get("quota_exceeded", 0))
        balance_failures = int(failure_reasons.get("insufficient_balance", 0))
        
        ttl_expiry_rate = (ttl_failures / total * 100) if total > 0 else 0
        token_mismatch_rate = (token_failures / total * 100) if total > 0 else 0
        quota_exceeded_rate = (quota_failures / total * 100) if total > 0 else 0
        balance_insufficient_rate = (balance_failures / total * 100) if total > 0 else 0
        
        # Get challenge metrics
        challenge_metrics = await self.redis_client.hgetall(self.CHALLENGE_METRICS_KEY)
        
        captcha_presented = int(challenge_metrics.get("captcha_presented", 0))
        captcha_solved = int(challenge_metrics.get("captcha_solved", 0))
        captcha_pass_rate = (captcha_solved / captcha_presented * 100) if captcha_presented > 0 else 0
        
        pow_presented = int(challenge_metrics.get("proof_of_work_presented", 0))
        pow_solved = int(challenge_metrics.get("proof_of_work_solved", 0))
        pow_pass_rate = (pow_solved / pow_presented * 100) if pow_presented > 0 else 0
        
        challenge_pass_rate = ((captcha_solved + pow_solved) / (captcha_presented + pow_presented) * 100) \
            if (captcha_presented + pow_presented) > 0 else 0
        
        return TransactionMetrics(
            timestamp=datetime.now(),
            total_transactions=total,
            successful=success,
            failed=failure,
            success_rate=success_rate,
            avg_duration_ms=int(avg_duration),
            p95_duration_ms=int(p95_duration),
            active_transactions=0,  # Would need to track from transaction manager
            failure_reasons=failure_reasons,
            ttl_expiry_rate=ttl_expiry_rate,
            token_mismatch_rate=token_mismatch_rate,
            quota_exceeded_rate=quota_exceeded_rate,
            balance_insufficient_rate=balance_insufficient_rate,
            challenge_pass_rate=challenge_pass_rate
        )
    
    async def get_rejection_analytics(self, period: str = "1h") -> RejectionAnalytics:
        """Get rejection analytics for specified period."""
        await self.connect()
        
        # Calculate time window
        now = time.time()
        if period == "1h":
            start_time = now - 3600
        elif period == "24h":
            start_time = now - 86400
        elif period == "7d":
            start_time = now - 604800
        else:
            start_time = now - 3600  # Default to 1 hour
        
        # Get rejections in time window
        all_rejections = await self.redis_client.lrange(self.REJECTION_LOG_KEY, 0, -1)
        
        rejections_in_window = []
        rejection_reasons = defaultdict(int)
        hourly_buckets = defaultdict(int)
        
        for rejection_json in all_rejections:
            try:
                rejection = json.loads(rejection_json)
                if rejection["timestamp"] >= start_time:
                    rejections_in_window.append(rejection)
                    rejection_reasons[rejection["reason"]] += 1
                    
                    # Bucket by hour
                    hour_bucket = int(rejection["timestamp"] / 3600) * 3600
                    hourly_buckets[hour_bucket] += 1
                    
            except json.JSONDecodeError:
                continue
        
        # Calculate hourly time series
        hourly_rejections = []
        current_hour = int(start_time / 3600) * 3600
        while current_hour <= now:
            hourly_rejections.append({
                "timestamp": datetime.fromtimestamp(current_hour).isoformat(),
                "rejections": hourly_buckets.get(current_hour, 0)
            })
            current_hour += 3600
        
        # Get challenge metrics
        challenge_metrics = await self.redis_client.hgetall(self.CHALLENGE_METRICS_KEY)
        
        captcha_presented = int(challenge_metrics.get("captcha_presented", 0))
        captcha_solved = int(challenge_metrics.get("captcha_solved", 0))
        captcha_success_rate = (captcha_solved / captcha_presented * 100) if captcha_presented > 0 else 0
        
        pow_presented = int(challenge_metrics.get("proof_of_work_presented", 0))
        pow_solved = int(challenge_metrics.get("proof_of_work_solved", 0))
        pow_success_rate = (pow_solved / pow_presented * 100) if pow_presented > 0 else 0
        
        # Calculate overall metrics
        total_rejections = len(rejections_in_window)
        
        # Estimate total requests (would need actual tracking)
        metrics = await self.redis_client.hgetall(self.METRICS_KEY)
        total_requests = int(metrics.get("success_count", 0)) + int(metrics.get("failure_count", 0))
        
        rejection_rate = (total_rejections / total_requests * 100) if total_requests > 0 else 0
        
        return RejectionAnalytics(
            period=period,
            total_requests=total_requests,
            total_rejections=total_rejections,
            rejection_rate=rejection_rate,
            reasons=dict(rejection_reasons),
            hourly_rejections=hourly_rejections,
            captcha_presented=captcha_presented,
            captcha_solved=captcha_solved,
            captcha_success_rate=captcha_success_rate,
            pow_presented=pow_presented,
            pow_solved=pow_solved,
            pow_success_rate=pow_success_rate
        )

# Global monitor instance
_monitor = None

def get_transaction_monitor() -> TransactionMonitor:
    """Get or create monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = TransactionMonitor()
    return _monitor

# API Endpoints
@router.get("/transactions/metrics", response_model=TransactionMetrics)
async def get_transaction_metrics():
    """Get real-time transaction metrics."""
    monitor = get_transaction_monitor()
    return await monitor.get_transaction_metrics()

@router.get("/transactions/rejections", response_model=RejectionAnalytics)
async def get_rejection_analytics(period: str = "1h"):
    """Get rejection analytics for specified period."""
    if period not in ["1h", "24h", "7d"]:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 1h, 24h, or 7d")
    
    monitor = get_transaction_monitor()
    return await monitor.get_rejection_analytics(period)

@router.get("/transactions/health")
async def get_transaction_health():
    """Get overall transaction system health."""
    monitor = get_transaction_monitor()
    metrics = await monitor.get_transaction_metrics()
    
    # Define health thresholds
    is_healthy = (
        metrics.success_rate > 95 and
        metrics.avg_duration_ms < 500 and
        metrics.ttl_expiry_rate < 5 and
        metrics.challenge_pass_rate > 85
    )
    
    health_status = "healthy" if is_healthy else "degraded"
    
    issues = []
    if metrics.success_rate <= 95:
        issues.append(f"Low success rate: {metrics.success_rate:.1f}%")
    if metrics.avg_duration_ms >= 500:
        issues.append(f"High latency: {metrics.avg_duration_ms}ms")
    if metrics.ttl_expiry_rate >= 5:
        issues.append(f"High TTL expiry rate: {metrics.ttl_expiry_rate:.1f}%")
    if metrics.challenge_pass_rate <= 85:
        issues.append(f"Low challenge pass rate: {metrics.challenge_pass_rate:.1f}%")
    
    return {
        "status": health_status,
        "metrics": {
            "success_rate": f"{metrics.success_rate:.1f}%",
            "avg_latency_ms": metrics.avg_duration_ms,
            "p95_latency_ms": metrics.p95_duration_ms,
            "active_transactions": metrics.active_transactions
        },
        "issues": issues,
        "timestamp": metrics.timestamp
    }