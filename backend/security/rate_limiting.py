#!/usr/bin/env python3
"""
PoL-Based Rate Limiting System
Enterprise-grade API protection that enhances rather than competes with AI workloads.
"""

import time
import json
import hashlib
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from fastapi import HTTPException
from fastapi.requests import Request

@dataclass
class UserStats:
    """User reputation and usage statistics."""
    user_id: str
    reputation_level: str = "newbie"  # newbie, trusted, expert
    uploads_today: int = 0
    inference_requests_hour: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_reset_time: float = 0.0
    pol_contributions: int = 0  # Number of PoL contributions made
    
    def reset_daily_counters(self):
        """Reset daily counters if a day has passed."""
        current_day = int(time.time() // 86400)
        last_day = int(self.last_reset_time // 86400)
        
        if current_day > last_day:
            self.uploads_today = 0
            self.last_reset_time = time.time()
    
    def reset_hourly_counters(self):
        """Reset hourly counters if an hour has passed."""
        current_hour = int(time.time() // 3600)
        last_hour = int(self.last_reset_time // 3600)
        
        if current_hour > last_hour:
            self.inference_requests_hour = 0


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    reason: str = ""
    challenge_type: Optional[str] = None
    challenge_description: Optional[str] = None
    bypass_available: bool = False
    retry_after: Optional[int] = None


@dataclass
class ContributionProof:
    """Proof of AI contribution for quota bypass."""
    expert_hash: str
    performance_improvement: float
    contribution_type: str
    timestamp: float
    user_id: str
    proof_hash: str


class PoLBasedRateLimiter:
    """Rate limiter that uses AI contribution instead of economic barriers."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/rate_limiting")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_quotas = {
            "newbie": {
                "uploads_per_day": 20,
                "inference_per_hour": 100,
                "promotion_threshold": 3  # 3 successes to become trusted
            },
            "trusted": {
                "uploads_per_day": 200,
                "inference_per_hour": 1000,
                "promotion_threshold": 10  # 10 successes to become expert
            },
            "expert": {
                "uploads_per_day": 1000,
                "inference_per_hour": 10000,
                "promotion_threshold": float('inf')  # Already at top level
            }
        }
        
        self.user_stats_cache: Dict[str, UserStats] = {}
        self._load_user_stats()
    
    def _get_user_id(self, request: Request) -> str:
        """Extract user ID from request (API key, IP, etc.)."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Fallback to IP address
        client_ip = request.client.host
        return hashlib.sha256(f"ip_{client_ip}".encode()).hexdigest()[:16]
    
    def _load_user_stats(self):
        """Load user statistics from persistent storage."""
        stats_file = self.storage_dir / "user_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                    for user_id, stats_dict in data.items():
                        self.user_stats_cache[user_id] = UserStats(**stats_dict)
            except Exception as e:
                print(f"Warning: Failed to load user stats: {e}")
    
    def _save_user_stats(self):
        """Save user statistics to persistent storage."""
        stats_file = self.storage_dir / "user_stats.json"
        try:
            data = {
                user_id: asdict(stats) 
                for user_id, stats in self.user_stats_cache.items()
            }
            with open(stats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save user stats: {e}")
    
    def get_user_stats(self, user_id: str) -> UserStats:
        """Get or create user statistics."""
        if user_id not in self.user_stats_cache:
            self.user_stats_cache[user_id] = UserStats(
                user_id=user_id,
                last_reset_time=time.time()
            )
        
        stats = self.user_stats_cache[user_id]
        stats.reset_daily_counters()
        stats.reset_hourly_counters()
        return stats
    
    def check_rate_limit(self, request: Request, action: str) -> RateLimitResult:
        """Check if request should be rate limited."""
        user_id = self._get_user_id(request)
        user_stats = self.get_user_stats(user_id)
        quota_config = self.user_quotas[user_stats.reputation_level]
        
        # Check daily upload quota
        if action == "upload":
            if user_stats.uploads_today >= quota_config["uploads_per_day"]:
                return RateLimitResult(
                    allowed=False,
                    reason=f"Daily upload quota exceeded ({quota_config['uploads_per_day']})",
                    challenge_type="prove_ai_contribution",
                    challenge_description="Validate 3 expert blocks or contribute 1 improved expert to increase quota",
                    bypass_available=True,
                    retry_after=self._seconds_until_daily_reset()
                )
        
        # Check hourly inference quota
        elif action == "inference":
            if user_stats.inference_requests_hour >= quota_config["inference_per_hour"]:
                return RateLimitResult(
                    allowed=False,
                    reason=f"Hourly inference quota exceeded ({quota_config['inference_per_hour']})",
                    challenge_type="prove_ai_contribution",
                    challenge_description="Contribute AI expertise to increase inference quota",
                    bypass_available=True,
                    retry_after=self._seconds_until_hourly_reset()
                )
        
        # Check for suspicious behavior patterns
        suspicious_patterns = self._detect_suspicious_patterns(user_stats)
        if suspicious_patterns:
            return RateLimitResult(
                allowed=False,
                reason=f"Suspicious activity detected: {suspicious_patterns}",
                challenge_type="manual_review",
                challenge_description="Account flagged for manual review due to suspicious patterns",
                bypass_available=False,
                retry_after=3600  # 1 hour timeout
            )
        
        return RateLimitResult(allowed=True)
    
    def _detect_suspicious_patterns(self, stats: UserStats) -> Optional[str]:
        """Detect suspicious behavioral patterns."""
        patterns = []
        
        # High failure rate
        total_attempts = stats.successful_uploads + stats.failed_uploads
        if total_attempts > 10 and stats.failed_uploads / total_attempts > 0.8:
            patterns.append("high_failure_rate")
        
        # Too many consecutive failures
        if stats.consecutive_failures >= 5:
            patterns.append("consecutive_failures")
        
        # Rapid-fire requests (would need additional tracking)
        # This is a placeholder for more sophisticated timing analysis
        
        return ", ".join(patterns) if patterns else None
    
    def pol_challenge_bypass(self, user_id: str, contribution_proof: ContributionProof) -> bool:
        """Allow quota bypass through AI contribution instead of payment."""
        if self.validate_contribution_proof(contribution_proof):
            user_stats = self.get_user_stats(user_id)
            
            # Increase quota temporarily based on contribution quality
            improvement = contribution_proof.performance_improvement
            if improvement >= 0.05:  # 5% improvement
                multiplier = 3.0
            elif improvement >= 0.02:  # 2% improvement  
                multiplier = 2.0
            else:  # 1% improvement
                multiplier = 1.5
            
            self.increase_temporary_quota(user_id, multiplier)
            user_stats.pol_contributions += 1
            self._save_user_stats()
            return True
        
        return False
    
    def validate_contribution_proof(self, proof: ContributionProof) -> bool:
        """Validate AI contribution proof for quota bypass."""
        # Verify proof hash integrity
        expected_hash = hashlib.sha256(
            f"{proof.expert_hash}{proof.performance_improvement}{proof.user_id}".encode()
        ).hexdigest()
        
        if proof.proof_hash != expected_hash:
            return False
        
        # Check minimum improvement threshold
        if proof.performance_improvement < 0.01:  # 1% minimum improvement
            return False
        
        # Check proof is recent (within last hour)
        if time.time() - proof.timestamp > 3600:
            return False
        
        return True
    
    def increase_temporary_quota(self, user_id: str, multiplier: float):
        """Temporarily increase user quota based on AI contribution."""
        user_stats = self.get_user_stats(user_id)
        current_quota = self.user_quotas[user_stats.reputation_level]
        
        # Reset counters and apply multiplier
        user_stats.uploads_today = max(0, user_stats.uploads_today - int(current_quota["uploads_per_day"] * (multiplier - 1.0)))
        user_stats.inference_requests_hour = max(0, user_stats.inference_requests_hour - int(current_quota["inference_per_hour"] * (multiplier - 1.0)))
        
        print(f"💡 Quota increased for user {user_id[:8]} by {multiplier}x due to AI contribution")
    
    def record_action(self, request: Request, action: str, success: bool):
        """Record user action for reputation tracking."""
        user_id = self._get_user_id(request)
        user_stats = self.get_user_stats(user_id)
        
        if action == "upload":
            user_stats.uploads_today += 1
            if success:
                user_stats.successful_uploads += 1
                user_stats.consecutive_successes += 1
                user_stats.consecutive_failures = 0
                
                # Check for promotion
                self._check_promotion(user_stats)
            else:
                user_stats.failed_uploads += 1
                user_stats.consecutive_failures += 1
                user_stats.consecutive_successes = 0
                
                # Check for demotion
                self._check_demotion(user_stats)
        
        elif action == "inference":
            user_stats.inference_requests_hour += 1
        
        self._save_user_stats()
    
    def _check_promotion(self, stats: UserStats):
        """Check if user should be promoted to higher reputation level."""
        quota_config = self.user_quotas[stats.reputation_level]
        
        if stats.consecutive_successes >= quota_config["promotion_threshold"]:
            if stats.reputation_level == "newbie":
                stats.reputation_level = "trusted"
                print(f"🎉 User {stats.user_id[:8]} promoted to trusted contributor")
            elif stats.reputation_level == "trusted":
                stats.reputation_level = "expert"
                print(f"🎉 User {stats.user_id[:8]} promoted to expert contributor")
    
    def _check_demotion(self, stats: UserStats):
        """Check if user should be demoted due to poor performance."""
        if stats.consecutive_failures >= 5:
            if stats.reputation_level in ["trusted", "expert"]:
                stats.reputation_level = "newbie"
                print(f"⚠️ User {stats.user_id[:8]} demoted to newbie due to consecutive failures")
    
    def _seconds_until_daily_reset(self) -> int:
        """Calculate seconds until daily quota reset."""
        current_time = time.time()
        tomorrow = int((current_time // 86400) + 1) * 86400
        return int(tomorrow - current_time)
    
    def _seconds_until_hourly_reset(self) -> int:
        """Calculate seconds until hourly quota reset."""
        current_time = time.time()
        next_hour = int((current_time // 3600) + 1) * 3600
        return int(next_hour - current_time)
    
    async def __call__(self, request: Request, action: str = "general"):
        """FastAPI middleware interface."""
        result = self.check_rate_limit(request, action)
        
        if not result.allowed:
            if result.bypass_available:
                # Return 429 with bypass information
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "reason": result.reason,
                        "challenge_type": result.challenge_type,
                        "challenge_description": result.challenge_description,
                        "bypass_endpoint": "/auth/pol_challenge",
                        "retry_after": result.retry_after
                    }
                )
            else:
                # Hard block with no bypass
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "reason": result.reason,
                        "retry_after": result.retry_after
                    }
                )
    
    def get_stats_summary(self) -> Dict:
        """Get rate limiting statistics summary."""
        total_users = len(self.user_stats_cache)
        reputation_counts = {"newbie": 0, "trusted": 0, "expert": 0}
        total_contributions = 0
        
        for stats in self.user_stats_cache.values():
            reputation_counts[stats.reputation_level] += 1
            total_contributions += stats.pol_contributions
        
        return {
            "total_users": total_users,
            "reputation_distribution": reputation_counts,
            "total_pol_contributions": total_contributions,
            "quota_config": self.user_quotas
        }


# Global rate limiter instance
rate_limiter = PoLBasedRateLimiter()


def create_contribution_proof(user_id: str, expert_hash: str, performance_improvement: float) -> ContributionProof:
    """Create a contribution proof for quota bypass."""
    timestamp = time.time()
    proof_hash = hashlib.sha256(f"{expert_hash}{performance_improvement}{user_id}".encode()).hexdigest()
    
    return ContributionProof(
        expert_hash=expert_hash,
        performance_improvement=performance_improvement,
        contribution_type="pol_proof",
        timestamp=timestamp,
        user_id=user_id,
        proof_hash=proof_hash
    )


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting with tier-based limits."""
    
    def __init__(self, default_limit: int = 60, premium_limit: int = 300, enterprise_limit: int = 1000):
        self.rate_limiter = PoLBasedRateLimiter()
        self.tier_limits = {
            "basic": default_limit,
            "premium": premium_limit,
            "enterprise": enterprise_limit,
            "unlimited": float('inf')
        }
        
    async def __call__(self, request: Request, call_next):
        """Rate limiting middleware."""
        from backend.security.api_auth import get_api_key_info
        
        # Get API key info if available
        key_info = getattr(request.state, "api_key_info", None)
        
        if key_info:
            # Use tier-based rate limiting
            tier = key_info.rate_limit_tier
            limit = self.tier_limits.get(tier, 60)
            
            # Track requests per minute  
            user_id = key_info.key_id
            # Create mock request for rate limiter
            from fastapi import Request
            mock_request = type('MockRequest', (), {
                'client': type('client', (), {'host': '127.0.0.1'}),
                'headers': {'user-agent': 'api'},
                'state': type('state', (), {'api_key_info': key_info})()
            })()
            result = self.rate_limiter.check_rate_limit(mock_request, "inference")
            
            if not result.allowed and limit != float('inf'):
                # Apply tier-specific limit
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "tier": tier,
                        "limit": limit,
                        "retry_after": result.retry_after or 60
                    }
                )
        
        # Process request
        response = await call_next(request)
        return response