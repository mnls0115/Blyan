#!/usr/bin/env python3
"""
Free Tier and User Limits Management System
Single Source of Truth (SSOT) for user quotas and limits using Redis

### FEATURES ###
- Adaptive free tier limits based on trust level
- Anti-abuse protection with multi-factor detection
- Progressive trust system with contribution rewards
- Redis-based SSOT for all user state management
- Automatic daily/hourly limit resets
"""

import time
import hashlib
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

# Redis instance for user limits (SSOT)
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=5,  # Dedicated DB for free tier management
    decode_responses=True
)

class TrustLevel(Enum):
    """User trust levels with associated privileges."""
    NEWCOMER = 1      # New users, minimal privileges
    VERIFIED = 2      # Email verified, basic privileges  
    CONTRIBUTOR = 3   # Made contributions, enhanced privileges
    TRUSTED = 4       # Established contributor, high privileges
    EXPERT = 5        # Top contributor, maximum privileges

@dataclass
class UserLimits:
    """User limits and quota information."""
    address: str
    trust_level: int
    contribution_score: float
    
    # Daily limits
    free_requests_per_day: int
    free_requests_remaining: int
    daily_reset_timestamp: float
    
    # Hourly limits (burst protection)
    requests_per_hour: int
    hourly_requests_remaining: int
    hourly_reset_timestamp: float
    
    # Usage tracking
    total_requests_made: int
    successful_validations: int
    datasets_contributed: int
    experts_improved: int
    
    # Anti-abuse tracking
    suspicion_score: float
    consecutive_failures: int
    rate_limit_violations: int
    last_request_timestamp: float
    
    # Credits and rewards
    earned_credits: float
    bonus_requests_available: int

class FreeTierManager:
    """Manages free tier limits and user quotas with Redis SSOT."""
    
    # Base limits per trust level
    TRUST_LEVEL_LIMITS = {
        TrustLevel.NEWCOMER: {
            "daily_free": 5,
            "hourly_burst": 10,
            "max_input_tokens": 2000,
            "max_output_tokens": 500
        },
        TrustLevel.VERIFIED: {
            "daily_free": 15,
            "hourly_burst": 25,
            "max_input_tokens": 4000,
            "max_output_tokens": 1000
        },
        TrustLevel.CONTRIBUTOR: {
            "daily_free": 50,
            "hourly_burst": 75,
            "max_input_tokens": 8000,
            "max_output_tokens": 2000
        },
        TrustLevel.TRUSTED: {
            "daily_free": 150,
            "hourly_burst": 200,
            "max_input_tokens": 16000,
            "max_output_tokens": 4000
        },
        TrustLevel.EXPERT: {
            "daily_free": 500,
            "hourly_burst": 600,
            "max_input_tokens": 32000,
            "max_output_tokens": 8000
        }
    }
    
    def __init__(self):
        """Initialize FreeTierManager with Redis connection."""
        self.redis = redis_client
        
    def get_user_key(self, address: str) -> str:
        """Generate Redis key for user limits."""
        if not address:
            address = "anonymous"
        return f"user_limits:{address.lower()}"
    
    def get_user_limits(self, address: str) -> UserLimits:
        """Get user limits from Redis SSOT, creating default if not exists."""
        if not address:
            address = "anonymous"
        user_key = self.get_user_key(address)
        user_data = self.redis.hgetall(user_key)
        
        if not user_data:
            # Create new user with default limits
            return self._create_new_user(address)
        
        # Parse data from Redis
        try:
            limits = UserLimits(
                address=address.lower(),
                trust_level=int(user_data.get("trust_level", 1)),
                contribution_score=float(user_data.get("contribution_score", 0)),
                free_requests_per_day=int(user_data.get("free_requests_per_day", 5)),
                free_requests_remaining=int(user_data.get("free_requests_remaining", 5)),
                daily_reset_timestamp=float(user_data.get("daily_reset_timestamp", 0)),
                requests_per_hour=int(user_data.get("requests_per_hour", 10)),
                hourly_requests_remaining=int(user_data.get("hourly_requests_remaining", 10)),
                hourly_reset_timestamp=float(user_data.get("hourly_reset_timestamp", 0)),
                total_requests_made=int(user_data.get("total_requests_made", 0)),
                successful_validations=int(user_data.get("successful_validations", 0)),
                datasets_contributed=int(user_data.get("datasets_contributed", 0)),
                experts_improved=int(user_data.get("experts_improved", 0)),
                suspicion_score=float(user_data.get("suspicion_score", 0)),
                consecutive_failures=int(user_data.get("consecutive_failures", 0)),
                rate_limit_violations=int(user_data.get("rate_limit_violations", 0)),
                last_request_timestamp=float(user_data.get("last_request_timestamp", 0)),
                earned_credits=float(user_data.get("earned_credits", 0)),
                bonus_requests_available=int(user_data.get("bonus_requests_available", 0))
            )
            
            # Check if limits need reset
            limits = self._check_and_reset_limits(limits)
            
            return limits
            
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse user data for {address}: {e}")
            return self._create_new_user(address)
    
    def _create_new_user(self, address: str) -> UserLimits:
        """Create new user with default newcomer limits."""
        if not address:
            address = "anonymous"
        trust_level = TrustLevel.NEWCOMER
        base_limits = self.TRUST_LEVEL_LIMITS[trust_level]
        current_time = time.time()
        
        limits = UserLimits(
            address=address.lower(),
            trust_level=trust_level.value,
            contribution_score=0.0,
            free_requests_per_day=base_limits["daily_free"],
            free_requests_remaining=base_limits["daily_free"],
            daily_reset_timestamp=self._get_next_daily_reset(),
            requests_per_hour=base_limits["hourly_burst"],
            hourly_requests_remaining=base_limits["hourly_burst"],
            hourly_reset_timestamp=self._get_next_hourly_reset(),
            total_requests_made=0,
            successful_validations=0,
            datasets_contributed=0,
            experts_improved=0,
            suspicion_score=0.0,
            consecutive_failures=0,
            rate_limit_violations=0,
            last_request_timestamp=0.0,
            earned_credits=0.0,
            bonus_requests_available=0
        )
        
        # Save to Redis
        self._save_user_limits(limits)
        
        logger.info(f"Created new user {address} with trust level {trust_level.name}")
        return limits
    
    def _check_and_reset_limits(self, limits: UserLimits) -> UserLimits:
        """Check and reset daily/hourly limits if expired."""
        current_time = time.time()
        updated = False
        
        # Daily reset
        if current_time >= limits.daily_reset_timestamp:
            trust_level = TrustLevel(limits.trust_level)
            base_limits = self.TRUST_LEVEL_LIMITS[trust_level]
            
            # Reset daily limits with potential bonuses
            bonus_from_credits = min(10, int(limits.earned_credits // 10))  # 1 bonus per 10 credits
            
            limits.free_requests_per_day = base_limits["daily_free"] + bonus_from_credits
            limits.free_requests_remaining = limits.free_requests_per_day
            limits.daily_reset_timestamp = self._get_next_daily_reset()
            
            # Decay suspicion score daily
            limits.suspicion_score = max(0, limits.suspicion_score * 0.8)
            
            updated = True
            
        # Hourly reset
        if current_time >= limits.hourly_reset_timestamp:
            trust_level = TrustLevel(limits.trust_level)
            base_limits = self.TRUST_LEVEL_LIMITS[trust_level]
            
            limits.requests_per_hour = base_limits["hourly_burst"]
            limits.hourly_requests_remaining = limits.requests_per_hour
            limits.hourly_reset_timestamp = self._get_next_hourly_reset()
            
            updated = True
        
        if updated:
            self._save_user_limits(limits)
            
        return limits
    
    def _save_user_limits(self, limits: UserLimits):
        """Save user limits to Redis SSOT."""
        user_key = self.get_user_key(limits.address)
        user_data = asdict(limits)
        
        # Convert all values to strings for Redis
        redis_data = {k: str(v) for k, v in user_data.items()}
        
        self.redis.hset(user_key, mapping=redis_data)
        
        # Set expiry for inactive users (90 days)
        self.redis.expire(user_key, 90 * 24 * 3600)
    
    def can_make_request(self, address: str, input_tokens: int, output_tokens_est: int) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if user can make a request within their limits.
        
        Returns:
            - allowed: Whether request is allowed
            - reason: Reason if denied
            - limits_info: Current user limits for client display
        """
        if not address:
            address = "anonymous"
        limits = self.get_user_limits(address)
        
        # Check token limits
        trust_level = TrustLevel(limits.trust_level)
        token_limits = self.TRUST_LEVEL_LIMITS[trust_level]
        
        if input_tokens > token_limits["max_input_tokens"]:
            return False, f"Input too long ({input_tokens} tokens, max {token_limits['max_input_tokens']})", self._get_limits_info(limits)
        
        if output_tokens_est > token_limits["max_output_tokens"]:
            return False, f"Output estimate too high ({output_tokens_est} tokens, max {token_limits['max_output_tokens']})", self._get_limits_info(limits)
        
        # Check daily free requests
        if limits.free_requests_remaining <= 0 and limits.bonus_requests_available <= 0:
            return False, "Daily free requests exhausted", self._get_limits_info(limits)
        
        # Check hourly burst limits
        if limits.hourly_requests_remaining <= 0:
            return False, "Hourly request limit exceeded", self._get_limits_info(limits)
        
        # Check suspicion score
        if limits.suspicion_score > 0.8:
            return False, "Account flagged for suspicious activity", self._get_limits_info(limits)
        
        # Check rate limiting (prevent rapid fire requests)
        current_time = time.time()
        if current_time - limits.last_request_timestamp < 2.0:  # 2 second minimum gap
            limits.rate_limit_violations += 1
            limits.suspicion_score = min(1.0, limits.suspicion_score + 0.1)
            self._save_user_limits(limits)
            return False, "Requests too frequent (min 2 second gap)", self._get_limits_info(limits)
        
        return True, "Request allowed", self._get_limits_info(limits)
    
    def consume_request(self, address: str, success: bool = True) -> UserLimits:
        """
        Consume a request from user's quota and update usage stats.
        
        Args:
            address: User wallet address
            success: Whether the request was successful
            
        Returns:
            Updated user limits
        """
        limits = self.get_user_limits(address)
        current_time = time.time()
        
        # Update request tracking
        limits.total_requests_made += 1
        limits.last_request_timestamp = current_time
        
        # Consume from quota (prefer bonus requests first)
        if limits.bonus_requests_available > 0:
            limits.bonus_requests_available -= 1
        elif limits.free_requests_remaining > 0:
            limits.free_requests_remaining -= 1
        
        # Consume hourly limit
        limits.hourly_requests_remaining = max(0, limits.hourly_requests_remaining - 1)
        
        # Update success/failure tracking
        if success:
            limits.consecutive_failures = 0
            # Small suspicion score improvement on success
            limits.suspicion_score = max(0, limits.suspicion_score - 0.01)
        else:
            limits.consecutive_failures += 1
            # Increase suspicion on repeated failures
            if limits.consecutive_failures >= 3:
                limits.suspicion_score = min(1.0, limits.suspicion_score + 0.1)
        
        # Save updated limits
        self._save_user_limits(limits)
        
        return limits
    
    def award_contribution_credits(self, address: str, contribution_type: str, amount: float = 1.0) -> UserLimits:
        """
        Award credits for user contributions (validation, data upload, etc.).
        
        Args:
            address: User wallet address
            contribution_type: Type of contribution made
            amount: Credit amount to award
            
        Returns:
            Updated user limits
        """
        limits = self.get_user_limits(address)
        
        # Award credits and update contribution tracking
        limits.earned_credits += amount
        limits.contribution_score += amount * 10  # 10 score per credit
        
        # Update specific contribution counters
        if contribution_type == "validation":
            limits.successful_validations += 1
        elif contribution_type == "dataset":
            limits.datasets_contributed += 1
        elif contribution_type == "expert":
            limits.experts_improved += 1
        
        # Check for trust level promotion
        old_trust = limits.trust_level
        limits = self._check_trust_level_promotion(limits)
        
        if limits.trust_level > old_trust:
            logger.info(f"User {address} promoted to trust level {limits.trust_level}")
        
        # Award bonus requests for significant contributions
        if contribution_type == "expert" and amount >= 5.0:
            limits.bonus_requests_available += 10  # 10 bonus requests for expert improvement
        elif contribution_type == "dataset" and amount >= 3.0:
            limits.bonus_requests_available += 5   # 5 bonus requests for quality dataset
        
        self._save_user_limits(limits)
        return limits
    
    def _check_trust_level_promotion(self, limits: UserLimits) -> UserLimits:
        """Check and update user trust level based on contributions."""
        current_trust = TrustLevel(limits.trust_level)
        
        # Promotion criteria
        if (current_trust == TrustLevel.NEWCOMER and 
            limits.contribution_score >= 50 and 
            limits.total_requests_made >= 10):
            limits.trust_level = TrustLevel.VERIFIED.value
            
        elif (current_trust == TrustLevel.VERIFIED and 
              limits.contribution_score >= 200 and
              limits.successful_validations >= 5):
            limits.trust_level = TrustLevel.CONTRIBUTOR.value
            
        elif (current_trust == TrustLevel.CONTRIBUTOR and
              limits.contribution_score >= 1000 and
              limits.datasets_contributed >= 3):
            limits.trust_level = TrustLevel.TRUSTED.value
            
        elif (current_trust == TrustLevel.TRUSTED and
              limits.contribution_score >= 5000 and
              limits.experts_improved >= 2):
            limits.trust_level = TrustLevel.EXPERT.value
        
        # Update base limits if promoted
        if limits.trust_level != current_trust.value:
            new_trust = TrustLevel(limits.trust_level)
            base_limits = self.TRUST_LEVEL_LIMITS[new_trust]
            
            limits.free_requests_per_day = base_limits["daily_free"]
            limits.requests_per_hour = base_limits["hourly_burst"]
        
        return limits
    
    def _get_limits_info(self, limits: UserLimits) -> Dict[str, Any]:
        """Get user limits info for client display."""
        trust_level = TrustLevel(limits.trust_level)
        
        return {
            "trust_level": trust_level.name,
            "trust_level_numeric": limits.trust_level,
            "contribution_score": limits.contribution_score,
            "free_requests_remaining": limits.free_requests_remaining,
            "free_requests_per_day": limits.free_requests_per_day,
            "bonus_requests_available": limits.bonus_requests_available,
            "hourly_requests_remaining": limits.hourly_requests_remaining,
            "daily_reset_in": max(0, limits.daily_reset_timestamp - time.time()),
            "hourly_reset_in": max(0, limits.hourly_reset_timestamp - time.time()),
            "earned_credits": limits.earned_credits,
            "total_requests_made": limits.total_requests_made,
            "suspicion_score": limits.suspicion_score
        }
    
    def _get_next_daily_reset(self) -> float:
        """Get timestamp for next daily reset (midnight UTC)."""
        current_time = time.time()
        # Next midnight UTC
        next_midnight = int(current_time // 86400 + 1) * 86400
        return float(next_midnight)
    
    def _get_next_hourly_reset(self) -> float:
        """Get timestamp for next hourly reset."""
        current_time = time.time()
        # Next hour boundary
        next_hour = int(current_time // 3600 + 1) * 3600
        return float(next_hour)
    
    def get_user_statistics(self, address: str) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        limits = self.get_user_limits(address)
        trust_level = TrustLevel(limits.trust_level)
        
        # Calculate progress to next trust level
        next_level_requirements = self._get_next_level_requirements(trust_level)
        
        return {
            "address": limits.address,
            "trust_level": trust_level.name,
            "contribution_score": limits.contribution_score,
            "total_requests": limits.total_requests_made,
            "successful_validations": limits.successful_validations,
            "datasets_contributed": limits.datasets_contributed,
            "experts_improved": limits.experts_improved,
            "earned_credits": limits.earned_credits,
            "current_limits": self._get_limits_info(limits),
            "next_level": next_level_requirements,
            "account_health": {
                "suspicion_score": limits.suspicion_score,
                "consecutive_failures": limits.consecutive_failures,
                "rate_limit_violations": limits.rate_limit_violations,
                "status": self._get_account_health_status(limits)
            }
        }
    
    def _get_next_level_requirements(self, current_level: TrustLevel) -> Dict[str, Any]:
        """Get requirements for next trust level."""
        if current_level == TrustLevel.EXPERT:
            return {"message": "Maximum trust level achieved"}
        
        requirements = {
            TrustLevel.NEWCOMER: {
                "next_level": "VERIFIED",
                "contribution_score_needed": 50,
                "requests_needed": 10
            },
            TrustLevel.VERIFIED: {
                "next_level": "CONTRIBUTOR", 
                "contribution_score_needed": 200,
                "validations_needed": 5
            },
            TrustLevel.CONTRIBUTOR: {
                "next_level": "TRUSTED",
                "contribution_score_needed": 1000,
                "datasets_needed": 3
            },
            TrustLevel.TRUSTED: {
                "next_level": "EXPERT",
                "contribution_score_needed": 5000,
                "expert_improvements_needed": 2
            }
        }
        
        return requirements.get(current_level, {})
    
    def _get_account_health_status(self, limits: UserLimits) -> str:
        """Get account health status based on metrics."""
        if limits.suspicion_score > 0.8:
            return "HIGH_RISK"
        elif limits.suspicion_score > 0.5:
            return "MODERATE_RISK"
        elif limits.consecutive_failures > 5:
            return "DEGRADED"
        else:
            return "HEALTHY"

# Global instance
_free_tier_manager = None

def get_free_tier_manager() -> FreeTierManager:
    """Get or create FreeTierManager singleton."""
    global _free_tier_manager
    if _free_tier_manager is None:
        _free_tier_manager = FreeTierManager()
    return _free_tier_manager