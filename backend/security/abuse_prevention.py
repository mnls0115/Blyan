#!/usr/bin/env python3
"""
Advanced Abuse Prevention System
Multi-layered protection against bot attacks, sybil attacks, and service abuse

### DEFENSE LAYERS ###
- Rate limiting with burst detection
- Behavioral pattern analysis
- Hardware fingerprinting
- CAPTCHA challenges  
- Proof-of-Work (PoW) requirements
- Semantic duplicate detection
- IP reputation tracking
"""

import time
import hashlib
import hmac
import secrets
import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import redis
import aiohttp
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Redis for abuse tracking
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=6,  # Dedicated DB for abuse prevention
    decode_responses=True
)

class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

class ChallengeType(Enum):
    """Types of anti-abuse challenges."""
    NONE = "none"
    CAPTCHA = "captcha"
    PROOF_OF_WORK = "proof_of_work"
    RATE_DELAY = "rate_delay"
    ACCOUNT_VERIFY = "account_verify"
    MANUAL_REVIEW = "manual_review"

@dataclass
class RequestFingerprint:
    """Comprehensive request fingerprint for abuse detection."""
    user_address: str
    ip_address: str
    user_agent: str
    hardware_fingerprint: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def generate_composite_key(self) -> str:
        """Generate composite key for tracking."""
        components = [
            self.user_address or "anonymous",
            hashlib.sha256(self.ip_address.encode()).hexdigest()[:16],
            hashlib.sha256(self.user_agent.encode()).hexdigest()[:8]
        ]
        if self.hardware_fingerprint:
            components.append(self.hardware_fingerprint[:8])
        
        return "|".join(components)

@dataclass
class BehaviorPattern:
    """User behavior pattern analysis."""
    request_intervals: List[float] = field(default_factory=list)
    prompt_lengths: List[int] = field(default_factory=list)
    response_patterns: List[str] = field(default_factory=list)
    success_failure_ratio: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    suspicious_score: float = 0.0
    
    def add_request(self, prompt_length: int, success: bool, response_time: float):
        """Add request to behavior pattern."""
        current_time = time.time()
        
        # Track intervals (last 10 requests)
        if len(self.request_intervals) > 0:
            interval = current_time - self.request_intervals[-1]
            self.request_intervals.append(interval)
            if len(self.request_intervals) > 10:
                self.request_intervals.pop(0)
        else:
            self.request_intervals.append(current_time)
        
        # Track prompt lengths
        self.prompt_lengths.append(prompt_length)
        if len(self.prompt_lengths) > 20:
            self.prompt_lengths.pop(0)
            
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
            
        self.success_failure_ratio = (self.total_requests - self.failed_requests) / max(self.total_requests, 1)
        
        # Calculate suspicion score
        self._update_suspicion_score()
    
    def _update_suspicion_score(self):
        """Update suspicion score based on patterns."""
        score = 0.0
        
        # Check for bot-like regular intervals
        if len(self.request_intervals) >= 3:
            intervals = self.request_intervals[-5:]  # Last 5 intervals
            avg_interval = sum(intervals) / len(intervals)
            variance = sum(abs(i - avg_interval) for i in intervals) / len(intervals)
            
            # Low variance indicates bot behavior
            if avg_interval > 0 and variance / avg_interval < 0.1:
                score += 0.3
                
            # Too fast requests
            if avg_interval < 2.0:
                score += 0.2
        
        # Check prompt length patterns
        if len(self.prompt_lengths) >= 5:
            # All prompts very similar length (bot characteristic)
            lengths = self.prompt_lengths[-10:]
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum(abs(l - avg_length) for l in lengths) / len(lengths)
            
            if avg_length > 0 and length_variance / avg_length < 0.1:
                score += 0.2
        
        # High failure rate
        if self.success_failure_ratio < 0.5 and self.total_requests >= 5:
            score += 0.3
            
        self.suspicious_score = min(1.0, score)

class AbusePreventionSystem:
    """Advanced abuse prevention with multiple detection layers."""
    
    def __init__(self):
        self.redis = redis_client
        
        # Tracking dictionaries
        self.behavior_patterns: Dict[str, BehaviorPattern] = {}
        self.ip_reputation: Dict[str, float] = {}
        self.recent_prompts: Dict[str, List[str]] = {}  # For duplicate detection
        
        # Configuration
        self.config = {
            "max_requests_per_minute": 10,
            "max_requests_per_hour": 100,
            "min_request_interval": 2.0,  # seconds
            "duplicate_threshold": 0.85,  # similarity threshold
            "high_suspicion_threshold": 0.7,
            "critical_suspicion_threshold": 0.9,
            "captcha_solve_window": 300,  # 5 minutes
            "pow_difficulty": 4,  # number of leading zeros required
        }
    
    async def assess_request(self, 
                           fingerprint: RequestFingerprint,
                           prompt: str,
                           request_context: Dict[str, Any] = None) -> Tuple[bool, ChallengeType, Dict[str, Any]]:
        """
        Comprehensive request assessment for abuse detection.
        
        Returns:
            - allowed: Whether request should be allowed
            - challenge_type: Type of challenge required (if any)  
            - context: Additional context for client
        """
        
        composite_key = fingerprint.generate_composite_key()
        
        # Layer 1: Rate limiting
        rate_limit_result = await self._check_rate_limits(composite_key, fingerprint.ip_address)
        if not rate_limit_result["allowed"]:
            return False, ChallengeType.RATE_DELAY, rate_limit_result
        
        # Layer 2: Behavioral analysis
        behavior_result = await self._analyze_behavior_pattern(composite_key, prompt)
        if behavior_result["threat_level"] >= ThreatLevel.HIGH.value:
            if behavior_result["threat_level"] == ThreatLevel.CRITICAL.value:
                return False, ChallengeType.MANUAL_REVIEW, behavior_result
            else:
                return False, ChallengeType.CAPTCHA, behavior_result
        
        # Layer 3: IP reputation check
        ip_reputation = await self._check_ip_reputation(fingerprint.ip_address)
        if ip_reputation < 0.3:  # Low reputation IP
            return False, ChallengeType.CAPTCHA, {
                "reason": "Low IP reputation",
                "ip_score": ip_reputation
            }
        
        # Layer 4: Semantic duplicate detection
        duplicate_result = await self._check_semantic_duplicates(composite_key, prompt)
        if duplicate_result["is_duplicate"]:
            return False, ChallengeType.PROOF_OF_WORK, duplicate_result
        
        # Layer 5: Hardware fingerprint validation (if available)
        if fingerprint.hardware_fingerprint:
            hw_result = await self._validate_hardware_fingerprint(fingerprint)
            if not hw_result["valid"]:
                return False, ChallengeType.ACCOUNT_VERIFY, hw_result
        
        # Request allowed
        return True, ChallengeType.NONE, {"status": "approved"}
    
    async def _check_rate_limits(self, composite_key: str, ip_address: str) -> Dict[str, Any]:
        """Check various rate limits."""
        current_time = time.time()
        
        # Per-user rate limiting
        user_minute_key = f"rate_limit:user:{composite_key}:minute"
        user_hour_key = f"rate_limit:user:{composite_key}:hour"
        
        user_minute_count = self.redis.incr(user_minute_key)
        self.redis.expire(user_minute_key, 60)
        
        if user_minute_count > self.config["max_requests_per_minute"]:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded: {user_minute_count}/{self.config['max_requests_per_minute']} per minute",
                "retry_after": 60,
                "challenge_type": "rate_delay"
            }
        
        user_hour_count = self.redis.incr(user_hour_key)
        self.redis.expire(user_hour_key, 3600)
        
        if user_hour_count > self.config["max_requests_per_hour"]:
            return {
                "allowed": False,
                "reason": f"Hourly limit exceeded: {user_hour_count}/{self.config['max_requests_per_hour']} per hour",
                "retry_after": 3600,
                "challenge_type": "captcha"
            }
        
        # IP-based rate limiting (more aggressive)
        ip_key = f"rate_limit:ip:{hashlib.sha256(ip_address.encode()).hexdigest()[:16]}:minute"
        ip_count = self.redis.incr(ip_key)
        self.redis.expire(ip_key, 60)
        
        if ip_count > self.config["max_requests_per_minute"] * 5:  # 5x user limit for IPs
            return {
                "allowed": False,
                "reason": "IP rate limit exceeded",
                "retry_after": 60,
                "challenge_type": "captcha"
            }
        
        return {"allowed": True}
    
    async def _analyze_behavior_pattern(self, composite_key: str, prompt: str) -> Dict[str, Any]:
        """Analyze user behavior patterns for bot detection."""
        
        # Get or create behavior pattern
        if composite_key not in self.behavior_patterns:
            self.behavior_patterns[composite_key] = BehaviorPattern()
        
        pattern = self.behavior_patterns[composite_key]
        
        # Add current request (assume success for now)
        pattern.add_request(
            prompt_length=len(prompt),
            success=True,  # Will be updated later
            response_time=0.0  # Not available yet
        )
        
        # Determine threat level
        suspicion = pattern.suspicious_score
        
        if suspicion >= self.config["critical_suspicion_threshold"]:
            threat_level = ThreatLevel.CRITICAL
        elif suspicion >= self.config["high_suspicion_threshold"]:
            threat_level = ThreatLevel.HIGH
        elif suspicion >= 0.4:
            threat_level = ThreatLevel.MODERATE
        else:
            threat_level = ThreatLevel.LOW
        
        return {
            "threat_level": threat_level.value,
            "suspicion_score": suspicion,
            "total_requests": pattern.total_requests,
            "success_rate": pattern.success_failure_ratio,
            "behavior_flags": self._get_behavior_flags(pattern)
        }
    
    def _get_behavior_flags(self, pattern: BehaviorPattern) -> List[str]:
        """Get specific behavior flags for transparency."""
        flags = []
        
        if len(pattern.request_intervals) >= 3:
            intervals = pattern.request_intervals[-5:]
            avg_interval = sum(intervals) / len(intervals)
            variance = sum(abs(i - avg_interval) for i in intervals) / len(intervals)
            
            if avg_interval > 0 and variance / avg_interval < 0.1:
                flags.append("regular_timing_pattern")
            
            if avg_interval < 2.0:
                flags.append("high_frequency_requests")
        
        if pattern.success_failure_ratio < 0.5 and pattern.total_requests >= 5:
            flags.append("high_failure_rate")
            
        if len(pattern.prompt_lengths) >= 5:
            lengths = pattern.prompt_lengths[-10:]
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum(abs(l - avg_length) for l in lengths) / len(lengths)
            
            if avg_length > 0 and length_variance / avg_length < 0.1:
                flags.append("similar_prompt_lengths")
        
        return flags
    
    async def _check_ip_reputation(self, ip_address: str) -> float:
        """Check IP reputation score (0.0 to 1.0)."""
        # In production, integrate with IP reputation services
        # For now, simple heuristics based on our own data
        
        ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
        
        # Check our local reputation cache
        cached_rep = self.redis.get(f"ip_reputation:{ip_hash}")
        if cached_rep:
            return float(cached_rep)
        
        # Calculate reputation based on historical behavior
        reputation = 0.8  # Default neutral-good reputation
        
        # Check historical abuse patterns
        abuse_key = f"ip_abuse:{ip_hash}:24h"
        abuse_count = self.redis.get(abuse_key)
        if abuse_count:
            abuse_count = int(abuse_count)
            reputation -= min(0.5, abuse_count * 0.1)
        
        # Cache reputation for 1 hour
        self.redis.setex(f"ip_reputation:{ip_hash}", 3600, str(reputation))
        
        return max(0.0, reputation)
    
    async def _check_semantic_duplicates(self, composite_key: str, prompt: str) -> Dict[str, Any]:
        """Check for semantic duplicate prompts (simple version)."""
        
        # Normalize prompt for comparison
        normalized_prompt = self._normalize_prompt(prompt)
        
        # Get recent prompts for this user
        recent_prompts_key = f"recent_prompts:{composite_key}"
        recent_prompts_data = self.redis.lrange(recent_prompts_key, 0, 19)  # Last 20 prompts
        
        # Check for exact or near-exact duplicates
        for recent_prompt_data in recent_prompts_data:
            try:
                recent_prompt = json.loads(recent_prompt_data)
                similarity = self._calculate_prompt_similarity(
                    normalized_prompt, 
                    recent_prompt["normalized"]
                )
                
                if similarity >= self.config["duplicate_threshold"]:
                    return {
                        "is_duplicate": True,
                        "similarity": similarity,
                        "original_timestamp": recent_prompt["timestamp"],
                        "reason": "Similar prompt detected within recent history"
                    }
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Store current prompt
        prompt_data = {
            "normalized": normalized_prompt,
            "timestamp": time.time(),
            "original": prompt[:100]  # Store first 100 chars for debugging
        }
        
        self.redis.lpush(recent_prompts_key, json.dumps(prompt_data))
        self.redis.ltrim(recent_prompts_key, 0, 19)  # Keep only last 20
        self.redis.expire(recent_prompts_key, 3600)  # 1 hour expiry
        
        return {"is_duplicate": False}
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for duplicate detection."""
        import re
        
        # Convert to lowercase
        normalized = prompt.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common filler words (simple approach)
        filler_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = normalized.split()
        normalized = ' '.join(word for word in words if word not in filler_words)
        
        return normalized
    
    def _calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts (simple method)."""
        if not prompt1 or not prompt2:
            return 0.0
        
        # Simple character-level similarity
        if prompt1 == prompt2:
            return 1.0
        
        # Jaccard similarity on words
        words1 = set(prompt1.split())
        words2 = set(prompt2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _validate_hardware_fingerprint(self, fingerprint: RequestFingerprint) -> Dict[str, Any]:
        """Validate hardware fingerprint for consistency."""
        hw_key = f"hardware_fp:{fingerprint.user_address}"
        stored_fp = self.redis.get(hw_key)
        
        if not stored_fp:
            # First time seeing this hardware fingerprint
            self.redis.setex(hw_key, 86400 * 7, fingerprint.hardware_fingerprint)  # 7 days
            return {"valid": True, "status": "new_device_registered"}
        
        if stored_fp != fingerprint.hardware_fingerprint:
            # Hardware fingerprint changed - possible account sharing or compromise
            return {
                "valid": False,
                "reason": "Hardware fingerprint mismatch",
                "action_required": "Device verification needed"
            }
        
        return {"valid": True, "status": "device_verified"}
    
    def generate_captcha_challenge(self, composite_key: str) -> Dict[str, Any]:
        """Generate CAPTCHA challenge."""
        challenge_id = secrets.token_hex(16)
        
        # Simple math CAPTCHA (in production, use proper CAPTCHA service)
        a = secrets.randbelow(10) + 1
        b = secrets.randbelow(10) + 1
        operation = secrets.choice(['+', '-'])
        
        if operation == '+':
            answer = a + b
            question = f"What is {a} + {b}?"
        else:
            answer = a - b
            question = f"What is {a} - {b}?"
        
        # Store challenge
        challenge_data = {
            "answer": answer,
            "composite_key": composite_key,
            "created_at": time.time()
        }
        
        self.redis.setex(
            f"captcha_challenge:{challenge_id}",
            self.config["captcha_solve_window"],
            json.dumps(challenge_data)
        )
        
        return {
            "challenge_id": challenge_id,
            "challenge_type": "math",
            "question": question,
            "expires_in": self.config["captcha_solve_window"]
        }
    
    def verify_captcha_response(self, challenge_id: str, response: str) -> bool:
        """Verify CAPTCHA response."""
        challenge_key = f"captcha_challenge:{challenge_id}"
        challenge_data = self.redis.get(challenge_key)
        
        if not challenge_data:
            return False
        
        try:
            challenge = json.loads(challenge_data)
            correct_answer = str(challenge["answer"])
            
            if response.strip() == correct_answer:
                # Mark challenge as solved
                self.redis.delete(challenge_key)
                
                # Grant temporary bypass
                bypass_key = f"captcha_bypass:{challenge['composite_key']}"
                self.redis.setex(bypass_key, 300, "verified")  # 5 minute bypass
                
                return True
                
        except (json.JSONDecodeError, KeyError):
            pass
        
        return False
    
    def generate_pow_challenge(self, composite_key: str) -> Dict[str, Any]:
        """Generate Proof-of-Work challenge."""
        challenge_id = secrets.token_hex(16)
        
        # Generate random data to hash
        data = secrets.token_hex(32)
        difficulty = self.config["pow_difficulty"]
        
        challenge_data = {
            "data": data,
            "difficulty": difficulty,
            "composite_key": composite_key,
            "created_at": time.time()
        }
        
        self.redis.setex(
            f"pow_challenge:{challenge_id}",
            300,  # 5 minutes to solve
            json.dumps(challenge_data)
        )
        
        return {
            "challenge_id": challenge_id,
            "challenge_type": "proof_of_work",
            "data": data,
            "difficulty": difficulty,
            "description": f"Find a nonce such that SHA256(data + nonce) starts with {difficulty} zeros",
            "expires_in": 300
        }
    
    def verify_pow_solution(self, challenge_id: str, nonce: str) -> bool:
        """Verify Proof-of-Work solution."""
        challenge_key = f"pow_challenge:{challenge_id}"
        challenge_data = self.redis.get(challenge_key)
        
        if not challenge_data:
            return False
        
        try:
            challenge = json.loads(challenge_data)
            data = challenge["data"]
            difficulty = challenge["difficulty"]
            
            # Verify solution
            hash_input = data + nonce
            hash_result = hashlib.sha256(hash_input.encode()).hexdigest()
            
            if hash_result.startswith('0' * difficulty):
                # Mark challenge as solved
                self.redis.delete(challenge_key)
                
                # Grant temporary bypass
                bypass_key = f"pow_bypass:{challenge['composite_key']}"
                self.redis.setex(bypass_key, 600, "verified")  # 10 minute bypass
                
                return True
                
        except (json.JSONDecodeError, KeyError):
            pass
        
        return False
    
    def record_request_outcome(self, composite_key: str, success: bool):
        """Record the outcome of a request for learning."""
        if composite_key in self.behavior_patterns:
            pattern = self.behavior_patterns[composite_key]
            if not success:
                pattern.failed_requests += 1
                pattern.success_failure_ratio = (pattern.total_requests - pattern.failed_requests) / max(pattern.total_requests, 1)
                pattern._update_suspicion_score()
    
    def get_user_threat_assessment(self, composite_key: str) -> Dict[str, Any]:
        """Get comprehensive threat assessment for a user."""
        if composite_key not in self.behavior_patterns:
            return {
                "threat_level": ThreatLevel.LOW.name,
                "suspicion_score": 0.0,
                "total_requests": 0,
                "flags": []
            }
        
        pattern = self.behavior_patterns[composite_key]
        
        if pattern.suspicious_score >= self.config["critical_suspicion_threshold"]:
            threat_level = ThreatLevel.CRITICAL
        elif pattern.suspicious_score >= self.config["high_suspicion_threshold"]:
            threat_level = ThreatLevel.HIGH
        elif pattern.suspicious_score >= 0.4:
            threat_level = ThreatLevel.MODERATE
        else:
            threat_level = ThreatLevel.LOW
        
        return {
            "threat_level": threat_level.name,
            "suspicion_score": pattern.suspicious_score,
            "total_requests": pattern.total_requests,
            "success_rate": pattern.success_failure_ratio,
            "flags": self._get_behavior_flags(pattern),
            "recent_intervals": pattern.request_intervals[-5:] if pattern.request_intervals else []
        }

# Global instance
_abuse_prevention_system = None

def get_abuse_prevention_system() -> AbusePreventionSystem:
    """Get or create AbusePreventionSystem singleton."""
    global _abuse_prevention_system
    if _abuse_prevention_system is None:
        _abuse_prevention_system = AbusePreventionSystem()
    return _abuse_prevention_system