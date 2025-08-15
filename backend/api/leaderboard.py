#!/usr/bin/env python3
"""
Leaderboard and User Statistics API
Shows node contributions, expert usage, and user rankings

### FEATURES ###
- Node reputation leaderboard
- Expert contribution rankings  
- User statistics and personal ranking
- Regional/category filtering
- Real-time contribution tracking
"""

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import time
import logging
from pathlib import Path
import redis
import json

from backend.p2p.node_reputation import get_reputation_manager
from backend.accounting.ledger import get_ledger
from backend.model.moe_infer import MoEModelManager
from backend.core.dataset_chain import DatasetChain

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])

# Redis for caching leaderboard data
redis_client = redis.Redis(
    host='localhost', port=6379, db=3, decode_responses=True
)

class NodeRanking(BaseModel):
    """Node ranking entry."""
    rank: int
    node_id: str
    node_name: Optional[str] = None
    region: Optional[str] = None
    reputation_score: float
    total_requests: int
    success_rate: float
    avg_response_time: float
    p95_response_time: float
    total_rewards: Decimal
    experts_served: List[str]
    is_active: bool

class UserSummary(BaseModel):
    """User personal summary and ranking."""
    address: str
    rank: Optional[int] = None
    total_rewards: Decimal
    balance: Decimal
    contribution_score: float
    trust_level: int
    free_requests_remaining: int
    total_requests_made: int
    successful_validations: int
    datasets_contributed: int
    experts_improved: int
    badges: List[str]

class LeaderboardResponse(BaseModel):
    """Leaderboard response with rankings."""
    leaderboard_type: str
    total_entries: int
    updated_at: float
    rankings: List[Union[NodeRanking, Dict[str, Any]]]

def get_node_leaderboard(limit: int = 50, region: Optional[str] = None) -> List[NodeRanking]:
    """Get node reputation leaderboard."""
    reputation_manager = get_reputation_manager()
    
    # Get all node metrics
    metrics_summary = reputation_manager.get_metrics_summary()
    ledger = get_ledger()
    
    rankings = []
    for rank, (node_id, metrics) in enumerate(metrics_summary.items(), 1):
        if limit and rank > limit:
            break
            
        # Get rewards from ledger (mock for now)
        node_rewards = Decimal("0")  # TODO: Query actual rewards from ledger
        
        # Get served experts (mock data)
        experts_served = [f"layer{i}.expert{j}" for i in range(2) for j in range(2)]
        
        # Filter by region if specified
        node_region = "us-west"  # TODO: Get from node registry
        if region and node_region != region:
            continue
            
        ranking = NodeRanking(
            rank=rank,
            node_id=node_id,
            node_name=f"Node-{node_id[:8]}",
            region=node_region,
            reputation_score=metrics.get("reputation", 0),
            total_requests=metrics.get("total_requests", 0),
            success_rate=float(metrics.get("success_rate", "0%").rstrip('%')) / 100,
            avg_response_time=float(metrics.get("avg_response_time", "0s").rstrip('s')),
            p95_response_time=float(metrics.get("p95_response_time", "0s").rstrip('s')),
            total_rewards=node_rewards,
            experts_served=experts_served,
            is_active=not metrics.get("is_blacklisted", False)
        )
        rankings.append(ranking)
    
    return rankings

def get_expert_leaderboard(limit: int = 50) -> List[Dict[str, Any]]:
    """Get expert usage leaderboard."""
    cache_key = f"expert_leaderboard:{limit}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # Mock expert data - in production, query from MoEModelManager usage tracker
    experts_data = [
        {
            "rank": 1,
            "expert_name": "layer0.expert0",
            "call_count": 15420,
            "avg_response_time": 0.85,
            "quality_score": 4.7,
            "total_rewards": "2500.50",
            "last_used": time.time() - 300,
            "specialization": "language_understanding"
        },
        {
            "rank": 2, 
            "expert_name": "layer1.expert2",
            "call_count": 12890,
            "avg_response_time": 0.92,
            "quality_score": 4.5,
            "total_rewards": "2100.25",
            "last_used": time.time() - 150,
            "specialization": "reasoning"
        },
        # Add more mock data...
    ]
    
    # Cache for 5 minutes
    redis_client.setex(cache_key, 300, json.dumps(experts_data))
    return experts_data

def get_user_summary(address: str) -> UserSummary:
    """Get user personal summary and ranking."""
    ledger = get_ledger()
    
    # Get balance
    try:
        balance = ledger.get_user_balance(address)
    except:
        balance = Decimal("0")
    
    # Get free requests remaining from Redis SSOT
    user_limits_key = f"user_limits:{address.lower()}"
    user_limits = redis_client.hgetall(user_limits_key)
    
    free_remaining = int(user_limits.get("free_requests_remaining", 5))
    trust_level = int(user_limits.get("trust_level", 1))
    contribution_score = float(user_limits.get("contribution_score", 0))
    
    # Calculate rank (mock for now)
    rank = calculate_user_rank(address, contribution_score)
    
    # Generate badges based on achievements
    badges = generate_user_badges(user_limits)
    
    return UserSummary(
        address=address,
        rank=rank,
        total_rewards=balance,
        balance=balance,
        contribution_score=contribution_score,
        trust_level=trust_level,
        free_requests_remaining=free_remaining,
        total_requests_made=int(user_limits.get("total_requests_made", 0)),
        successful_validations=int(user_limits.get("successful_validations", 0)),
        datasets_contributed=int(user_limits.get("datasets_contributed", 0)),
        experts_improved=int(user_limits.get("experts_improved", 0)),
        badges=badges
    )

def calculate_user_rank(address: str, contribution_score: float) -> Optional[int]:
    """Calculate user rank based on contribution score."""
    # Get all users' scores from Redis and rank
    all_users_pattern = "user_limits:*"
    user_keys = redis_client.keys(all_users_pattern)
    
    scores = []
    for key in user_keys:
        user_data = redis_client.hgetall(key)
        user_addr = key.split(":")[1]
        score = float(user_data.get("contribution_score", 0))
        scores.append((user_addr, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Find user rank
    for rank, (user_addr, score) in enumerate(scores, 1):
        if user_addr == address:
            return rank
    
    return None

def generate_user_badges(user_limits: Dict[str, str]) -> List[str]:
    """Generate user achievement badges."""
    badges = []
    
    contribution_score = float(user_limits.get("contribution_score", 0))
    trust_level = int(user_limits.get("trust_level", 1))
    successful_validations = int(user_limits.get("successful_validations", 0))
    datasets_contributed = int(user_limits.get("datasets_contributed", 0))
    experts_improved = int(user_limits.get("experts_improved", 0))
    
    # Contribution badges
    if contribution_score >= 1000:
        badges.append("🏆 Elite Contributor")
    elif contribution_score >= 500:
        badges.append("🥇 Gold Contributor")
    elif contribution_score >= 100:
        badges.append("🥈 Silver Contributor")
    elif contribution_score >= 10:
        badges.append("🥉 Bronze Contributor")
    
    # Trust level badges
    if trust_level >= 5:
        badges.append("🛡️ Trusted Validator")
    elif trust_level >= 3:
        badges.append("✅ Verified Member")
    
    # Activity badges
    if successful_validations >= 100:
        badges.append("🔍 Master Validator")
    elif successful_validations >= 50:
        badges.append("🔎 Expert Validator")
    
    if datasets_contributed >= 10:
        badges.append("📊 Data Champion")
    
    if experts_improved >= 5:
        badges.append("🧠 AI Architect")
    
    # Special badges
    if len(badges) >= 5:
        badges.append("🌟 Renaissance Contributor")
    
    return badges

@router.get("/nodes")
async def get_node_leaderboard_api(
    limit: int = Query(50, ge=1, le=100),
    region: Optional[str] = Query(None, description="Filter by region")
) -> LeaderboardResponse:
    """
    Get node reputation leaderboard.
    Shows top performing nodes by reputation, success rate, and response time.
    """
    rankings = get_node_leaderboard(limit, region)
    
    return LeaderboardResponse(
        leaderboard_type="nodes",
        total_entries=len(rankings),
        updated_at=time.time(),
        rankings=rankings
    )

@router.get("/experts")
async def get_expert_leaderboard_api(
    limit: int = Query(50, ge=1, le=100)
) -> LeaderboardResponse:
    """
    Get expert usage leaderboard.
    Shows most used and highest quality experts.
    """
    rankings = get_expert_leaderboard(limit)
    
    return LeaderboardResponse(
        leaderboard_type="experts",
        total_entries=len(rankings),
        updated_at=time.time(),
        rankings=rankings
    )

@router.get("/users")
async def get_user_leaderboard_api(
    limit: int = Query(50, ge=1, le=100),
    sort_by: str = Query("contribution_score", description="Sort by: contribution_score, total_rewards, trust_level")
) -> LeaderboardResponse:
    """
    Get user contribution leaderboard.
    Shows top contributors by various metrics.
    """
    # Get all users from Redis
    all_users_pattern = "user_limits:*" 
    user_keys = redis_client.keys(all_users_pattern)
    
    users_data = []
    for key in user_keys[:limit]:  # Limit for performance
        user_addr = key.split(":")[1]
        try:
            summary = get_user_summary(user_addr)
            users_data.append(summary.dict())
        except Exception as e:
            logger.error(f"Failed to get summary for {user_addr}: {e}")
            continue
    
    # Sort by specified metric
    if sort_by == "total_rewards":
        users_data.sort(key=lambda x: float(x["total_rewards"]), reverse=True)
    elif sort_by == "trust_level":
        users_data.sort(key=lambda x: x["trust_level"], reverse=True)
    else:  # contribution_score
        users_data.sort(key=lambda x: x["contribution_score"], reverse=True)
    
    # Add ranks
    for i, user in enumerate(users_data):
        user["rank"] = i + 1
    
    return LeaderboardResponse(
        leaderboard_type="users",
        total_entries=len(users_data),
        updated_at=time.time(),
        rankings=users_data[:limit]
    )

@router.get("/me/summary")
async def get_my_summary(
    address: str = Query(..., description="User wallet address")
) -> UserSummary:
    """
    Get personal summary and ranking for a specific user.
    Shows individual stats, rank, badges, and remaining limits.
    """
    if not address or len(address) < 10:
        raise HTTPException(400, "Valid wallet address required")
    
    try:
        summary = get_user_summary(address)
        return summary
    except Exception as e:
        logger.error(f"Failed to get user summary for {address}: {e}")
        raise HTTPException(500, f"Failed to retrieve user summary: {str(e)}")

@router.get("/stats")
async def get_global_stats() -> Dict[str, Any]:
    """
    Get global network statistics.
    Overview of total activity, nodes, experts, and users.
    """
    reputation_manager = get_reputation_manager()
    metrics_summary = reputation_manager.get_metrics_summary()
    
    # Count active nodes
    active_nodes = sum(1 for m in metrics_summary.values() if not m.get("is_blacklisted", False))
    
    # Get total requests across all nodes
    total_requests = sum(m.get("total_requests", 0) for m in metrics_summary.values())
    
    # Count total users
    user_keys = redis_client.keys("user_limits:*")
    total_users = len(user_keys)
    
    # Calculate network success rate
    total_successful = sum(m.get("successful_requests", 0) for m in metrics_summary.values())
    network_success_rate = (total_successful / max(total_requests, 1)) * 100
    
    return {
        "total_nodes": len(metrics_summary),
        "active_nodes": active_nodes,
        "total_users": total_users,
        "total_requests_processed": total_requests,
        "network_success_rate": f"{network_success_rate:.2f}%",
        "total_experts_available": 32,  # Mock - get from MoE registry
        "avg_response_time": "0.89s",   # Mock - calculate from metrics
        "network_uptime": "99.7%",      # Mock - calculate from health checks
        "last_updated": time.time()
    }

@router.post("/refresh")
async def refresh_leaderboards() -> Dict[str, str]:
    """
    Manually refresh leaderboard cache.
    Forces recalculation of all rankings and stats.
    """
    # Clear Redis caches
    cache_patterns = ["expert_leaderboard:*", "user_ranking:*", "network_stats"]
    
    for pattern in cache_patterns:
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
    
    logger.info("Leaderboard caches cleared and will be regenerated on next request")
    
    return {
        "message": "Leaderboard data refreshed successfully",
        "timestamp": str(time.time())
    }