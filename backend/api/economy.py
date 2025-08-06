#!/usr/bin/env python3
"""
Economy API endpoints for Blyan Network
Provides token supply, distribution, and leaderboard data
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from backend.core.reward_engine import get_reward_engine
from backend.core.chain import Chain
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/economy", tags=["economy"])

# Foundation wallet address (would be from config in production)
FOUNDATION_ADDRESS = "0xABC123DEF456789...foundation.blyan"

@router.get("/supply")
async def get_token_supply() -> Dict[str, Any]:
    """
    Get current token supply information.
    
    Returns:
        - total: Total supply cap (1B BLY)
        - circulating: Current circulating supply
        - minted: Total minted through rewards
        - burned: Total burned tokens
        - foundation: Foundation treasury balance
        - locked: Tokens in vesting contracts
    """
    try:
        engine = get_reward_engine()
        stats = engine.get_reward_stats()
        
        # Calculate circulating supply
        # In production, this would query actual blockchain
        total_cap = 1_000_000_000
        genesis_supply = 100_000_000
        minted = stats["inflation"]["minted_total"]
        burned = 0  # Track burns when implemented
        
        # Vesting calculations (simplified)
        current_time = datetime.now()
        months_since_launch = 6  # Placeholder
        
        # Team vesting: 15% over 48 months after 12 month cliff
        team_allocation = total_cap * 0.15
        team_unlocked = 0
        if months_since_launch > 12:
            vested_months = min(months_since_launch - 12, 36)
            team_unlocked = team_allocation * (vested_months / 36)
            
        # Investor vesting: 15% over 24 months after 6 month cliff  
        investor_allocation = total_cap * 0.15
        investor_unlocked = 0
        if months_since_launch > 6:
            vested_months = min(months_since_launch - 6, 18)
            investor_unlocked = investor_allocation * (vested_months / 18)
            
        circulating = genesis_supply + minted + team_unlocked + investor_unlocked - burned
        
        return {
            "total": total_cap,
            "circulating": int(circulating),
            "minted": int(minted),
            "burned": int(burned),
            "foundation": {
                "address": FOUNDATION_ADDRESS,
                "balance": int(total_cap * 0.20)  # 20% allocation
            },
            "locked": {
                "team": int(team_allocation - team_unlocked),
                "investors": int(investor_allocation - investor_unlocked),
                "ecosystem": int(total_cap * 0.10)  # 10% ecosystem fund
            },
            "inflation": {
                "current_rate": f"{stats['inflation']['current_rate']:.2%}",
                "target_rate": f"{stats['inflation']['annual_target']:.1%}",
                "minted_this_year": int(stats['inflation']['minted_this_year'])
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get token supply: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution") 
async def get_token_distribution() -> Dict[str, Any]:
    """
    Get token distribution breakdown by category.
    """
    try:
        engine = get_reward_engine()
        stats = engine.get_reward_stats()
        dist_stats = stats["distribution"]
        
        total_cap = 1_000_000_000
        
        return {
            "allocations": {
                "foundation": {
                    "percentage": "20%",
                    "amount": 200_000_000,
                    "purpose": "Ecosystem development and operations"
                },
                "team": {
                    "percentage": "15%", 
                    "amount": 150_000_000,
                    "vesting": "48 months with 12 month cliff"
                },
                "investors": {
                    "percentage": "15%",
                    "amount": 150_000_000,
                    "vesting": "24 months with 6 month cliff"
                },
                "ecosystem_grants": {
                    "percentage": "10%",
                    "amount": 100_000_000,
                    "purpose": "Developer grants and partnerships"
                },
                "community_rewards": {
                    "percentage": "40%",
                    "amount": 400_000_000,
                    "breakdown": {
                        "inference": int(dist_stats.get("inference", 0)),
                        "training": int(dist_stats.get("training", 0)),
                        "remaining": 400_000_000 - int(dist_stats.get("inference", 0) + dist_stats.get("training", 0))
                    }
                }
            },
            "reward_coefficients": {
                "alpha": stats["coefficients"]["alpha"],
                "beta": stats["coefficients"]["beta"],
                "last_adjusted": datetime.fromtimestamp(
                    stats["coefficients"]["last_updated"]
                ).isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/leaderboard")
async def get_leaderboard(
    category: str = Query("all", regex="^(all|inference|training)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """
    Get anonymous leaderboard of top contributors.
    
    Args:
        category: Filter by contribution type (all/inference/training)
        limit: Number of entries to return
        offset: Pagination offset
        
    Returns:
        Anonymous ranked list with earnings and contribution scores
    """
    try:
        # In production, this would query actual blockchain data
        # For now, return mock data structure
        
        # Mock leaderboard data
        leaderboard_entries = []
        
        for i in range(offset, min(offset + limit, 500)):  # Mock 500 contributors
            # Generate realistic looking data
            rank = i + 1
            
            if category == "inference":
                earnings = 50_000 / (rank ** 0.5)  # Power law distribution
                contribution_score = 10_000 / (rank ** 0.3)
            elif category == "training":
                earnings = 100_000 / (rank ** 0.7)
                contribution_score = 5_000 / (rank ** 0.4)
            else:  # all
                earnings = 75_000 / (rank ** 0.6)
                contribution_score = 7_500 / (rank ** 0.35)
                
            entry = {
                "rank": rank,
                "address_hash": f"0x{'a' * 8}...{'b' * 8}",  # Anonymized
                "earnings": int(earnings),
                "contribution_score": int(contribution_score),
                "last_active": datetime.now().isoformat()
            }
            
            # Add category breakdown for "all" view
            if category == "all":
                entry["breakdown"] = {
                    "inference": int(earnings * 0.4),
                    "training": int(earnings * 0.6)
                }
                
            leaderboard_entries.append(entry)
            
        return {
            "category": category,
            "total_contributors": 500,  # Mock total
            "entries": leaderboard_entries,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < 500
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rewards/calculate")
async def calculate_reward(
    contribution_type: str = Query(..., regex="^(inference|training)$"),
    tokens_served: Optional[int] = Query(None, ge=0),
    quality_score: Optional[float] = Query(None, ge=0, le=1),
    latency_ms: Optional[float] = Query(None, ge=0),
    delta_loss: Optional[float] = Query(None, ge=0),
    baseline_loss: Optional[float] = Query(None, gt=0)
) -> Dict[str, Any]:
    """
    Calculate potential reward for a contribution.
    Useful for contributors to estimate earnings.
    """
    try:
        engine = get_reward_engine()
        
        if contribution_type == "inference":
            if tokens_served is None or quality_score is None or latency_ms is None:
                raise HTTPException(
                    status_code=400,
                    detail="tokens_served, quality_score, and latency_ms required for inference"
                )
                
            reward = engine.reward_inference(
                tokens_served=tokens_served,
                quality_score=quality_score,
                latency_ms=latency_ms
            )
            
            return {
                "contribution_type": "inference",
                "parameters": {
                    "tokens_served": tokens_served,
                    "quality_score": quality_score,
                    "latency_ms": latency_ms
                },
                "reward": reward,
                "details": {
                    "base_rate": engine.cache.get('coefficients', (0, 0))[1],
                    "quality_multiplier": 1.0 if quality_score >= 0.95 else 0.8,
                    "latency_penalty": latency_ms > 400
                }
            }
            
        else:  # training
            if delta_loss is None:
                raise HTTPException(
                    status_code=400,
                    detail="delta_loss required for training rewards"
                )
                
            baseline = baseline_loss or 1.0
            reward = engine.reward_training(
                delta_loss=delta_loss,
                baseline_loss=baseline
            )
            
            improvement_pct = (delta_loss / baseline) * 100
            
            return {
                "contribution_type": "training",
                "parameters": {
                    "delta_loss": delta_loss,
                    "baseline_loss": baseline,
                    "improvement_percentage": f"{improvement_pct:.2f}%"
                },
                "reward": reward,
                "details": {
                    "base_rate": engine.cache.get('coefficients', (0, 0))[0],
                    "quality_bonus": improvement_pct >= 2.0
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate reward: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router in main app
def include_router(app):
    """Include economy routes in FastAPI app."""
    app.include_router(router)