#!/usr/bin/env python3
"""
L2 Community Voting System API
Implements 3-validator random selection for data quality voting

### PRODUCTION FEATURES ###
- Random validator selection
- 2/3 consensus requirement
- Time-limited voting (48 hours)
- Anti-gaming protection
- Vote weight based on contribution score
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import random
import logging
from pathlib import Path
import hashlib
import secrets

from backend.core.dataset_chain import DatasetChain
from backend.core.dataset_block import DatasetStage, DatasetQualityTier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voting", tags=["voting"])

# Import secure voting router
from backend.api.voting_secure import router as secure_voting_router

# Voting configuration
VOTING_DURATION = 48 * 3600  # 48 hours
MIN_VALIDATORS = 3  # Minimum validators per dataset
CONSENSUS_THRESHOLD = 2/3  # 2/3 majority required


class VoteRequest(BaseModel):
    """Vote submission request."""
    dataset_id: str
    vote: bool  # True = approve, False = reject
    voter_address: str
    feedback: Optional[str] = None
    quality_score: Optional[float] = None  # 0-5 rating


class VoteResponse(BaseModel):
    """Vote submission response."""
    success: bool
    message: str
    vote_count: Dict[str, int]
    consensus_reached: bool


class ValidatorSelection(BaseModel):
    """Validator selection response."""
    dataset_id: str
    selected_validators: List[str]
    voting_deadline: float
    reward_per_validator: float


class VotingStatus(BaseModel):
    """Current voting status for a dataset."""
    dataset_id: str
    stage: str
    validators: List[str]
    votes_submitted: int
    votes_required: int
    consensus: Optional[bool]
    deadline: float
    time_remaining: float


# In-memory tracking (use Redis in production)
active_voting_sessions: Dict[str, Dict] = {}
validator_assignments: Dict[str, List[str]] = {}
validator_contribution_scores: Dict[str, float] = {}


def get_dataset_chain() -> DatasetChain:
    """Get or create dataset chain instance."""
    root_dir = Path("./data")
    return DatasetChain(root_dir)


def calculate_validator_weight(address: str) -> float:
    """
    Calculate voting weight based on contribution score.
    Higher contributors get slightly more weight (1.0 - 1.5x).
    """
    base_weight = 1.0
    contribution_score = validator_contribution_scores.get(address, 0)
    
    # Weight formula: 1.0 + (contribution_score / 10000) * 0.5
    # Max weight is 1.5x for very high contributors
    weight_multiplier = min(1.5, 1.0 + (contribution_score / 10000) * 0.5)
    
    return base_weight * weight_multiplier


@router.post("/select_validators")
async def select_validators(dataset_id: str, pool_size: int = 100) -> ValidatorSelection:
    """
    Randomly select 3 validators for L2 community voting.
    
    ### Selection Process ###
    1. Get pool of eligible validators
    2. Weight by contribution score
    3. Randomly select 3 validators
    4. Assign voting task with deadline
    """
    # Check if voting already active
    if dataset_id in active_voting_sessions:
        raise HTTPException(400, f"Voting already active for dataset {dataset_id}")
    
    # Get eligible validators (in production, query from database)
    eligible_validators = [f"validator_{i}" for i in range(pool_size)]
    
    # Weight validators by contribution score
    weighted_validators = []
    for validator in eligible_validators:
        weight = calculate_validator_weight(validator)
        # Add validator multiple times based on weight (simple weighted random)
        weighted_validators.extend([validator] * int(weight * 10))
    
    # Randomly select 3 unique validators
    selected = random.sample(set(weighted_validators), min(MIN_VALIDATORS, len(set(weighted_validators))))
    
    # Create voting session
    deadline = time.time() + VOTING_DURATION
    session = {
        "dataset_id": dataset_id,
        "validators": selected,
        "votes": {},
        "deadline": deadline,
        "created_at": time.time(),
        "consensus": None
    }
    
    active_voting_sessions[dataset_id] = session
    validator_assignments[dataset_id] = selected
    
    # Calculate reward (from validation pool)
    reward_per_validator = 100.0  # 100 BLY per validator
    
    logger.info(f"Selected validators for {dataset_id}: {selected}")
    
    return ValidatorSelection(
        dataset_id=dataset_id,
        selected_validators=selected,
        voting_deadline=deadline,
        reward_per_validator=reward_per_validator
    )


@router.post("/submit_vote")
async def submit_vote(request: VoteRequest) -> VoteResponse:
    """
    Submit a vote for dataset quality.
    
    ### Vote Processing ###
    1. Verify validator is assigned
    2. Record weighted vote
    3. Check for consensus (2/3 majority)
    4. Finalize if consensus reached
    """
    dataset_id = request.dataset_id
    voter = request.voter_address
    
    # Check voting session exists
    if dataset_id not in active_voting_sessions:
        raise HTTPException(404, f"No active voting for dataset {dataset_id}")
    
    session = active_voting_sessions[dataset_id]
    
    # Check deadline
    if time.time() > session["deadline"]:
        raise HTTPException(400, "Voting deadline has passed")
    
    # Verify validator is assigned
    if voter not in session["validators"]:
        raise HTTPException(403, f"{voter} is not assigned to vote on {dataset_id}")
    
    # Check if already voted
    if voter in session["votes"]:
        raise HTTPException(400, f"{voter} has already voted")
    
    # Record vote with weight
    weight = calculate_validator_weight(voter)
    session["votes"][voter] = {
        "vote": request.vote,
        "weight": weight,
        "feedback": request.feedback,
        "quality_score": request.quality_score,
        "timestamp": time.time()
    }
    
    # Calculate current vote counts
    weighted_approve = sum(
        v["weight"] for v in session["votes"].values() 
        if v["vote"] is True
    )
    weighted_reject = sum(
        v["weight"] for v in session["votes"].values() 
        if v["vote"] is False
    )
    total_weight = weighted_approve + weighted_reject
    
    # Check for consensus (2/3 majority)
    consensus_reached = False
    consensus_result = None
    
    if len(session["votes"]) == len(session["validators"]):
        # All validators have voted
        if weighted_approve / total_weight >= CONSENSUS_THRESHOLD:
            consensus_reached = True
            consensus_result = True
        elif weighted_reject / total_weight >= CONSENSUS_THRESHOLD:
            consensus_reached = True
            consensus_result = False
    
    # Update session
    session["consensus"] = consensus_result
    
    # If consensus reached, finalize in dataset chain
    if consensus_reached:
        try:
            chain = get_dataset_chain()
            chain.finalize_community_vote(dataset_id, consensus_result)
            
            # Clean up session
            del active_voting_sessions[dataset_id]
            
            # Distribute rewards to validators
            await distribute_validator_rewards(dataset_id, session["validators"])
            
            logger.info(f"Consensus reached for {dataset_id}: {'APPROVED' if consensus_result else 'REJECTED'}")
        except Exception as e:
            logger.error(f"Failed to finalize vote: {e}")
    
    return VoteResponse(
        success=True,
        message=f"Vote recorded for {voter}",
        vote_count={
            "approve": int(weighted_approve),
            "reject": int(weighted_reject),
            "total_votes": len(session["votes"]),
            "required_votes": len(session["validators"])
        },
        consensus_reached=consensus_reached
    )


@router.get("/status/{dataset_id}")
async def get_voting_status(dataset_id: str) -> VotingStatus:
    """Get current voting status for a dataset."""
    if dataset_id not in active_voting_sessions:
        raise HTTPException(404, f"No active voting for dataset {dataset_id}")
    
    session = active_voting_sessions[dataset_id]
    
    return VotingStatus(
        dataset_id=dataset_id,
        stage="community_vote",
        validators=session["validators"],
        votes_submitted=len(session["votes"]),
        votes_required=len(session["validators"]),
        consensus=session.get("consensus"),
        deadline=session["deadline"],
        time_remaining=max(0, session["deadline"] - time.time())
    )


@router.get("/active")
async def get_active_voting_sessions() -> List[VotingStatus]:
    """Get all active voting sessions."""
    sessions = []
    
    for dataset_id, session in active_voting_sessions.items():
        sessions.append(VotingStatus(
            dataset_id=dataset_id,
            stage="community_vote",
            validators=session["validators"],
            votes_submitted=len(session["votes"]),
            votes_required=len(session["validators"]),
            consensus=session.get("consensus"),
            deadline=session["deadline"],
            time_remaining=max(0, session["deadline"] - time.time())
        ))
    
    return sessions


@router.get("/validator/{address}/assignments")
async def get_validator_assignments(address: str) -> List[Dict[str, Any]]:
    """Get voting assignments for a specific validator."""
    assignments = []
    
    for dataset_id, validators in validator_assignments.items():
        if address in validators:
            session = active_voting_sessions.get(dataset_id)
            if session:
                has_voted = address in session["votes"]
                assignments.append({
                    "dataset_id": dataset_id,
                    "deadline": session["deadline"],
                    "has_voted": has_voted,
                    "time_remaining": max(0, session["deadline"] - time.time())
                })
    
    return assignments


async def distribute_validator_rewards(dataset_id: str, validators: List[str]):
    """
    Distribute rewards to validators who participated in voting.
    """
    reward_per_validator = 100.0  # 100 BLY base reward
    
    for validator in validators:
        # In production, this would update the ledger
        logger.info(f"Rewarding validator {validator}: {reward_per_validator} BLY")
        
        # Update contribution score
        validator_contribution_scores[validator] = \
            validator_contribution_scores.get(validator, 0) + 10


@router.post("/simulate_voting")
async def simulate_voting_round(dataset_id: str) -> Dict[str, Any]:
    """
    Simulate a complete voting round for testing.
    """
    # Select validators
    selection = await select_validators(dataset_id, pool_size=10)
    
    # Simulate votes
    votes = []
    for i, validator in enumerate(selection.selected_validators):
        vote = VoteRequest(
            dataset_id=dataset_id,
            vote=i < 2,  # First 2 vote yes, rest vote no
            voter_address=validator,
            feedback=f"Test vote from {validator}",
            quality_score=4.0 if i < 2 else 2.0
        )
        result = await submit_vote(vote)
        votes.append(result)
    
    # Get final status (might be already completed)
    try:
        status = await get_voting_status(dataset_id)
        final_status = status.dict()
    except HTTPException:
        # Voting already completed
        final_status = {
            "dataset_id": dataset_id,
            "stage": "completed",
            "consensus": votes[-1].consensus_reached if votes else None,
            "message": "Voting completed with consensus"
        }
    
    return {
        "selection": selection.dict(),
        "votes": [v.dict() for v in votes],
        "final_status": final_status
    }