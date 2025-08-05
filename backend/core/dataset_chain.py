#!/usr/bin/env python3
"""
Dataset-Chain D Implementation - Training Data Governance Blockchain

This module implements Chain D, the dedicated blockchain for dataset governance
in the Blyan network, enabling complete transparency and democratic control
over AI training data.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from .chain import Chain
from .dataset_block import DatasetBlock, DatasetMetadata, DatasetStage, DatasetQualityTier
from .block import Block


class DatasetChain(Chain):
    """Chain D - Dataset governance blockchain extending base Chain functionality."""
    
    def __init__(self, data_root: Path, chain_id: str = "D"):
        """Initialize Dataset-Chain D."""
        super().__init__(data_root, chain_id)
        self.chain_id = chain_id
        
        # Dataset-specific tracking
        self.pending_audits = {}        # dataset_id -> DatasetBlock
        self.community_votes = {}       # dataset_id -> voting info
        self.quality_tiers = {          # tier -> list of dataset_ids
            DatasetQualityTier.GOLD: [],
            DatasetQualityTier.SILVER: [],
            DatasetQualityTier.EXPERIMENTAL: [],
            DatasetQualityTier.QUARANTINED: []
        }
        
        # Load existing state
        self._load_dataset_state()
    
    def add_dataset(self, metadata: DatasetMetadata, anti_spam_nonce: int = 0) -> Tuple[bool, str]:
        """
        Add new dataset to Chain D following 4-stage pipeline.
        
        Stage 1: Pending - Zero-barrier upload with lightweight anti-spam
        """
        try:
            # Create dataset block
            dataset_block = DatasetBlock(metadata, self.chain_id)
            
            # Basic validation
            is_valid, errors = dataset_block.validate_quality_requirements()
            if not is_valid:
                return False, f"Dataset validation failed: {'; '.join(errors)}"
            
            # Set anti-spam nonce if provided (lightweight spam prevention)
            dataset_block.header.nonce = anti_spam_nonce
            
            # Add to blockchain
            block = self.add_block(
                payload=dataset_block.payload,
                block_type='dataset',
                depends_on=[]  # New datasets typically don't depend on others
            )
            success = block is not None
            
            if success:
                # Track in pending audits for Stage 2 processing
                self.pending_audits[metadata.dataset_id] = dataset_block
                
                # Save dataset state
                self._save_dataset_state()
                
                return True, f"Dataset {metadata.dataset_id} added to Chain D (Stage 1: Pending)"
            else:
                return False, "Failed to add dataset block to chain"
                
        except Exception as e:
            return False, f"Error adding dataset: {str(e)}"
    
    def process_auto_audit(self, dataset_id: str, quality_report) -> Tuple[bool, str]:
        """
        Stage 2: Auto-Audit - AI Quality Gate processing (≤30 minutes)
        """
        if dataset_id not in self.pending_audits:
            return False, f"Dataset {dataset_id} not found in pending audits"
        
        dataset_block = self.pending_audits[dataset_id]
        
        # Update quality report
        dataset_block.metadata.quality_report = quality_report
        
        # Check auto-reject conditions
        is_valid, errors = dataset_block.validate_quality_requirements()
        
        if not is_valid:
            # Auto-reject
            dataset_block.update_stage(
                DatasetStage.REJECTED,
                {'rejection_reason': '; '.join(errors)}
            )
            
            # Remove from pending audits
            del self.pending_audits[dataset_id]
            
            return False, f"Dataset auto-rejected: {'; '.join(errors)}"
        
        else:
            # Passed auto-audit, move to community voting
            dataset_block.update_stage(DatasetStage.COMMUNITY_VOTE)
            
            # Initialize community voting
            self.community_votes[dataset_id] = {
                'votes_for': 0,
                'votes_against': 0,
                'total_voters': 0,
                'voting_started': time.time(),
                'voting_ends': time.time() + (72 * 3600),  # 72 hours
                'voters': {}  # voter_id -> vote
            }
            
            # Move from pending_audits to community_votes tracking
            del self.pending_audits[dataset_id]
            
            self._save_dataset_state()
            
            return True, f"Dataset {dataset_id} passed auto-audit, now in community voting (72h)"
    
    def submit_community_vote(self, dataset_id: str, voter_id: str, vote: bool, 
                            voter_gpu_hwid: str = None) -> Tuple[bool, str]:
        """
        Stage 3: Community Vote - Democratic governance (72 hours)
        
        Args:
            dataset_id: Dataset to vote on
            voter_id: Unique voter identifier
            vote: True for approve, False for reject
            voter_gpu_hwid: GPU hardware ID for 1-account-1-GPU enforcement
        """
        if dataset_id not in self.community_votes:
            return False, f"Dataset {dataset_id} not in community voting stage"
        
        vote_info = self.community_votes[dataset_id]
        
        # Check if voting period is still active
        if time.time() > vote_info['voting_ends']:
            return False, "Voting period has ended"
        
        # Prevent double voting
        if voter_id in vote_info['voters']:
            return False, f"Voter {voter_id} has already voted"
        
        # TODO: Implement GPU-HWID verification for 1-account-1-GPU
        # This would prevent sybil attacks by ensuring one vote per physical GPU
        if voter_gpu_hwid:
            # Check if this GPU has already voted
            existing_voters_with_hwid = [
                v for v in vote_info['voters'].values() 
                if v.get('gpu_hwid') == voter_gpu_hwid
            ]
            if existing_voters_with_hwid:
                return False, f"GPU {voter_gpu_hwid[:8]}... has already voted"
        
        # Record vote
        vote_info['voters'][voter_id] = {
            'vote': vote,
            'timestamp': time.time(),
            'gpu_hwid': voter_gpu_hwid
        }
        
        # Update vote counts
        if vote:
            vote_info['votes_for'] += 1
        else:
            vote_info['votes_against'] += 1
        
        vote_info['total_voters'] += 1
        
        self._save_dataset_state()
        
        return True, f"Vote recorded for {dataset_id}: {'APPROVE' if vote else 'REJECT'}"
    
    def finalize_community_vote(self, dataset_id: str) -> Tuple[bool, str]:
        """
        Finalize community voting and move to Stage 4: Approved/Rejected
        """
        if dataset_id not in self.community_votes:
            return False, f"Dataset {dataset_id} not in community voting"
        
        vote_info = self.community_votes[dataset_id]
        
        # Check if voting period has ended
        if time.time() <= vote_info['voting_ends']:
            return False, "Voting period still active"
        
        # Calculate approval percentage
        total_votes = vote_info['votes_for'] + vote_info['votes_against']
        if total_votes == 0:
            return False, "No votes received - extending voting period by 24h"
        
        approval_rate = vote_info['votes_for'] / total_votes
        
        # Get dataset block to determine tier and finalize
        dataset_block = None
        for block in self.blocks:
            if (hasattr(block, 'metadata') and 
                hasattr(block.metadata, 'dataset_id') and 
                block.metadata.dataset_id == dataset_id):
                dataset_block = DatasetBlock.from_dict({'metadata': asdict(block.metadata)})
                break
        
        if not dataset_block:
            return False, f"Dataset block {dataset_id} not found"
        
        # Determine quality tier based on approval rate and quality metrics
        quality_tier = dataset_block.calculate_quality_tier()
        
        # Apply approval thresholds (from whitepaper specification)
        tier_thresholds = {
            DatasetQualityTier.GOLD: 0.75,          # 75% approval required
            DatasetQualityTier.SILVER: 0.60,        # 60% approval required
            DatasetQualityTier.EXPERIMENTAL: 0.40   # 40% approval required
        }
        
        required_threshold = tier_thresholds.get(quality_tier, 0.50)
        
        if approval_rate >= required_threshold:
            # Approved!
            dataset_block.update_stage(
                DatasetStage.APPROVED,
                {
                    'quality_tier': quality_tier,
                    'community_rating': approval_rate * 5.0,  # Convert to 5-star rating
                    'approval_votes': vote_info['votes_for'],
                    'rejection_votes': vote_info['votes_against'],
                    'final_approval_rate': approval_rate
                }
            )
            
            # Add to appropriate quality tier tracking
            self.quality_tiers[quality_tier].append(dataset_id)
            
            result_msg = f"Dataset {dataset_id} APPROVED as {quality_tier.value.upper()} tier ({approval_rate:.1%} approval)"
            
        else:
            # Rejected
            dataset_block.update_stage(
                DatasetStage.REJECTED,
                {
                    'rejection_reason': f'Insufficient community approval: {approval_rate:.1%} < {required_threshold:.1%}',
                    'approval_votes': vote_info['votes_for'],
                    'rejection_votes': vote_info['votes_against'],
                    'final_approval_rate': approval_rate
                }
            )
            
            result_msg = f"Dataset {dataset_id} REJECTED ({approval_rate:.1%} approval < {required_threshold:.1%} required)"
        
        # Clean up voting tracking
        del self.community_votes[dataset_id]
        
        self._save_dataset_state()
        
        return True, result_msg
    
    def get_datasets_by_tier(self, tier: DatasetQualityTier) -> List[str]:
        """Get all dataset IDs in a specific quality tier."""
        return self.quality_tiers.get(tier, [])
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get complete information about a dataset."""
        # Check in pending audits
        if dataset_id in self.pending_audits:
            dataset_block = self.pending_audits[dataset_id]
            return {
                'dataset_id': dataset_id,
                'stage': dataset_block.metadata.stage.value,
                'metadata': asdict(dataset_block.metadata),
                'in_pending_audit': True
            }
        
        # Check in community votes
        if dataset_id in self.community_votes:
            vote_info = self.community_votes[dataset_id]
            return {
                'dataset_id': dataset_id,
                'stage': 'community_vote',
                'voting_info': vote_info,
                'time_remaining': max(0, vote_info['voting_ends'] - time.time())
            }
        
        # Search in blockchain blocks
        for block in self.blocks:
            # This is a simplified search - in production would use indexing
            try:
                payload_data = json.loads(block.payload.decode('utf-8'))
                if payload_data.get('dataset_id') == dataset_id:
                    return {
                        'dataset_id': dataset_id,
                        'block_hash': block.compute_hash(),
                        'stage': payload_data.get('stage', 'unknown'),
                        'metadata': payload_data
                    }
            except:
                continue
        
        return None
    
    def get_pending_audits_count(self) -> int:
        """Get number of datasets waiting for auto-audit."""
        return len(self.pending_audits)
    
    def get_active_votes_count(self) -> int:
        """Get number of datasets in community voting."""
        return len(self.community_votes)
    
    def get_datasets_by_stage(self, stage: DatasetStage) -> List[str]:
        """Get list of dataset IDs by stage."""
        dataset_ids = []
        
        if stage == DatasetStage.PENDING:
            dataset_ids = list(self.pending_audits.keys())
        elif stage == DatasetStage.COMMUNITY_VOTE:
            dataset_ids = list(self.community_votes.keys())
        else:
            # For approved/rejected, would need to scan blockchain
            # For now, return empty list
            dataset_ids = []
        
        return dataset_ids
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Chain D statistics."""
        all_blocks = self.get_all_blocks()
        stats = {
            'total_datasets': len(all_blocks),
            'pending_audits': len(self.pending_audits),
            'active_votes': len(self.community_votes),
            'quality_tiers': {
                tier.value: len(dataset_ids) 
                for tier, dataset_ids in self.quality_tiers.items()
            },
            'chain_id': self.chain_id,
            'latest_block_index': len(all_blocks) - 1 if all_blocks else -1
        }
        
        return stats
    
    def _save_dataset_state(self):
        """Save dataset-specific state to disk."""
        state_file = self.storage.dir_path / "dataset_state.json"
        
        # Convert quality_tiers keys to strings for JSON serialization
        serializable_quality_tiers = {
            tier.value: dataset_ids 
            for tier, dataset_ids in self.quality_tiers.items()
        }
        
        state = {
            'pending_audits': {
                dataset_id: asdict(dataset_block.metadata)
                for dataset_id, dataset_block in self.pending_audits.items()
            },
            'community_votes': self.community_votes,
            'quality_tiers': serializable_quality_tiers
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_dataset_state(self):
        """Load dataset-specific state from disk."""
        state_file = self.storage.dir_path / "dataset_state.json"
        
        if not state_file.exists():
            return  # No existing state
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore pending_audits
            self.pending_audits = {}
            for dataset_id, metadata_dict in state.get('pending_audits', {}).items():
                # Reconstruct DatasetBlock from metadata
                metadata = DatasetMetadata(**metadata_dict)
                self.pending_audits[dataset_id] = DatasetBlock(metadata)
            
            # Restore community_votes
            self.community_votes = state.get('community_votes', {})
            
            # Restore quality_tiers (convert string keys back to enums)
            quality_tiers_data = state.get('quality_tiers', {})
            self.quality_tiers = {
                DatasetQualityTier(tier_str): dataset_ids
                for tier_str, dataset_ids in quality_tiers_data.items()
            }
            
        except Exception as e:
            print(f"Warning: Could not load dataset state: {e}")
            # Continue with empty state