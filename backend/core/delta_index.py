"""
Delta Index for tracking layer deltas and checkpoints in dense LLM learning.

This module implements the core storage and retrieval logic for layer deltas,
enabling efficient composition of base layers with incremental updates.
Supports Proof of Learning by maintaining verifiable lineage of all weight updates.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeltaRecord:
    """Record of a single delta update for a layer."""
    delta_hash: str  # Block hash of the delta
    base_hash: str  # Hash of the base layer this delta applies to
    result_hash: Optional[str]  # Hash of layer after applying delta
    training_round_id: str  # Round that produced this delta
    timestamp: float  # When delta was created
    validation_score: float  # PoL validation score
    compression: str  # Compression type: sparse, dense, differential
    sparsity_ratio: float  # Ratio of non-zero values (for LoRA)
    size_bytes: int  # Size of delta in bytes
    trainer_id: str  # GPU node that created this delta
    metadata: Dict[str, Any]  # Additional metadata (LoRA config, etc)


@dataclass
class CheckpointRecord:
    """Record of a full layer checkpoint."""
    checkpoint_hash: str  # Block hash of the checkpoint
    layer_name: str  # Layer identifier
    timestamp: float  # When checkpoint was created
    round_id: str  # Training round that triggered checkpoint
    size_bytes: int  # Size of checkpoint
    deltas_merged: List[str]  # Delta hashes merged into this checkpoint
    validation_score: float  # Aggregate validation score


class DeltaIndex:
    """
    Manages the index of layer deltas and checkpoints for dense model learning.
    
    This is the learning equivalent of ParameterIndex, tracking the evolution
    of model weights through training rounds with full lineage preservation.
    """
    
    def __init__(self, path: Path):
        """
        Initialize DeltaIndex with persistent storage.
        
        Args:
            path: Path to JSON storage file
        """
        self.path = path
        self._deltas: Dict[str, List[DeltaRecord]] = defaultdict(list)  # layer_name -> deltas
        self._checkpoints: Dict[str, List[CheckpointRecord]] = defaultdict(list)  # layer_name -> checkpoints
        self._current_base: Dict[str, str] = {}  # layer_name -> current base hash
        self._round_deltas: Dict[str, List[str]] = defaultdict(list)  # round_id -> delta hashes
        self._load()
    
    def _load(self) -> None:
        """Load index from persistent storage."""
        if self.path.exists():
            try:
                with self.path.open() as fp:
                    data = json.load(fp)
                    
                    # Load deltas
                    for layer_name, delta_list in data.get('deltas', {}).items():
                        self._deltas[layer_name] = [
                            DeltaRecord(**d) for d in delta_list
                        ]
                    
                    # Load checkpoints
                    for layer_name, checkpoint_list in data.get('checkpoints', {}).items():
                        self._checkpoints[layer_name] = [
                            CheckpointRecord(**c) for c in checkpoint_list
                        ]
                    
                    # Load current bases
                    self._current_base = data.get('current_base', {})
                    
                    # Load round mappings
                    self._round_deltas = defaultdict(list, data.get('round_deltas', {}))
                    
                logger.info(f"Loaded delta index with {sum(len(d) for d in self._deltas.values())} deltas")
            except Exception as e:
                logger.error(f"Failed to load delta index: {e}")
                self._deltas = defaultdict(list)
                self._checkpoints = defaultdict(list)
                self._current_base = {}
                self._round_deltas = defaultdict(list)
    
    def _save(self) -> None:
        """Save index to persistent storage."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'deltas': {
                layer: [asdict(d) for d in delta_list]
                for layer, delta_list in self._deltas.items()
            },
            'checkpoints': {
                layer: [asdict(c) for c in checkpoint_list]
                for layer, checkpoint_list in self._checkpoints.items()
            },
            'current_base': self._current_base,
            'round_deltas': dict(self._round_deltas),
            'updated_at': time.time()
        }
        
        with self.path.open('w') as fp:
            json.dump(data, fp, indent=2)
    
    def add_delta(
        self,
        layer_name: str,
        delta_hash: str,
        base_hash: str,
        training_round_id: str,
        validation_score: float,
        trainer_id: str,
        compression: str = 'dense',
        sparsity_ratio: float = 1.0,
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new delta record for a layer.
        
        Args:
            layer_name: Name of the layer (e.g., "layer_15")
            delta_hash: Block hash of the delta
            base_hash: Hash of the base layer
            training_round_id: Round that created this delta
            validation_score: PoL validation score
            trainer_id: GPU node identifier
            compression: Compression type
            sparsity_ratio: Ratio of non-zero values
            size_bytes: Size of delta
            metadata: Additional metadata
        """
        record = DeltaRecord(
            delta_hash=delta_hash,
            base_hash=base_hash,
            result_hash=None,  # Computed lazily
            training_round_id=training_round_id,
            timestamp=time.time(),
            validation_score=validation_score,
            compression=compression,
            sparsity_ratio=sparsity_ratio,
            size_bytes=size_bytes,
            trainer_id=trainer_id,
            metadata=metadata or {}
        )
        
        self._deltas[layer_name].append(record)
        self._round_deltas[training_round_id].append(delta_hash)
        
        logger.info(f"Added delta for {layer_name}: {delta_hash[:8]}... (score: {validation_score:.4f})")
        self._save()
    
    def add_checkpoint(
        self,
        layer_name: str,
        checkpoint_hash: str,
        round_id: str,
        deltas_merged: List[str],
        validation_score: float,
        size_bytes: int
    ) -> None:
        """
        Add a checkpoint record for a layer.
        
        Args:
            layer_name: Name of the layer
            checkpoint_hash: Block hash of the checkpoint
            round_id: Training round that triggered checkpoint
            deltas_merged: List of delta hashes merged
            validation_score: Aggregate validation score
            size_bytes: Size of checkpoint
        """
        record = CheckpointRecord(
            checkpoint_hash=checkpoint_hash,
            layer_name=layer_name,
            timestamp=time.time(),
            round_id=round_id,
            size_bytes=size_bytes,
            deltas_merged=deltas_merged,
            validation_score=validation_score
        )
        
        self._checkpoints[layer_name].append(record)
        self._current_base[layer_name] = checkpoint_hash
        
        # Clear merged deltas
        self._deltas[layer_name] = [
            d for d in self._deltas[layer_name]
            if d.delta_hash not in deltas_merged
        ]
        
        logger.info(f"Added checkpoint for {layer_name}: {checkpoint_hash[:8]}... (merged {len(deltas_merged)} deltas)")
        self._save()
    
    def get_layer_deltas(self, layer_name: str, base_hash: Optional[str] = None) -> List[DeltaRecord]:
        """
        Get all deltas for a layer, optionally filtered by base hash.
        
        Args:
            layer_name: Name of the layer
            base_hash: Optional base hash to filter by
            
        Returns:
            List of delta records
        """
        deltas = self._deltas.get(layer_name, [])
        
        if base_hash:
            deltas = [d for d in deltas if d.base_hash == base_hash]
        
        # Sort by validation score (highest first)
        return sorted(deltas, key=lambda d: d.validation_score, reverse=True)
    
    def get_best_delta(self, layer_name: str, base_hash: str) -> Optional[DeltaRecord]:
        """
        Get the best delta for a layer based on validation score.
        
        Args:
            layer_name: Name of the layer
            base_hash: Base hash to match
            
        Returns:
            Best delta record or None
        """
        deltas = self.get_layer_deltas(layer_name, base_hash)
        return deltas[0] if deltas else None
    
    def get_current_base(self, layer_name: str) -> Optional[str]:
        """
        Get the current base hash for a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Current base hash or None
        """
        return self._current_base.get(layer_name)
    
    def get_latest_checkpoint(self, layer_name: str) -> Optional[CheckpointRecord]:
        """
        Get the latest checkpoint for a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Latest checkpoint record or None
        """
        checkpoints = self._checkpoints.get(layer_name, [])
        return checkpoints[-1] if checkpoints else None
    
    def get_round_deltas(self, round_id: str) -> List[str]:
        """
        Get all delta hashes for a training round.
        
        Args:
            round_id: Training round identifier
            
        Returns:
            List of delta hashes
        """
        return self._round_deltas.get(round_id, [])
    
    def get_delta_chain(self, layer_name: str, target_hash: str) -> List[DeltaRecord]:
        """
        Get the chain of deltas needed to reach a target state.
        
        Args:
            layer_name: Name of the layer
            target_hash: Target state hash
            
        Returns:
            Ordered list of deltas to apply
        """
        chain = []
        current = target_hash
        
        # Walk backwards through delta dependencies
        while current:
            found = False
            for delta in self._deltas.get(layer_name, []):
                if delta.result_hash == current:
                    chain.insert(0, delta)
                    current = delta.base_hash
                    found = True
                    break
            
            if not found:
                # Check if current is a checkpoint
                for checkpoint in self._checkpoints.get(layer_name, []):
                    if checkpoint.checkpoint_hash == current:
                        return chain  # Reached a checkpoint base
                break
        
        return chain
    
    def compute_delta_hash(self, base_hash: str, delta_hash: str) -> str:
        """
        Compute the result hash after applying a delta.
        
        Args:
            base_hash: Base layer hash
            delta_hash: Delta hash
            
        Returns:
            Result hash
        """
        # Deterministic hash computation
        combined = f"{base_hash}:{delta_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the delta index.
        
        Returns:
            Dictionary of statistics
        """
        total_deltas = sum(len(d) for d in self._deltas.values())
        total_checkpoints = sum(len(c) for c in self._checkpoints.values())
        total_size = sum(
            sum(d.size_bytes for d in delta_list)
            for delta_list in self._deltas.values()
        )
        
        layer_stats = {}
        for layer_name in set(list(self._deltas.keys()) + list(self._checkpoints.keys())):
            deltas = self._deltas.get(layer_name, [])
            checkpoints = self._checkpoints.get(layer_name, [])
            
            layer_stats[layer_name] = {
                'deltas': len(deltas),
                'checkpoints': len(checkpoints),
                'avg_validation_score': sum(d.validation_score for d in deltas) / len(deltas) if deltas else 0,
                'total_size_mb': sum(d.size_bytes for d in deltas) / 1024 / 1024 if deltas else 0
            }
        
        return {
            'total_deltas': total_deltas,
            'total_checkpoints': total_checkpoints,
            'total_rounds': len(self._round_deltas),
            'total_size_gb': total_size / 1024 / 1024 / 1024,
            'layers_with_deltas': len(self._deltas),
            'layer_statistics': layer_stats
        }
    
    def prune_old_deltas(self, keep_last_n: int = 10) -> int:
        """
        Prune old deltas keeping only the last N per layer.
        
        Args:
            keep_last_n: Number of deltas to keep per layer
            
        Returns:
            Number of deltas pruned
        """
        pruned = 0
        
        for layer_name in self._deltas:
            deltas = self._deltas[layer_name]
            if len(deltas) > keep_last_n:
                # Sort by timestamp and keep newest
                deltas.sort(key=lambda d: d.timestamp)
                to_remove = len(deltas) - keep_last_n
                self._deltas[layer_name] = deltas[to_remove:]
                pruned += to_remove
        
        if pruned > 0:
            logger.info(f"Pruned {pruned} old deltas")
            self._save()
        
        return pruned