"""
Delta Evolution Tracker for Blyan
Tracks and analyzes delta block performance, contribution, and rejection patterns
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np

from .pol import EnhancedPoLValidator, PoLScore
from .delta_compression import DeltaBase

class DeltaStatus(Enum):
    """Status of a delta block."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    REVERTED = "reverted"

class RejectionReason(Enum):
    """Reasons for delta rejection."""
    POL_FAILURE = "pol_failure"
    FRAUD_DETECTED = "fraud_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    VALIDATION_TIMEOUT = "validation_timeout"
    TECHNICAL_ERROR = "technical_error"
    CONSENSUS_FAILURE = "consensus_failure"

@dataclass
class DeltaRecord:
    """Complete record of a delta submission and its lifecycle."""
    delta_id: str
    tile_id: str
    expert_name: str
    submitter_node_id: str
    submission_timestamp: float
    
    # Version control (CAS)
    base_block_hash: Optional[str] = None  # Base version this delta was computed from
    target_block_hash: Optional[str] = None  # Target version after applying delta
    
    # Delta content
    delta_size_bytes: int
    compression_method: str
    compression_ratio: float
    
    # Validation results
    pol_score: Optional[PoLScore] = None
    validation_timestamp: Optional[float] = None
    status: DeltaStatus = DeltaStatus.PENDING
    rejection_reason: Optional[RejectionReason] = None
    
    # Performance metrics
    inference_improvement: float = 0.0
    processing_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    
    # Lifecycle tracking
    acceptance_timestamp: Optional[float] = None
    application_timestamp: Optional[float] = None
    reversion_timestamp: Optional[float] = None
    
    # Impact metrics
    downstream_applications: int = 0
    cumulative_usage_count: int = 0
    quality_degradation_reports: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def lifecycle_duration(self) -> float:
        """Total time from submission to final status."""
        end_time = (
            self.reversion_timestamp or 
            self.application_timestamp or 
            self.acceptance_timestamp or 
            time.time()
        )
        return end_time - self.submission_timestamp
    
    @property
    def is_successful(self) -> bool:
        """Whether delta was ultimately successful."""
        return (
            self.status == DeltaStatus.ACCEPTED and 
            self.reversion_timestamp is None and
            self.inference_improvement > 0
        )

@dataclass
class NodeContributionStats:
    """Statistics for a contributing node."""
    node_id: str
    total_submissions: int = 0
    successful_submissions: int = 0
    rejected_submissions: int = 0
    quarantined_submissions: int = 0
    
    # Quality metrics
    avg_pol_score: float = 0.0
    avg_improvement: float = 0.0
    fraud_detection_rate: float = 0.0
    
    # Performance metrics
    avg_submission_size: float = 0.0
    avg_processing_time: float = 0.0
    total_contribution_value: float = 0.0
    
    # Recent performance (sliding window)
    recent_submissions: deque = field(default_factory=lambda: deque(maxlen=50))
    trust_score: float = 0.5
    
    @property
    def success_rate(self) -> float:
        """Success rate for this node."""
        if self.total_submissions == 0:
            return 0.0
        return self.successful_submissions / self.total_submissions
    
    @property
    def rejection_rate(self) -> float:
        """Rejection rate for this node."""
        if self.total_submissions == 0:
            return 0.0
        return self.rejected_submissions / self.total_submissions

@dataclass
class ExpertEvolutionStats:
    """Evolution statistics for an expert."""
    expert_name: str
    tile_id: str
    total_deltas_received: int = 0
    successful_deltas: int = 0
    cumulative_improvement: float = 0.0
    
    # Evolution patterns
    improvement_trajectory: List[float] = field(default_factory=list)
    quality_trend: List[float] = field(default_factory=list)
    contributor_diversity: Set[str] = field(default_factory=set)
    
    # Performance metrics
    avg_delta_size: float = 0.0
    avg_validation_time: float = 0.0
    last_successful_update: Optional[float] = None
    
    @property
    def evolution_velocity(self) -> float:
        """Rate of successful improvements over time."""
        if len(self.improvement_trajectory) < 2:
            return 0.0
        
        recent_improvements = self.improvement_trajectory[-10:]  # Last 10
        if len(recent_improvements) < 2:
            return 0.0
        
        # Calculate average improvement per update
        return np.mean(recent_improvements)
    
    @property
    def contributor_count(self) -> int:
        """Number of unique contributors."""
        return len(self.contributor_diversity)

class DeltaEvolutionTracker:
    """
    Comprehensive tracker for delta evolution and performance.
    
    Features:
    - Complete delta lifecycle tracking
    - Node contribution analysis
    - Expert evolution patterns
    - Performance trend analysis
    - Fraud and quality monitoring
    """
    
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 pol_validator: Optional[EnhancedPoLValidator] = None):
        
        self.data_dir = data_dir or Path("./data/delta_evolution")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pol_validator = pol_validator
        
        # Core tracking data
        self.delta_records: Dict[str, DeltaRecord] = {}
        self.node_stats: Dict[str, NodeContributionStats] = {}
        self.expert_stats: Dict[str, ExpertEvolutionStats] = {}
        
        # Performance monitoring
        self.rejection_patterns: Dict[RejectionReason, List[str]] = defaultdict(list)
        self.quality_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.last_analysis_time: float = 0.0
        self.analysis_interval: float = 300.0  # 5 minutes
        
        # Load existing data
        self._load_tracking_data()
        
        print(f"📈 Delta Evolution Tracker initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Tracking {len(self.delta_records)} delta records")
        print(f"   Monitoring {len(self.node_stats)} nodes")
        print(f"   Following {len(self.expert_stats)} experts")
    
    def track_delta_submission(self, 
                              delta_id: str,
                              tile_id: str,
                              expert_name: str,
                              submitter_node_id: str,
                              delta: DeltaBase,
                              metadata: Optional[Dict[str, Any]] = None) -> DeltaRecord:
        """Track a new delta submission."""
        
        # Create delta record
        record = DeltaRecord(
            delta_id=delta_id,
            tile_id=tile_id,
            expert_name=expert_name,
            submitter_node_id=submitter_node_id,
            submission_timestamp=time.time(),
            delta_size_bytes=len(delta.to_bytes()),
            compression_method=type(delta).__name__,
            compression_ratio=delta.get_compression_ratio(),
            metadata=metadata or {}
        )
        
        self.delta_records[delta_id] = record
        
        # Update node statistics
        if submitter_node_id not in self.node_stats:
            self.node_stats[submitter_node_id] = NodeContributionStats(node_id=submitter_node_id)
        
        node_stats = self.node_stats[submitter_node_id]
        node_stats.total_submissions += 1
        node_stats.recent_submissions.append({
            'delta_id': delta_id,
            'timestamp': record.submission_timestamp,
            'size': record.delta_size_bytes
        })
        
        # Update expert statistics
        if expert_name not in self.expert_stats:
            self.expert_stats[expert_name] = ExpertEvolutionStats(
                expert_name=expert_name,
                tile_id=tile_id
            )
        
        expert_stats = self.expert_stats[expert_name]
        expert_stats.total_deltas_received += 1
        expert_stats.contributor_diversity.add(submitter_node_id)
        
        # Save state
        self._save_tracking_data()
        
        print(f"📥 Tracking delta submission: {delta_id}")
        print(f"   Submitter: {submitter_node_id}")
        print(f"   Expert: {expert_name}")
        print(f"   Size: {record.delta_size_bytes} bytes")
        
        return record
    
    def update_delta_validation(self, 
                               delta_id: str,
                               pol_score: PoLScore,
                               processing_time_ms: float = 0.0,
                               validation_time_ms: float = 0.0):
        """Update delta record with validation results."""
        
        if delta_id not in self.delta_records:
            print(f"⚠️ Delta {delta_id} not found for validation update")
            return
        
        record = self.delta_records[delta_id]
        record.pol_score = pol_score
        record.validation_timestamp = time.time()
        record.processing_time_ms = processing_time_ms
        record.validation_time_ms = validation_time_ms
        record.inference_improvement = pol_score.improvement_score
        
        # Determine status based on PoL score
        if pol_score.is_valid:
            record.status = DeltaStatus.ACCEPTED
            record.acceptance_timestamp = time.time()
            
            # Update success statistics
            node_stats = self.node_stats[record.submitter_node_id]
            node_stats.successful_submissions += 1
            node_stats.total_contribution_value += pol_score.improvement_score
            
            expert_stats = self.expert_stats[record.expert_name]
            expert_stats.successful_deltas += 1
            expert_stats.cumulative_improvement += pol_score.improvement_score
            expert_stats.improvement_trajectory.append(pol_score.improvement_score)
            expert_stats.last_successful_update = time.time()
            
        else:
            record.status = DeltaStatus.REJECTED
            
            # Determine rejection reason
            if pol_score.fraud_probability > 0.7:
                record.rejection_reason = RejectionReason.FRAUD_DETECTED
            elif pol_score.improvement_score < -0.1:
                record.rejection_reason = RejectionReason.PERFORMANCE_DEGRADATION
            else:
                record.rejection_reason = RejectionReason.POL_FAILURE
            
            # Update rejection statistics
            node_stats = self.node_stats[record.submitter_node_id]
            node_stats.rejected_submissions += 1
            
            # Track rejection patterns
            self.rejection_patterns[record.rejection_reason].append(delta_id)
        
        # Update quality trends
        self.quality_trends[record.expert_name].append(pol_score.improvement_score)
        
        # Update node trust score
        self._update_node_trust_score(record.submitter_node_id)
        
        # Save state
        self._save_tracking_data()
        
        print(f"✅ Updated validation for delta {delta_id}")
        print(f"   Status: {record.status.value}")
        print(f"   PoL Score: {pol_score.improvement_score:.4f}")
        if record.rejection_reason:
            print(f"   Rejection Reason: {record.rejection_reason.value}")
    
    def track_delta_application(self, delta_id: str):
        """Track when a delta is actually applied to the blockchain."""
        if delta_id not in self.delta_records:
            return
        
        record = self.delta_records[delta_id]
        record.application_timestamp = time.time()
        
        # Update expert evolution stats
        expert_stats = self.expert_stats[record.expert_name]
        expert_stats.quality_trend.append(record.inference_improvement)
        
        print(f"🔗 Delta {delta_id} applied to blockchain")
    
    def track_delta_reversion(self, delta_id: str, reason: str = ""):
        """Track when a delta is reverted due to problems."""
        if delta_id not in self.delta_records:
            return
        
        record = self.delta_records[delta_id]
        record.status = DeltaStatus.REVERTED
        record.reversion_timestamp = time.time()
        record.metadata['reversion_reason'] = reason
        
        # Update node statistics (penalty for reverted deltas)
        node_stats = self.node_stats[record.submitter_node_id]
        node_stats.trust_score = max(0.1, node_stats.trust_score - 0.1)
        
        print(f"🔄 Delta {delta_id} reverted: {reason}")
    
    def _update_node_trust_score(self, node_id: str):
        """Update trust score for a node based on recent performance."""
        if node_id not in self.node_stats:
            return
        
        stats = self.node_stats[node_id]
        
        # Base trust on success rate
        base_trust = stats.success_rate
        
        # Adjust for fraud detection rate
        fraud_penalty = stats.fraud_detection_rate * 0.5
        
        # Adjust for recent performance
        if stats.recent_submissions:
            recent_delta_ids = [sub['delta_id'] for sub in stats.recent_submissions]
            recent_records = [self.delta_records[did] for did in recent_delta_ids if did in self.delta_records]
            
            recent_success_rate = sum(1 for r in recent_records if r.is_successful) / len(recent_records)
            
            # Weight recent performance more heavily
            adjusted_trust = base_trust * 0.3 + recent_success_rate * 0.7
        else:
            adjusted_trust = base_trust
        
        # Apply fraud penalty
        final_trust = max(0.0, min(1.0, adjusted_trust - fraud_penalty))
        
        stats.trust_score = final_trust
    
    def analyze_evolution_patterns(self) -> Dict[str, Any]:
        """Analyze evolution patterns across all experts."""
        current_time = time.time()
        
        if current_time - self.last_analysis_time < self.analysis_interval:
            return self.analysis_cache.get('evolution_analysis', {})
        
        analysis = {
            'global_metrics': self._calculate_global_metrics(),
            'expert_rankings': self._rank_experts_by_evolution(),
            'node_performance': self._analyze_node_performance(),
            'rejection_analysis': self._analyze_rejection_patterns(),
            'quality_trends': self._analyze_quality_trends(),
            'recommendations': self._generate_evolution_recommendations()
        }
        
        self.analysis_cache['evolution_analysis'] = analysis
        self.last_analysis_time = current_time
        
        return analysis
    
    def _calculate_global_metrics(self) -> Dict[str, Any]:
        """Calculate global evolution metrics."""
        total_deltas = len(self.delta_records)
        successful_deltas = sum(1 for r in self.delta_records.values() if r.is_successful)
        
        if total_deltas == 0:
            return {'error': 'No delta data available'}
        
        # Overall success rate
        global_success_rate = successful_deltas / total_deltas
        
        # Average improvement
        improvements = [r.inference_improvement for r in self.delta_records.values() if r.inference_improvement > 0]
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # Processing efficiency
        processing_times = [r.processing_time_ms for r in self.delta_records.values() if r.processing_time_ms > 0]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        return {
            'total_deltas': total_deltas,
            'successful_deltas': successful_deltas,
            'global_success_rate': global_success_rate,
            'avg_improvement': avg_improvement,
            'avg_processing_time_ms': avg_processing_time,
            'unique_contributors': len(self.node_stats),
            'active_experts': len(self.expert_stats)
        }
    
    def _rank_experts_by_evolution(self) -> List[Dict[str, Any]]:
        """Rank experts by evolution performance."""
        expert_rankings = []
        
        for expert_name, stats in self.expert_stats.items():
            if stats.total_deltas_received == 0:
                continue
            
            # Calculate composite evolution score
            success_rate = stats.successful_deltas / stats.total_deltas_received
            velocity = stats.evolution_velocity
            diversity = stats.contributor_count
            improvement = stats.cumulative_improvement
            
            # Weighted score
            evolution_score = (
                success_rate * 0.3 +
                min(1.0, velocity * 10) * 0.2 +  # Cap velocity impact
                min(1.0, diversity / 10) * 0.2 +  # Cap diversity impact
                min(1.0, improvement * 2) * 0.3   # Cap improvement impact
            )
            
            expert_rankings.append({
                'expert_name': expert_name,
                'evolution_score': evolution_score,
                'success_rate': success_rate,
                'evolution_velocity': velocity,
                'contributor_count': diversity,
                'cumulative_improvement': improvement,
                'total_deltas': stats.total_deltas_received
            })
        
        return sorted(expert_rankings, key=lambda x: x['evolution_score'], reverse=True)
    
    def _analyze_node_performance(self) -> Dict[str, Any]:
        """Analyze node contribution performance."""
        node_analysis = {}
        
        for node_id, stats in self.node_stats.items():
            if stats.total_submissions == 0:
                continue
            
            node_analysis[node_id] = {
                'total_submissions': stats.total_submissions,
                'success_rate': stats.success_rate,
                'rejection_rate': stats.rejection_rate,
                'trust_score': stats.trust_score,
                'avg_contribution_value': stats.total_contribution_value / max(1, stats.successful_submissions),
                'fraud_detection_rate': stats.fraud_detection_rate,
                'performance_category': self._categorize_node_performance(stats)
            }
        
        return node_analysis
    
    def _categorize_node_performance(self, stats: NodeContributionStats) -> str:
        """Categorize node performance."""
        if stats.success_rate > 0.8 and stats.trust_score > 0.8:
            return "excellent"
        elif stats.success_rate > 0.6 and stats.trust_score > 0.6:
            return "good"
        elif stats.success_rate > 0.4 and stats.trust_score > 0.4:
            return "average"
        elif stats.fraud_detection_rate > 0.3:
            return "suspicious"
        else:
            return "poor"
    
    def _analyze_rejection_patterns(self) -> Dict[str, Any]:
        """Analyze delta rejection patterns."""
        rejection_analysis = {}
        
        for reason, delta_ids in self.rejection_patterns.items():
            if not delta_ids:
                continue
            
            # Get records for rejected deltas
            rejected_records = [self.delta_records[did] for did in delta_ids if did in self.delta_records]
            
            if not rejected_records:
                continue
            
            # Analyze patterns
            submitter_counts = defaultdict(int)
            expert_counts = defaultdict(int)
            avg_size = np.mean([r.delta_size_bytes for r in rejected_records])
            
            for record in rejected_records:
                submitter_counts[record.submitter_node_id] += 1
                expert_counts[record.expert_name] += 1
            
            rejection_analysis[reason.value] = {
                'total_rejections': len(rejected_records),
                'avg_delta_size': avg_size,
                'top_submitters': dict(sorted(submitter_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                'affected_experts': dict(sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            }
        
        return rejection_analysis
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        trends = {}
        
        for expert_name, quality_scores in self.quality_trends.items():
            if len(quality_scores) < 3:
                continue
            
            scores_list = list(quality_scores)
            
            # Calculate trend
            if len(scores_list) >= 2:
                recent_avg = np.mean(scores_list[-10:])  # Last 10
                early_avg = np.mean(scores_list[:10])    # First 10
                trend_direction = "improving" if recent_avg > early_avg else "declining"
            else:
                trend_direction = "insufficient_data"
            
            trends[expert_name] = {
                'total_updates': len(scores_list),
                'avg_quality': np.mean(scores_list),
                'quality_std': np.std(scores_list),
                'trend_direction': trend_direction,
                'recent_performance': np.mean(scores_list[-5:]) if len(scores_list) >= 5 else 0.0
            }
        
        return trends
    
    def _generate_evolution_recommendations(self) -> List[str]:
        """Generate recommendations for improving evolution."""
        recommendations = []
        
        global_metrics = self._calculate_global_metrics()
        
        # Success rate recommendations
        if global_metrics.get('global_success_rate', 0) < 0.6:
            recommendations.append(
                "Low global success rate detected - consider tightening PoL validation criteria"
            )
        
        # Node performance recommendations
        suspicious_nodes = [
            node_id for node_id, stats in self.node_stats.items()
            if stats.fraud_detection_rate > 0.3 or stats.trust_score < 0.3
        ]
        
        if suspicious_nodes:
            recommendations.append(
                f"Monitor {len(suspicious_nodes)} nodes with low trust scores: {suspicious_nodes[:3]}"
            )
        
        # Expert evolution recommendations
        stagnant_experts = [
            name for name, stats in self.expert_stats.items()
            if stats.evolution_velocity < 0.01 and stats.total_deltas_received > 10
        ]
        
        if stagnant_experts:
            recommendations.append(
                f"Consider incentivizing contributions to {len(stagnant_experts)} stagnant experts"
            )
        
        return recommendations
    
    def export_evolution_report(self, format: str = 'json') -> str:
        """Export comprehensive evolution report."""
        report_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'tracking_period': self._get_tracking_period(),
                'report_version': '1.0'
            },
            'analysis': self.analyze_evolution_patterns(),
            'raw_data': {
                'delta_count': len(self.delta_records),
                'node_count': len(self.node_stats),
                'expert_count': len(self.expert_stats)
            }
        }
        
        if format.lower() == 'json':
            export_file = self.data_dir / f"evolution_report_{int(time.time())}.json"
            with open(export_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            print(f"📈 Evolution report exported to {export_file}")
            return str(export_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_tracking_period(self) -> Dict[str, float]:
        """Get the time period covered by tracking data."""
        if not self.delta_records:
            return {'start': 0.0, 'end': 0.0, 'duration': 0.0}
        
        timestamps = [r.submission_timestamp for r in self.delta_records.values()]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }
    
    def _save_tracking_data(self):
        """Save tracking data to disk."""
        try:
            # Save delta records
            delta_file = self.data_dir / "delta_records.json"
            with open(delta_file, 'w') as f:
                json.dump(
                    {k: v.__dict__ for k, v in self.delta_records.items()}, 
                    f, indent=2, default=str
                )
            
            # Save node stats
            node_file = self.data_dir / "node_stats.json"
            with open(node_file, 'w') as f:
                json.dump(
                    {k: v.__dict__ for k, v in self.node_stats.items()}, 
                    f, indent=2, default=str
                )
            
            # Save expert stats
            expert_file = self.data_dir / "expert_stats.json"  
            with open(expert_file, 'w') as f:
                json.dump(
                    {k: v.__dict__ for k, v in self.expert_stats.items()}, 
                    f, indent=2, default=str
                )
                
        except Exception as e:
            print(f"⚠️ Failed to save tracking data: {e}")
    
    def _load_tracking_data(self):
        """Load existing tracking data from disk."""
        try:
            # Load delta records
            delta_file = self.data_dir / "delta_records.json"
            if delta_file.exists():
                with open(delta_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        record = DeltaRecord(**v)
                        self.delta_records[k] = record
            
            # Load node stats
            node_file = self.data_dir / "node_stats.json"
            if node_file.exists():
                with open(node_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        stats = NodeContributionStats(**v)  
                        self.node_stats[k] = stats
            
            # Load expert stats
            expert_file = self.data_dir / "expert_stats.json"
            if expert_file.exists():
                with open(expert_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        stats = ExpertEvolutionStats(**v)
                        self.expert_stats[k] = stats
                        
        except Exception as e:
            print(f"⚠️ Failed to load tracking data: {e}")

# Export main classes
__all__ = ['DeltaEvolutionTracker', 'DeltaRecord', 'NodeContributionStats', 'ExpertEvolutionStats', 'DeltaStatus', 'RejectionReason']