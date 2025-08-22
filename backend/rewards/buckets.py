#!/usr/bin/env python3
"""
Dynamic Budget Bucket Allocator
Manages daily BLY budget allocation across inference, learning, validation, and dataset rewards.
Implements floors, ceilings, rollovers, and backpay queues.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import threading
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BucketConfig:
    """Configuration for budget buckets."""
    daily_budget: float = 273_972.0
    
    # Target allocations
    splits: Dict[str, float] = field(default_factory=lambda: {
        'inference': 0.45,
        'learning': 0.35,
        'validation': 0.10,
        'dataset': 0.10
    })
    
    # Minimum guaranteed allocations
    floors: Dict[str, float] = field(default_factory=lambda: {
        'inference': 0.30,
        'learning': 0.25,
        'validation': 0.05,
        'dataset': 0.05
    })
    
    # Maximum allowed allocations
    ceilings: Dict[str, float] = field(default_factory=lambda: {
        'inference': 0.60,
        'learning': 0.50,
        'validation': 0.15,
        'dataset': 0.15
    })
    
    # Rollover settings
    rollover_hours: int = 24
    max_accumulation_factor: float = 3.0
    
    # Backpay settings
    backpay_enabled: bool = True
    max_queue_days: int = 7
    backpay_priority: List[str] = field(default_factory=lambda: [
        'learning', 'validation', 'dataset', 'inference'
    ])
    
    @classmethod
    def from_yaml(cls, path: str = "config/reward_policy.yaml"):
        """Load configuration from YAML."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            daily_budget=data.get('daily_budget_bly', 273_972.0),
            splits=data.get('bucket_split', cls().splits),
            floors=data.get('bucket_floor_pct', cls().floors),
            ceilings=data.get('bucket_ceiling_pct', cls().ceilings),
            rollover_hours=data['rollover'].get('hours', 24),
            max_accumulation_factor=data['rollover'].get('max_accumulation', 3.0),
            backpay_enabled=data['backpay'].get('enabled', True),
            max_queue_days=data['backpay'].get('max_queue_days', 7),
            backpay_priority=data['backpay'].get('priority', cls().backpay_priority)
        )


@dataclass
class BackpayRequest:
    """A queued reward request waiting for budget."""
    request_id: str
    bucket_type: str
    amount_bly: float
    requester: str
    timestamp: float
    metadata: Dict = field(default_factory=dict)
    attempts: int = 0
    
    def is_expired(self, max_days: int = 7) -> bool:
        """Check if request has expired."""
        age_days = (time.time() - self.timestamp) / 86400
        return age_days > max_days


class BucketAllocator:
    """
    Manages dynamic budget allocation with floors, ceilings, and rollovers.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, config: Optional[BucketConfig] = None, 
                 state_path: str = "data/bucket_state.json"):
        """Initialize bucket allocator."""
        self.config = config or BucketConfig.from_yaml()
        self.state_path = Path(state_path)
        self.lock = threading.RLock()
        
        # Current epoch tracking
        self.current_epoch = self._get_current_epoch()
        
        # Budget buckets (available BLY)
        self.buckets: Dict[str, float] = {}
        
        # Accumulated rollover amounts
        self.rollover: Dict[str, float] = {}
        
        # Backpay queues
        self.backpay_queues: Dict[str, deque] = {
            bucket: deque() for bucket in self.config.splits.keys()
        }
        
        # Usage tracking
        self.usage_history: Dict[str, List[float]] = {
            bucket: [] for bucket in self.config.splits.keys()
        }
        
        # Load or initialize state
        self._load_state()
        
        # Initialize buckets for current hour if needed
        self._refresh_buckets()
    
    def _get_current_epoch(self) -> int:
        """Get current hour epoch."""
        return int(time.time() // 3600)
    
    def _get_hourly_budget(self) -> float:
        """Get hourly budget allocation."""
        return self.config.daily_budget / 24
    
    def _load_state(self):
        """Load persisted state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                
                self.current_epoch = state.get('epoch', self._get_current_epoch())
                self.buckets = state.get('buckets', {})
                self.rollover = state.get('rollover', {})
                
                # Restore backpay queues
                for bucket, requests in state.get('backpay_queues', {}).items():
                    self.backpay_queues[bucket] = deque([
                        BackpayRequest(**req) for req in requests
                    ])
                
                self.usage_history = state.get('usage_history', {})
                logger.info(f"Loaded bucket state from {self.state_path}")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self._initialize_buckets()
        else:
            self._initialize_buckets()
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            state = {
                'epoch': self.current_epoch,
                'buckets': self.buckets,
                'rollover': self.rollover,
                'backpay_queues': {
                    bucket: [asdict(req) for req in queue]
                    for bucket, queue in self.backpay_queues.items()
                },
                'usage_history': self.usage_history,
                'timestamp': time.time()
            }
            
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _initialize_buckets(self):
        """Initialize buckets with fresh allocations."""
        hourly_budget = self._get_hourly_budget()
        
        for bucket, split in self.config.splits.items():
            self.buckets[bucket] = hourly_budget * split
            self.rollover[bucket] = 0
            
        logger.info(f"Initialized buckets with hourly budget: {hourly_budget} BLY")
    
    def _refresh_buckets(self):
        """Refresh buckets for new hour if needed."""
        current_epoch = self._get_current_epoch()
        
        if current_epoch > self.current_epoch:
            with self.lock:
                # Calculate hours elapsed
                hours_elapsed = current_epoch - self.current_epoch
                
                # Add new hourly allocations
                hourly_budget = self._get_hourly_budget()
                
                for bucket, split in self.config.splits.items():
                    # Calculate new allocation
                    new_allocation = hourly_budget * split * hours_elapsed
                    
                    # Add rollover from previous hour(s)
                    if hours_elapsed <= self.config.rollover_hours:
                        new_allocation += self.buckets.get(bucket, 0)
                    
                    # Apply max accumulation cap
                    max_allowed = hourly_budget * split * self.config.max_accumulation_factor
                    self.buckets[bucket] = min(new_allocation, max_allowed)
                    
                    # Track usage for metrics
                    if bucket not in self.usage_history:
                        self.usage_history[bucket] = []
                    
                    # Keep only last 24 hours of history
                    self.usage_history[bucket] = self.usage_history[bucket][-24:]
                
                self.current_epoch = current_epoch
                
                # Process backpay queue with new budget
                self._process_backpay()
                
                # Save updated state
                self._save_state()
                
                logger.info(f"Refreshed buckets for epoch {current_epoch}")
    
    def allocate(self, bucket_type: str, amount_bly: float, 
                requester: str = "unknown", metadata: Dict = None) -> Tuple[float, str]:
        """
        Allocate BLY from a bucket.
        
        Args:
            bucket_type: Type of bucket (inference, learning, validation, dataset)
            amount_bly: Amount requested
            requester: Identifier of requester
            metadata: Additional context
        
        Returns:
            Tuple of (granted_amount, request_id)
            If budget exhausted, amount is queued and request_id returned for tracking
        """
        self._refresh_buckets()
        
        with self.lock:
            if bucket_type not in self.buckets:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            available = self.buckets[bucket_type]
            
            # Check if we can fulfill immediately
            if available >= amount_bly:
                # Grant full amount
                self.buckets[bucket_type] -= amount_bly
                
                # Track usage
                if bucket_type not in self.usage_history:
                    self.usage_history[bucket_type] = []
                self.usage_history[bucket_type].append(amount_bly)
                
                self._save_state()
                
                logger.debug(f"Allocated {amount_bly} BLY from {bucket_type} bucket")
                return amount_bly, f"granted_{int(time.time()*1000000)}"
            
            # Partial or no allocation - queue for backpay if enabled
            granted = min(available, amount_bly)
            remaining = amount_bly - granted
            
            if granted > 0:
                self.buckets[bucket_type] -= granted
                self.usage_history[bucket_type].append(granted)
            
            request_id = f"backpay_{bucket_type}_{int(time.time()*1000000)}"
            
            if remaining > 0 and self.config.backpay_enabled:
                # Queue the remainder
                backpay_req = BackpayRequest(
                    request_id=request_id,
                    bucket_type=bucket_type,
                    amount_bly=remaining,
                    requester=requester,
                    timestamp=time.time(),
                    metadata=metadata or {}
                )
                
                self.backpay_queues[bucket_type].append(backpay_req)
                logger.info(f"Queued {remaining} BLY for backpay in {bucket_type}")
            
            self._save_state()
            return granted, request_id
    
    def _process_backpay(self):
        """Process backpay queues in priority order."""
        if not self.config.backpay_enabled:
            return
        
        # Process in priority order
        for bucket_type in self.config.backpay_priority:
            if bucket_type not in self.backpay_queues:
                continue
            
            queue = self.backpay_queues[bucket_type]
            available = self.buckets.get(bucket_type, 0)
            
            processed = []
            while queue and available > 0:
                req = queue[0]
                
                # Check if expired
                if req.is_expired(self.config.max_queue_days):
                    queue.popleft()
                    logger.info(f"Expired backpay request: {req.request_id}")
                    continue
                
                # Try to fulfill
                if available >= req.amount_bly:
                    # Full payment
                    self.buckets[bucket_type] -= req.amount_bly
                    available -= req.amount_bly
                    processed.append(queue.popleft())
                    logger.info(f"Fulfilled backpay: {req.request_id} ({req.amount_bly} BLY)")
                else:
                    # Partial payment - update request
                    req.amount_bly -= available
                    req.attempts += 1
                    self.buckets[bucket_type] = 0
                    available = 0
                    logger.info(f"Partial backpay: {req.request_id} ({available} BLY paid)")
                    break
            
            # TODO: Notify recipients of fulfilled backpay
            for req in processed:
                self._notify_backpay_fulfilled(req)
    
    def _notify_backpay_fulfilled(self, request: BackpayRequest):
        """Notify that a backpay request has been fulfilled."""
        # This would integrate with the reward distribution system
        logger.info(f"Backpay fulfilled: {request.request_id} to {request.requester}")
    
    def get_bucket_status(self) -> Dict:
        """Get current status of all buckets."""
        self._refresh_buckets()
        
        with self.lock:
            hourly_budget = self._get_hourly_budget()
            
            status = {
                'epoch': self.current_epoch,
                'hourly_budget': hourly_budget,
                'buckets': {}
            }
            
            for bucket in self.config.splits.keys():
                available = self.buckets.get(bucket, 0)
                target = hourly_budget * self.config.splits[bucket]
                floor = hourly_budget * self.config.floors[bucket]
                ceiling = hourly_budget * self.config.ceilings[bucket]
                
                # Calculate utilization
                recent_usage = sum(self.usage_history.get(bucket, [])[-24:])
                target_24h = target * 24
                utilization = (recent_usage / target_24h * 100) if target_24h > 0 else 0
                
                status['buckets'][bucket] = {
                    'available': round(available, 2),
                    'target': round(target, 2),
                    'floor': round(floor, 2),
                    'ceiling': round(ceiling, 2),
                    'utilization_24h': round(utilization, 1),
                    'backpay_queue_size': len(self.backpay_queues.get(bucket, [])),
                    'backpay_queue_bly': sum(
                        req.amount_bly for req in self.backpay_queues.get(bucket, [])
                    )
                }
            
            return status
    
    def get_metrics(self) -> Dict:
        """Get metrics for monitoring."""
        status = self.get_bucket_status()
        
        metrics = {
            'bucket_utilization': {},
            'backpay_queue_size': {},
            'backpay_queue_bly': {},
            'budget_health': {}
        }
        
        for bucket, info in status['buckets'].items():
            metrics['bucket_utilization'][bucket] = info['utilization_24h']
            metrics['backpay_queue_size'][bucket] = info['backpay_queue_size']
            metrics['backpay_queue_bly'][bucket] = info['backpay_queue_bly']
            
            # Check if bucket is healthy (above floor)
            is_healthy = info['available'] >= info['floor']
            metrics['budget_health'][bucket] = 1 if is_healthy else 0
        
        # Add alerts
        metrics['alerts'] = []
        
        # Check for low utilization
        for bucket, util in metrics['bucket_utilization'].items():
            if bucket == 'learning' and util < 15:
                metrics['alerts'].append({
                    'type': 'low_utilization',
                    'bucket': bucket,
                    'utilization': util,
                    'threshold': 15
                })
        
        # Check for large backpay queue
        total_backpay = sum(metrics['backpay_queue_bly'].values())
        if total_backpay > 100000:
            metrics['alerts'].append({
                'type': 'high_backpay',
                'total_bly': total_backpay,
                'threshold': 100000
            })
        
        return metrics


if __name__ == "__main__":
    # Example usage
    allocator = BucketAllocator()
    
    # Try some allocations
    granted, req_id = allocator.allocate('inference', 100, 'node_1')
    print(f"Inference allocation: {granted} BLY (request: {req_id})")
    
    granted, req_id = allocator.allocate('learning', 500, 'node_2')
    print(f"Learning allocation: {granted} BLY (request: {req_id})")
    
    # Check status
    status = allocator.get_bucket_status()
    print(f"\nBucket Status:")
    print(json.dumps(status, indent=2))
    
    # Get metrics
    metrics = allocator.get_metrics()
    print(f"\nMetrics:")
    print(json.dumps(metrics, indent=2))