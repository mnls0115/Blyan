"""
Queue Manager for Fair Admission Control

Purpose: Manage chat request queue with FIFO ordering, per-user limits,
and live position/ETA updates. Provides bounded capacity and fairness.
"""

import asyncio
import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Deque
import logging

logger = logging.getLogger(__name__)

# Queue states
QueueState = Literal["queued", "assigned", "started", "canceled", "expired", "done", "failed"]


@dataclass
class Ticket:
    """Admission ticket for queued request."""
    ticket_id: str
    user_key: str  # IP or auth user ID
    created_at: float
    state: QueueState
    position: int = 0  # Computed dynamically
    eta_seconds: int = 0  # Computed based on average job time
    prompt_meta: Dict = field(default_factory=dict)  # Minimal metadata
    deadline_at: float = 0  # TTL deadline
    assigned_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class QueueFull(Exception):
    """Queue has reached maximum capacity."""
    pass


class LimitExceeded(Exception):
    """User has exceeded concurrent request limit."""
    pass


class NotFound(Exception):
    """Ticket not found."""
    pass


class QueueManager:
    """Manages admission queue with fairness and capacity control."""
    
    def __init__(self, 
                 max_queue_length: int = 200,
                 max_user_concurrency: int = 1,
                 admission_timeout_s: int = 600,
                 job_timeout_s: int = 600):
        """
        Initialize queue manager.
        
        Args:
            max_queue_length: Maximum queue size
            max_user_concurrency: Max concurrent requests per user
            admission_timeout_s: TTL for queued tickets
            job_timeout_s: Maximum job duration
        """
        self.max_queue_length = max_queue_length
        self.max_user_concurrency = max_user_concurrency
        self.admission_timeout_s = admission_timeout_s
        self.job_timeout_s = job_timeout_s
        
        # Core data structures
        self._lock = asyncio.Lock()
        self._queue: Deque[str] = deque()  # FIFO queue of ticket IDs
        self._tickets: Dict[str, Ticket] = {}  # ticket_id -> Ticket
        self._user_active: Dict[str, List[str]] = defaultdict(list)  # user -> active ticket IDs
        
        # Metrics
        self._metrics = {
            'total_enqueued': 0,
            'total_admitted': 0,
            'total_completed': 0,
            'total_canceled': 0,
            'total_expired': 0,
            'total_failed': 0,
            'total_dropped': 0,  # Queue full or rate limited
            'total_wait_time': 0.0,
            'job_durations': deque(maxlen=100),  # Last 100 job durations for ETA
            'admission_times': deque(maxlen=100),  # Track admission timestamps for rate calc
            'drop_times': deque(maxlen=100)  # Track drop timestamps
        }
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def enqueue(self, user_key: str, prompt_meta: Optional[Dict] = None) -> Ticket:
        """
        Enqueue a new request.
        
        Args:
            user_key: User identifier (IP or auth ID)
            prompt_meta: Optional metadata for scheduling
            
        Returns:
            Ticket object
            
        Raises:
            QueueFull: Queue at capacity
            LimitExceeded: User at concurrent limit
        """
        async with self._lock:
            # Check queue capacity
            if len(self._queue) >= self.max_queue_length:
                self._metrics['total_dropped'] += 1
                self._metrics['drop_times'].append(time.time())
                raise QueueFull(f"Queue full ({self.max_queue_length} tickets)")
            
            # Count user's active tickets (queued, assigned, started)
            active_states = {"queued", "assigned", "started"}
            user_active_count = sum(
                1 for tid in self._user_active.get(user_key, [])
                if tid in self._tickets and self._tickets[tid].state in active_states
            )
            
            if user_active_count >= self.max_user_concurrency:
                self._metrics['total_dropped'] += 1
                self._metrics['drop_times'].append(time.time())
                raise LimitExceeded(f"User {user_key} has {user_active_count} active requests (max: {self.max_user_concurrency})")
            
            # Create ticket
            ticket_id = uuid.uuid4().hex
            now = time.time()
            ticket = Ticket(
                ticket_id=ticket_id,
                user_key=user_key,
                created_at=now,
                state="queued",
                prompt_meta=prompt_meta or {},
                deadline_at=now + self.admission_timeout_s
            )
            
            # Add to structures
            self._queue.append(ticket_id)
            self._tickets[ticket_id] = ticket
            self._user_active[user_key].append(ticket_id)
            
            # Update position and ETA
            self._update_positions()
            ticket.position = self._compute_position(ticket_id)
            ticket.eta_seconds = self._compute_eta(ticket.position)
            
            # Update metrics
            self._metrics['total_enqueued'] += 1
            
            logger.info(f"Enqueued ticket {ticket_id} for {user_key}, position: {ticket.position}")
            return ticket
    
    async def cancel(self, ticket_id: str, user_key: str) -> bool:
        """
        Cancel a ticket.
        
        Args:
            ticket_id: Ticket to cancel
            user_key: User requesting cancellation
            
        Returns:
            True if canceled, False if not found or not owned by user
        """
        async with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.user_key != user_key:
                return False
            
            if ticket.state in {"canceled", "done", "failed", "expired"}:
                return False  # Already terminated
            
            # Mark as canceled
            ticket.state = "canceled"
            ticket.completed_at = time.time()
            
            # Remove from queue if queued
            if ticket_id in self._queue:
                self._queue.remove(ticket_id)
                self._update_positions()
            
            # Update metrics
            self._metrics['total_canceled'] += 1
            
            logger.info(f"Canceled ticket {ticket_id}")
            return True
    
    async def get(self, ticket_id: str) -> Optional[Ticket]:
        """Get ticket by ID."""
        async with self._lock:
            return self._tickets.get(ticket_id)
    
    async def position(self, ticket_id: str) -> int:
        """Get current queue position (0 if not queued)."""
        async with self._lock:
            return self._compute_position(ticket_id)
    
    async def eta(self, ticket_id: str) -> int:
        """Get estimated time to admission in seconds."""
        async with self._lock:
            pos = self._compute_position(ticket_id)
            return self._compute_eta(pos)
    
    async def admit_next(self, capacity: int = 1) -> List[Ticket]:
        """
        Admit next tickets up to capacity, respecting per-user limits.
        
        Args:
            capacity: Number of slots available
            
        Returns:
            List of admitted tickets
        """
        async with self._lock:
            admitted = []
            
            # Track users who are at their limit
            users_at_limit = set()
            for user_key, ticket_ids in self._user_active.items():
                active_count = sum(
                    1 for tid in ticket_ids
                    if tid in self._tickets and self._tickets[tid].state in {"assigned", "started"}
                )
                if active_count >= self.max_user_concurrency:
                    users_at_limit.add(user_key)
            
            # Admit from queue
            to_remove = []
            for ticket_id in list(self._queue):
                if len(admitted) >= capacity:
                    break
                
                ticket = self._tickets.get(ticket_id)
                if not ticket:
                    to_remove.append(ticket_id)
                    continue
                
                # Skip if user at limit
                if ticket.user_key in users_at_limit:
                    continue
                
                # Skip if expired
                if time.time() > ticket.deadline_at:
                    ticket.state = "expired"
                    to_remove.append(ticket_id)
                    self._metrics['total_expired'] += 1
                    continue
                
                # Admit this ticket
                ticket.state = "assigned"
                ticket.assigned_at = time.time()
                admitted.append(ticket)
                to_remove.append(ticket_id)
                users_at_limit.add(ticket.user_key)  # Update limit tracking
                
                # Update metrics
                self._metrics['total_admitted'] += 1
                self._metrics['admission_times'].append(ticket.assigned_at)
                wait_time = ticket.assigned_at - ticket.created_at
                self._metrics['total_wait_time'] += wait_time
                
                logger.info(f"Admitted ticket {ticket_id}, waited {wait_time:.1f}s")
            
            # Remove admitted/expired tickets from queue
            for ticket_id in to_remove:
                if ticket_id in self._queue:
                    self._queue.remove(ticket_id)
            
            # Update positions for remaining queued tickets
            if admitted or to_remove:
                self._update_positions()
            
            return admitted
    
    async def start(self, ticket_id: str) -> bool:
        """
        Mark ticket as started.
        
        Args:
            ticket_id: Ticket to start
            
        Returns:
            True if started, False if not found or wrong state
        """
        async with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.state != "assigned":
                return False
            
            ticket.state = "started"
            ticket.started_at = time.time()
            
            logger.info(f"Started ticket {ticket_id}")
            return True
    
    async def complete(self, ticket_id: str, success: bool) -> None:
        """
        Mark ticket as completed.
        
        Args:
            ticket_id: Ticket to complete
            success: Whether job succeeded
        """
        async with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            
            ticket.completed_at = time.time()
            
            if success:
                ticket.state = "done"
                self._metrics['total_completed'] += 1
                
                # Track job duration for ETA calculation
                if ticket.started_at:
                    duration = ticket.completed_at - ticket.started_at
                    self._metrics['job_durations'].append(duration)
                    logger.info(f"Completed ticket {ticket_id}, duration: {duration:.1f}s")
            else:
                ticket.state = "failed"
                self._metrics['total_failed'] += 1
                logger.warning(f"Failed ticket {ticket_id}")
    
    async def metrics(self) -> Dict:
        """Get queue metrics."""
        async with self._lock:
            active_states = {"queued", "assigned", "started"}
            queue_length = len(self._queue)
            active_count = sum(1 for t in self._tickets.values() if t.state in active_states)
            
            # Calculate average wait time
            admitted = self._metrics['total_admitted']
            avg_wait = self._metrics['total_wait_time'] / admitted if admitted > 0 else 0
            
            # Calculate average job duration
            durations = self._metrics['job_durations']
            avg_duration = sum(durations) / len(durations) if durations else 60  # Default 60s
            
            # Calculate admission rate (per second) over last 60 seconds
            now = time.time()
            recent_admissions = [t for t in self._metrics['admission_times'] if now - t < 60]
            admission_rate = len(recent_admissions) / 60.0 if recent_admissions else 0
            
            # Calculate drop rate over last 60 seconds
            recent_drops = [t for t in self._metrics['drop_times'] if now - t < 60]
            drop_rate = len(recent_drops) / 60.0 if recent_drops else 0
            
            return {
                'queue_length': queue_length,
                'active_jobs': active_count,
                'total_enqueued': self._metrics['total_enqueued'],
                'total_admitted': self._metrics['total_admitted'],
                'total_completed': self._metrics['total_completed'],
                'total_canceled': self._metrics['total_canceled'],
                'total_expired': self._metrics['total_expired'],
                'total_failed': self._metrics['total_failed'],
                'total_dropped': self._metrics['total_dropped'],
                'avg_wait_seconds': round(avg_wait, 1),
                'avg_job_seconds': round(avg_duration, 1),
                'admission_rate_per_sec': round(admission_rate, 3),
                'drop_rate_per_sec': round(drop_rate, 3),
                'users_active': len(self._user_active)
            }
    
    def _compute_position(self, ticket_id: str) -> int:
        """Compute queue position (1-based, 0 if not queued)."""
        try:
            return list(self._queue).index(ticket_id) + 1
        except ValueError:
            return 0
    
    def _compute_eta(self, position: int) -> int:
        """Estimate seconds until admission based on position."""
        if position <= 0:
            return 0
        
        # Use average job duration from recent completions
        durations = self._metrics['job_durations']
        avg_duration = sum(durations) / len(durations) if durations else 60  # Default 60s
        
        # Simple ETA: position * average duration
        # Could be refined with capacity and concurrency info
        return int(position * avg_duration)
    
    def _update_positions(self) -> None:
        """Update positions for all queued tickets."""
        for i, ticket_id in enumerate(self._queue):
            if ticket_id in self._tickets:
                self._tickets[ticket_id].position = i + 1
    
    async def _cleanup_expired(self) -> None:
        """Background task to clean up expired tickets."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                async with self._lock:
                    now = time.time()
                    expired = []
                    
                    for ticket_id, ticket in list(self._tickets.items()):
                        # Remove old completed/failed/canceled tickets after 5 minutes
                        if ticket.state in {"done", "failed", "canceled", "expired"}:
                            if ticket.completed_at and now - ticket.completed_at > 300:
                                expired.append(ticket_id)
                        
                        # Expire queued tickets past deadline
                        elif ticket.state == "queued" and now > ticket.deadline_at:
                            ticket.state = "expired"
                            ticket.completed_at = now
                            if ticket_id in self._queue:
                                self._queue.remove(ticket_id)
                            self._metrics['total_expired'] += 1
                            logger.info(f"Expired ticket {ticket_id}")
                        
                        # Timeout started jobs
                        elif ticket.state == "started" and ticket.started_at:
                            if now - ticket.started_at > self.job_timeout_s:
                                ticket.state = "failed"
                                ticket.completed_at = now
                                self._metrics['total_failed'] += 1
                                logger.warning(f"Timed out ticket {ticket_id}")
                    
                    # Clean up old tickets
                    for ticket_id in expired:
                        ticket = self._tickets.pop(ticket_id, None)
                        if ticket:
                            # Remove from user active list
                            if ticket.user_key in self._user_active:
                                self._user_active[ticket.user_key] = [
                                    tid for tid in self._user_active[ticket.user_key]
                                    if tid != ticket_id
                                ]
                                if not self._user_active[ticket.user_key]:
                                    del self._user_active[ticket.user_key]
                    
                    if expired:
                        logger.debug(f"Cleaned up {len(expired)} old tickets")
                        
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        
        # Cancel all queued tickets
        async with self._lock:
            for ticket_id in list(self._queue):
                ticket = self._tickets.get(ticket_id)
                if ticket:
                    ticket.state = "canceled"
                    ticket.completed_at = time.time()
            self._queue.clear()
            
        logger.info("Queue manager shut down")