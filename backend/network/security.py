"""Security measures for P2P network"""
import time
import hashlib
from typing import Dict, Set, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import ipaddress
import threading


@dataclass
class PeerBan:
    """Ban information for a peer"""
    peer_id: str
    ip_address: str
    reason: str
    timestamp: float
    duration: int  # seconds (0 = permanent)
    
    def is_expired(self) -> bool:
        """Check if ban has expired"""
        if self.duration == 0:
            return False
        return time.time() > self.timestamp + self.duration


class RateLimiter:
    """Token bucket rate limiter per peer"""
    
    def __init__(self, rate: float, burst: int):
        """
        Initialize rate limiter
        
        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.buckets: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def allow(self, peer_id: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed
        
        Args:
            peer_id: Peer identifier
            tokens: Number of tokens required
            
        Returns:
            True if allowed
        """
        with self.lock:
            now = time.time()
            
            # Initialize bucket
            if peer_id not in self.buckets:
                self.buckets[peer_id] = self.burst
                self.last_update[peer_id] = now
            
            # Refill bucket
            elapsed = now - self.last_update[peer_id]
            self.buckets[peer_id] = min(
                self.burst,
                self.buckets[peer_id] + elapsed * self.rate
            )
            self.last_update[peer_id] = now
            
            # Check if enough tokens
            if self.buckets[peer_id] >= tokens:
                self.buckets[peer_id] -= tokens
                return True
            
            return False
    
    def reset(self, peer_id: str):
        """Reset rate limit for peer"""
        with self.lock:
            if peer_id in self.buckets:
                del self.buckets[peer_id]
                del self.last_update[peer_id]


class ReplayProtection:
    """Protect against replay attacks"""
    
    def __init__(self, window_size: int = 10000, ttl: int = 3600):
        """
        Initialize replay protection
        
        Args:
            window_size: Max number of nonces to track
            ttl: Time to live for nonces (seconds)
        """
        self.window_size = window_size
        self.ttl = ttl
        self.nonces: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def check_nonce(self, nonce: str, timestamp: float) -> bool:
        """
        Check if nonce is valid (not replayed)
        
        Args:
            nonce: Message nonce
            timestamp: Message timestamp
            
        Returns:
            True if valid (not seen before)
        """
        with self.lock:
            now = time.time()
            
            # Reject old timestamps
            if abs(now - timestamp) > self.ttl:
                return False
            
            # Check if nonce seen
            if nonce in self.nonces:
                return False
            
            # Clean old nonces
            if len(self.nonces) > self.window_size:
                cutoff = now - self.ttl
                self.nonces = {
                    n: t for n, t in self.nonces.items()
                    if t > cutoff
                }
            
            # Store nonce
            self.nonces[nonce] = timestamp
            return True


class PeerScorer:
    """Score peers based on behavior"""
    
    def __init__(self):
        self.scores: Dict[str, float] = defaultdict(lambda: 100.0)
        self.events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Scoring weights
        self.weights = {
            'valid_block': 10,
            'invalid_block': -50,
            'valid_message': 1,
            'invalid_message': -10,
            'timeout': -5,
            'disconnect': -20,
            'rate_limit': -30,
        }
    
    def record_event(self, peer_id: str, event_type: str):
        """Record peer event"""
        with self.lock:
            weight = self.weights.get(event_type, 0)
            self.scores[peer_id] += weight
            
            # Clamp score
            self.scores[peer_id] = max(0, min(200, self.scores[peer_id]))
            
            # Record event
            self.events[peer_id].append({
                'type': event_type,
                'timestamp': time.time(),
                'weight': weight
            })
    
    def get_score(self, peer_id: str) -> float:
        """Get peer score"""
        return self.scores.get(peer_id, 100.0)
    
    def should_ban(self, peer_id: str) -> bool:
        """Check if peer should be banned"""
        return self.get_score(peer_id) <= 0


class SecurityManager:
    """Manages all security measures"""
    
    def __init__(self, config: dict = None):
        """
        Initialize security manager
        
        Args:
            config: Security configuration
        """
        config = config or {}
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            rate=config.get('rate_limit', 10),  # 10 req/sec
            burst=config.get('burst_limit', 100)
        )
        
        # Replay protection
        self.replay_protection = ReplayProtection()
        
        # Peer scoring
        self.peer_scorer = PeerScorer()
        
        # Ban list
        self.bans: Dict[str, PeerBan] = {}
        self.ip_bans: Set[str] = set()
        
        # DoS protection
        self.connection_limits = {
            'max_peers': config.get('max_peers', 100),
            'max_peers_per_ip': config.get('max_peers_per_ip', 3),
            'max_pending': config.get('max_pending', 50),
        }
        
        self.connections_per_ip: Dict[str, int] = defaultdict(int)
        self.pending_connections: Set[str] = set()
        
        self.lock = threading.Lock()
    
    def check_connection(self, peer_id: str, ip_address: str) -> tuple[bool, str]:
        """
        Check if connection should be allowed
        
        Args:
            peer_id: Peer identifier
            ip_address: Peer IP address
            
        Returns:
            (allowed, reason)
        """
        with self.lock:
            # Check bans
            if self.is_banned(peer_id) or self.is_ip_banned(ip_address):
                return False, "Banned"
            
            # Check IP limits
            if self.connections_per_ip[ip_address] >= self.connection_limits['max_peers_per_ip']:
                return False, "Too many connections from IP"
            
            # Check pending limit
            if len(self.pending_connections) >= self.connection_limits['max_pending']:
                return False, "Too many pending connections"
            
            # Check if IP is valid (not private/loopback unless in dev mode)
            try:
                ip = ipaddress.ip_address(ip_address)
                if ip.is_private or ip.is_loopback:
                    # Allow in development
                    pass
            except ValueError:
                return False, "Invalid IP address"
            
            self.pending_connections.add(peer_id)
            return True, "OK"
    
    def connection_established(self, peer_id: str, ip_address: str):
        """Mark connection as established"""
        with self.lock:
            self.pending_connections.discard(peer_id)
            self.connections_per_ip[ip_address] += 1
    
    def connection_closed(self, peer_id: str, ip_address: str):
        """Mark connection as closed"""
        with self.lock:
            self.pending_connections.discard(peer_id)
            if ip_address in self.connections_per_ip:
                self.connections_per_ip[ip_address] = max(
                    0, self.connections_per_ip[ip_address] - 1
                )
    
    def check_message(self, peer_id: str, message: dict) -> tuple[bool, str]:
        """
        Check if message should be processed
        
        Args:
            peer_id: Peer identifier
            message: Message to check
            
        Returns:
            (allowed, reason)
        """
        # Check rate limit
        if not self.rate_limiter.allow(peer_id):
            self.peer_scorer.record_event(peer_id, 'rate_limit')
            if self.peer_scorer.should_ban(peer_id):
                self.ban_peer(peer_id, "Rate limit violations")
            return False, "Rate limited"
        
        # Check replay
        nonce = message.get('nonce', '')
        timestamp = message.get('timestamp', 0)
        
        if nonce and not self.replay_protection.check_nonce(str(nonce), timestamp):
            self.peer_scorer.record_event(peer_id, 'invalid_message')
            return False, "Replay detected"
        
        # Check timestamp freshness
        if abs(time.time() - timestamp) > 300:  # 5 minutes
            self.peer_scorer.record_event(peer_id, 'invalid_message')
            return False, "Stale timestamp"
        
        return True, "OK"
    
    def ban_peer(self, peer_id: str, reason: str, duration: int = 3600):
        """
        Ban a peer
        
        Args:
            peer_id: Peer to ban
            reason: Ban reason
            duration: Ban duration in seconds (0 = permanent)
        """
        with self.lock:
            self.bans[peer_id] = PeerBan(
                peer_id=peer_id,
                ip_address="",  # Would be filled from connection info
                reason=reason,
                timestamp=time.time(),
                duration=duration
            )
    
    def ban_ip(self, ip_address: str):
        """Ban an IP address"""
        with self.lock:
            self.ip_bans.add(ip_address)
    
    def is_banned(self, peer_id: str) -> bool:
        """Check if peer is banned"""
        with self.lock:
            if peer_id in self.bans:
                ban = self.bans[peer_id]
                if ban.is_expired():
                    del self.bans[peer_id]
                    return False
                return True
            return False
    
    def is_ip_banned(self, ip_address: str) -> bool:
        """Check if IP is banned"""
        return ip_address in self.ip_bans
    
    def cleanup_expired_bans(self):
        """Remove expired bans"""
        with self.lock:
            expired = []
            for peer_id, ban in self.bans.items():
                if ban.is_expired():
                    expired.append(peer_id)
            
            for peer_id in expired:
                del self.bans[peer_id]