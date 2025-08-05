#!/usr/bin/env python3
"""
Genesis Pact Integrity Verification System
Ensures all nodes maintain consensus on the foundational Human-AI covenant.
"""

import hashlib
import requests
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PeerVerificationResult:
    """Result of peer Genesis verification."""
    peer_id: str
    genesis_hash: str
    verification_time: float
    network_id: str
    success: bool
    error: Optional[str] = None


@dataclass
class NetworkPeer:
    """Network peer information."""
    peer_id: str
    host: str
    port: int
    last_seen: float
    trusted: bool = False
    genesis_hash: Optional[str] = None


class GenesisIntegrityGuard:
    """Central Genesis Pact integrity verification system."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("./data/genesis_verification")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load expected Genesis hash
        self.expected_genesis_hash = self._load_expected_genesis_hash()
        if not self.expected_genesis_hash:
            raise RuntimeError("Genesis Pact hash not found. Run: python scripts/create_genesis_pact.py")
        
        # Network peers registry
        self.network_peers: Dict[str, NetworkPeer] = {}
        self.verification_history: List[PeerVerificationResult] = []
        
        # Configuration
        self.consensus_threshold = 0.75  # 75% consensus required
        self.verification_timeout = 10   # 10 seconds per peer
        self.quarantine_threshold = 3    # Failed verifications before quarantine
        
        self._load_peer_registry()
        
        print(f"🏛️ Genesis Integrity Guard initialized")
        print(f"   Expected Genesis Hash: {self.expected_genesis_hash[:16]}...")
        print(f"   Consensus Threshold: {self.consensus_threshold * 100}%")
    
    def _load_expected_genesis_hash(self) -> Optional[str]:
        """Load the expected Genesis hash from file."""
        genesis_files = [
            Path("./data/genesis_pact_hash.txt"),
            self.storage_dir / "genesis_hash.txt"
        ]
        
        for genesis_file in genesis_files:
            if genesis_file.exists():
                try:
                    return genesis_file.read_text().strip()
                except Exception as e:
                    print(f"Warning: Failed to read Genesis hash from {genesis_file}: {e}")
        
        return None
    
    def _load_peer_registry(self):
        """Load peer registry from storage."""
        registry_file = self.storage_dir / "peer_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)
                    for peer_id, peer_data in data.items():
                        self.network_peers[peer_id] = NetworkPeer(**peer_data)
            except Exception as e:
                print(f"Warning: Failed to load peer registry: {e}")
    
    def _save_peer_registry(self):
        """Save peer registry to storage."""
        registry_file = self.storage_dir / "peer_registry.json"
        try:
            data = {
                peer_id: {
                    "peer_id": peer.peer_id,
                    "host": peer.host,
                    "port": peer.port,
                    "last_seen": peer.last_seen,
                    "trusted": peer.trusted,
                    "genesis_hash": peer.genesis_hash
                }
                for peer_id, peer in self.network_peers.items()
            }
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save peer registry: {e}")
    
    def register_peer(self, peer_id: str, host: str, port: int) -> bool:
        """Register a new network peer."""
        peer = NetworkPeer(
            peer_id=peer_id,
            host=host,
            port=port,
            last_seen=time.time()
        )
        
        self.network_peers[peer_id] = peer
        self._save_peer_registry()
        
        print(f"📝 Registered peer {peer_id} at {host}:{port}")
        return True
    
    def verify_peer_genesis(self, peer: NetworkPeer) -> PeerVerificationResult:
        """Verify Genesis hash with a single peer."""
        start_time = time.time()
        
        try:
            # Make request to peer's genesis endpoint
            url = f"http://{peer.host}:{peer.port}/genesis/hash"
            response = requests.get(url, timeout=self.verification_timeout)
            
            if response.status_code == 200:
                data = response.json()
                peer_genesis_hash = data.get("genesis_hash", "")
                network_id = data.get("network", "unknown")
                
                # Update peer info
                peer.genesis_hash = peer_genesis_hash
                peer.last_seen = time.time()
                
                # Verify hash matches
                success = peer_genesis_hash == self.expected_genesis_hash
                
                result = PeerVerificationResult(
                    peer_id=peer.peer_id,
                    genesis_hash=peer_genesis_hash,
                    verification_time=time.time() - start_time,
                    network_id=network_id,
                    success=success
                )
                
                if not success:
                    result.error = f"Genesis hash mismatch: expected {self.expected_genesis_hash[:16]}..., got {peer_genesis_hash[:16]}..."
                    print(f"⚠️ Peer {peer.peer_id} has incorrect Genesis hash!")
                
                return result
            
            else:
                return PeerVerificationResult(
                    peer_id=peer.peer_id,
                    genesis_hash="",
                    verification_time=time.time() - start_time,
                    network_id="unknown",
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        
        except Exception as e:
            return PeerVerificationResult(
                peer_id=peer.peer_id,
                genesis_hash="",
                verification_time=time.time() - start_time,
                network_id="unknown",
                success=False,
                error=str(e)
            )
    
    async def verify_all_peers_async(self) -> List[PeerVerificationResult]:
        """Verify Genesis hash with all registered peers asynchronously."""
        if not self.network_peers:
            print("⚠️ No peers registered for Genesis verification")
            return []
        
        print(f"🔍 Verifying Genesis consensus with {len(self.network_peers)} peers...")
        
        # Use ThreadPoolExecutor for blocking network calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for peer in self.network_peers.values():
                task = loop.run_in_executor(executor, self.verify_peer_genesis, peer)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to results
        verification_results = []
        for result in results:
            if isinstance(result, PeerVerificationResult):
                verification_results.append(result)
                self.verification_history.append(result)
            else:
                print(f"⚠️ Verification task failed: {result}")
        
        # Save updated peer registry
        self._save_peer_registry()
        
        return verification_results
    
    def verify_network_consensus(self) -> Dict:
        """Verify Genesis consensus across the entire network."""
        # Run async verification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self.verify_all_peers_async())
        finally:
            loop.close()
        
        if not results:
            return {
                "consensus_achieved": False,
                "error": "No peers available for verification",
                "total_peers": 0,
                "successful_verifications": 0,
                "failed_verifications": 0
            }
        
        # Analyze results
        successful_verifications = [r for r in results if r.success]
        failed_verifications = [r for r in results if not r.success]
        
        consensus_percentage = len(successful_verifications) / len(results)
        consensus_achieved = consensus_percentage >= self.consensus_threshold
        
        # Quarantine peers with wrong Genesis hash
        for failed_result in failed_verifications:
            if failed_result.genesis_hash and failed_result.genesis_hash != self.expected_genesis_hash:
                self._quarantine_malicious_peer(failed_result)
        
        verification_summary = {
            "consensus_achieved": consensus_achieved,
            "consensus_percentage": consensus_percentage,
            "threshold_required": self.consensus_threshold,
            "total_peers": len(results),
            "successful_verifications": len(successful_verifications),
            "failed_verifications": len(failed_verifications),
            "expected_genesis_hash": self.expected_genesis_hash,
            "verification_timestamp": time.time(),
            "peer_results": [
                {
                    "peer_id": r.peer_id,
                    "success": r.success,
                    "genesis_hash": r.genesis_hash[:16] + "..." if r.genesis_hash else "",
                    "error": r.error,
                    "verification_time": r.verification_time
                }
                for r in results
            ]
        }
        
        if consensus_achieved:
            print(f"✅ Genesis consensus achieved: {consensus_percentage:.1%} of peers verified")
        else:
            print(f"❌ Genesis consensus FAILED: Only {consensus_percentage:.1%} of peers verified (need {self.consensus_threshold:.1%})")
            
            # Record security event
            from .monitoring import record_security_event
            record_security_event(
                "genesis_consensus_failure", 
                "genesis_integrity_guard",
                {
                    "consensus_percentage": consensus_percentage,
                    "failed_peers": len(failed_verifications),
                    "total_peers": len(results)
                }
            )
        
        return verification_summary
    
    def _quarantine_malicious_peer(self, failed_result: PeerVerificationResult):
        """Quarantine a peer with incorrect Genesis hash."""
        peer = self.network_peers.get(failed_result.peer_id)
        if peer:
            peer.trusted = False
            print(f"🚫 Quarantined malicious peer {failed_result.peer_id}: incorrect Genesis hash")
            
            # Record security event
            from .monitoring import record_security_event
            record_security_event(
                "malicious_peer_quarantined",
                "genesis_integrity_guard",
                {
                    "peer_id": failed_result.peer_id,
                    "expected_hash": self.expected_genesis_hash[:16] + "...",
                    "received_hash": failed_result.genesis_hash[:16] + "..." if failed_result.genesis_hash else "none",
                    "peer_host": peer.host,
                    "peer_port": peer.port
                }
            )
    
    def should_accept_peer_connection(self, peer_id: str, peer_genesis_hash: str) -> bool:
        """Check if we should accept a connection from a peer."""
        if peer_genesis_hash != self.expected_genesis_hash:
            print(f"🚫 Rejecting connection from peer {peer_id}: Genesis hash mismatch")
            
            # Record security event
            from .monitoring import record_security_event
            record_security_event(
                "malicious_peer_connection_rejected",
                "genesis_integrity_guard",
                {
                    "peer_id": peer_id,
                    "expected_hash": self.expected_genesis_hash[:16] + "...",
                    "received_hash": peer_genesis_hash[:16] + "..." if peer_genesis_hash else "none"
                }
            )
            
            return False
        
        print(f"✅ Accepting connection from peer {peer_id}: Genesis hash verified")
        return True
    
    def get_network_status(self) -> Dict:
        """Get current network Genesis verification status."""
        trusted_peers = [p for p in self.network_peers.values() if p.trusted]
        quarantined_peers = [p for p in self.network_peers.values() if not p.trusted]
        
        recent_verifications = [v for v in self.verification_history if time.time() - v.verification_time < 3600]  # Last hour
        
        return {
            "genesis_hash": self.expected_genesis_hash,
            "total_peers": len(self.network_peers),
            "trusted_peers": len(trusted_peers),
            "quarantined_peers": len(quarantined_peers),
            "consensus_threshold": self.consensus_threshold,
            "recent_verifications": len(recent_verifications),
            "successful_recent_verifications": len([v for v in recent_verifications if v.success]),
            "last_verification": max([v.verification_time for v in self.verification_history]) if self.verification_history else None,
            "peer_list": [
                {
                    "peer_id": peer.peer_id,
                    "host": peer.host,
                    "port": peer.port,
                    "trusted": peer.trusted,
                    "last_seen": peer.last_seen,
                    "genesis_hash_match": peer.genesis_hash == self.expected_genesis_hash if peer.genesis_hash else None
                }
                for peer in self.network_peers.values()
            ]
        }


# Global instance
genesis_guard = GenesisIntegrityGuard()


# Convenience functions
def register_network_peer(peer_id: str, host: str, port: int) -> bool:
    """Register a network peer for Genesis verification."""
    return genesis_guard.register_peer(peer_id, host, port)


def verify_network_genesis_consensus() -> Dict:
    """Verify Genesis consensus across all registered peers."""
    return genesis_guard.verify_network_consensus()


def should_accept_peer(peer_id: str, peer_genesis_hash: str) -> bool:
    """Check if peer connection should be accepted based on Genesis hash."""
    return genesis_guard.should_accept_peer_connection(peer_id, peer_genesis_hash)


def get_genesis_network_status() -> Dict:
    """Get current Genesis verification network status."""
    return genesis_guard.get_network_status()