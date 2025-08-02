"""Hot Expert Caching and Replication System

This module implements intelligent caching and replication of frequently used expert groups
to minimize network latency and improve distributed inference performance.
"""

from __future__ import annotations

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import aiohttp

from .expert_group_optimizer import ExpertGroup, NodeCapability, ExpertGroupIndex


@dataclass
class CacheEntry:
    """Represents a cached expert group on a node."""
    group_id: str
    experts: Set[str]
    node_id: str
    cache_time: float
    access_count: int = 0
    last_access: float = 0.0
    cache_size_mb: float = 0.0
    
    @property
    def age_hours(self) -> float:
        return (time.time() - self.cache_time) / 3600.0
    
    @property
    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        return self.age_hours > max_age_hours


@dataclass
class ReplicationTask:
    """Task for replicating an expert group to a target node."""
    source_node: str
    target_node: str
    expert_group: ExpertGroup
    priority: float = 0.0
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: float = 0.0
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class HotExpertCache:
    """Manages caching of hot expert groups across nodes."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.cache_entries: Dict[str, List[CacheEntry]] = defaultdict(list)  # group_id -> [cache_entries]
        self.node_cache_usage: Dict[str, float] = {}  # node_id -> cache_usage_mb
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)  # group_id -> [access_times]
        
        # Configuration
        self.max_cache_size_mb = 1024  # 1GB per node
        self.replication_threshold = 0.7  # Replicate when usage > 70%
        self.cache_ttl_hours = 24.0
        
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"
        if not metadata_file.exists():
            return
        
        try:
            with open(metadata_file) as f:
                data = json.load(f)
                
            # Restore cache entries
            for group_id, entries_data in data.get("cache_entries", {}).items():
                self.cache_entries[group_id] = [
                    CacheEntry(
                        group_id=entry["group_id"],
                        experts=set(entry["experts"]),
                        node_id=entry["node_id"],
                        cache_time=entry["cache_time"],
                        access_count=entry["access_count"],
                        last_access=entry["last_access"],
                        cache_size_mb=entry["cache_size_mb"]
                    )
                    for entry in entries_data
                ]
            
            self.node_cache_usage = data.get("node_cache_usage", {})
            self.access_patterns = data.get("access_patterns", {})
            
        except Exception as e:
            print(f"Warning: Could not load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            # Convert cache entries to serializable format
            cache_entries_data = {}
            for group_id, entries in self.cache_entries.items():
                cache_entries_data[group_id] = [
                    {
                        "group_id": entry.group_id,
                        "experts": list(entry.experts),
                        "node_id": entry.node_id,
                        "cache_time": entry.cache_time,
                        "access_count": entry.access_count,
                        "last_access": entry.last_access,
                        "cache_size_mb": entry.cache_size_mb
                    }
                    for entry in entries
                ]
            
            data = {
                "cache_entries": cache_entries_data,
                "node_cache_usage": self.node_cache_usage,
                "access_patterns": self.access_patterns,
                "last_updated": time.time()
            }
            
            metadata_file = self.cache_dir / "cache_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save cache metadata: {e}")
    
    def record_access(self, group_id: str, node_id: str):
        """Record access to an expert group."""
        current_time = time.time()
        
        # Update access patterns
        self.access_patterns[group_id].append(current_time)
        
        # Keep only recent access times (last 24 hours)
        cutoff_time = current_time - (24 * 3600)
        self.access_patterns[group_id] = [
            t for t in self.access_patterns[group_id] if t > cutoff_time
        ]
        
        # Update cache entry if it exists
        for entry in self.cache_entries[group_id]:
            if entry.node_id == node_id:
                entry.access_count += 1
                entry.last_access = current_time
                break
        
        self._save_cache_metadata()
    
    def add_cache_entry(self, group: ExpertGroup, node_id: str, size_mb: float):
        """Add a new cache entry."""
        entry = CacheEntry(
            group_id=group.group_id,
            experts=group.experts,
            node_id=node_id,
            cache_time=time.time(),
            cache_size_mb=size_mb
        )
        
        self.cache_entries[group.group_id].append(entry)
        
        # Update node cache usage
        current_usage = self.node_cache_usage.get(node_id, 0.0)
        self.node_cache_usage[node_id] = current_usage + size_mb
        
        self._save_cache_metadata()
        print(f"✓ Cached expert group {group.group_id} on node {node_id} ({size_mb:.1f}MB)")
    
    def remove_cache_entry(self, group_id: str, node_id: str):
        """Remove a cache entry."""
        entries = self.cache_entries[group_id]
        removed_entry = None
        
        for i, entry in enumerate(entries):
            if entry.node_id == node_id:
                removed_entry = entries.pop(i)
                break
        
        if removed_entry:
            # Update node cache usage
            current_usage = self.node_cache_usage.get(node_id, 0.0)
            self.node_cache_usage[node_id] = max(0.0, current_usage - removed_entry.cache_size_mb)
            
            self._save_cache_metadata()
            print(f"✗ Removed cached expert group {group_id} from node {node_id}")
    
    def get_hot_groups(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get the hottest expert groups based on recent access patterns."""
        current_time = time.time()
        group_heat_scores = {}
        
        for group_id, access_times in self.access_patterns.items():
            if not access_times:
                continue
            
            # Calculate heat score based on recent access frequency
            recent_accesses = len(access_times)
            time_decay = 1.0 / (1.0 + (current_time - max(access_times)) / 3600)  # Decay over hours
            heat_score = recent_accesses * time_decay
            
            group_heat_scores[group_id] = heat_score
        
        # Sort by heat score
        hot_groups = sorted(group_heat_scores.items(), key=lambda x: x[1], reverse=True)
        return hot_groups[:limit]
    
    def suggest_cache_evictions(self, node_id: str) -> List[str]:
        """Suggest cache entries to evict from a node."""
        current_usage = self.node_cache_usage.get(node_id, 0.0)
        
        if current_usage < self.max_cache_size_mb * self.replication_threshold:
            return []
        
        # Find candidates for eviction
        eviction_candidates = []
        for group_id, entries in self.cache_entries.items():
            for entry in entries:
                if entry.node_id == node_id:
                    # Score based on age, access frequency, and size
                    age_penalty = entry.age_hours / 24.0  # 0-1 scale
                    access_bonus = min(entry.access_count / 10.0, 1.0)  # 0-1 scale
                    size_penalty = entry.cache_size_mb / 100.0  # Larger files more likely to be evicted
                    
                    eviction_score = age_penalty + size_penalty - access_bonus
                    eviction_candidates.append((group_id, eviction_score))
        
        # Sort by eviction score (higher = more likely to evict)
        eviction_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return group IDs to evict
        return [group_id for group_id, _ in eviction_candidates[:3]]  # Evict up to 3 groups
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = sum(len(entries) for entries in self.cache_entries.values())
        total_cache_size = sum(self.node_cache_usage.values())
        
        hot_groups = self.get_hot_groups(5)
        
        return {
            "total_cached_groups": len(self.cache_entries),
            "total_cache_entries": total_entries,
            "total_cache_size_mb": total_cache_size,
            "nodes_with_cache": len(self.node_cache_usage),
            "hot_groups": [{"group_id": gid, "heat_score": score} for gid, score in hot_groups],
            "cache_utilization_by_node": dict(self.node_cache_usage)
        }


class ExpertReplicationManager:
    """Manages replication of expert groups across nodes."""
    
    def __init__(self, cache_manager: HotExpertCache, group_index: ExpertGroupIndex):
        self.cache_manager = cache_manager
        self.group_index = group_index
        self.replication_queue: List[ReplicationTask] = []
        self.active_replications: Dict[str, ReplicationTask] = {}  # task_id -> task
        
    def suggest_replications(self) -> List[ReplicationTask]:
        """Suggest expert groups that should be replicated."""
        suggestions = []
        hot_groups = self.cache_manager.get_hot_groups(10)
        
        for group_id, heat_score in hot_groups:
            # Find nodes that have this group
            current_replicas = []
            for entries in self.cache_manager.cache_entries.get(group_id, []):
                current_replicas.append(entries.node_id)
            
            # If group is hot but has few replicas, suggest replication
            if heat_score > 5.0 and len(current_replicas) < 3:
                # Find candidate target nodes
                available_nodes = []
                for node_id, node in self.group_index.nodes.items():
                    if node_id not in current_replicas:
                        cache_usage = self.cache_manager.node_cache_usage.get(node_id, 0.0)
                        if cache_usage < self.cache_manager.max_cache_size_mb * 0.8:  # Not too full
                            available_nodes.append(node_id)
                
                if available_nodes and current_replicas:
                    # Create replication task
                    source_node = current_replicas[0]  # Use first available replica as source
                    target_node = available_nodes[0]  # Use first available target
                    
                    # Mock expert group (in practice, would retrieve from index)
                    mock_group = ExpertGroup(
                        experts=set([f"expert_{i}" for i in range(3)]),  # Mock experts
                        usage_count=int(heat_score)
                    )
                    
                    task = ReplicationTask(
                        source_node=source_node,
                        target_node=target_node,
                        expert_group=mock_group,
                        priority=heat_score
                    )
                    suggestions.append(task)
        
        return suggestions
    
    async def execute_replication(self, task: ReplicationTask) -> bool:
        """Execute a replication task."""
        task_id = f"{task.source_node}->{task.target_node}:{task.expert_group.group_id}"
        self.active_replications[task_id] = task
        task.status = "in_progress"
        
        try:
            # Get source and target node information
            source_node = self.group_index.nodes.get(task.source_node)
            target_node = self.group_index.nodes.get(task.target_node)
            
            if not source_node or not target_node:
                raise Exception(f"Source or target node not found")
            
            # Step 1: Request expert group data from source node
            group_data = await self._fetch_expert_group(source_node, task.expert_group.group_id)
            
            # Step 2: Transfer data to target node
            await self._transfer_expert_group(target_node, task.expert_group.group_id, group_data)
            
            # Step 3: Update cache metadata
            estimated_size = len(str(group_data)) / (1024 * 1024)  # Rough size estimate
            self.cache_manager.add_cache_entry(task.expert_group, task.target_node, estimated_size)
            
            task.status = "completed"
            task.completed_at = time.time()
            
            print(f"✅ Replication completed: {task_id}")
            return True
            
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            print(f"❌ Replication failed: {task_id} - {e}")
            return False
            
        finally:
            if task_id in self.active_replications:
                del self.active_replications[task_id]
    
    async def _fetch_expert_group(self, source_node: NodeCapability, group_id: str) -> Dict:
        """Fetch expert group data from source node."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{source_node.endpoint}/cache/expert_group/{group_id}",
                timeout=aiohttp.ClientTimeout(total=30.0)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to fetch from source: HTTP {response.status}")
    
    async def _transfer_expert_group(self, target_node: NodeCapability, group_id: str, data: Dict):
        """Transfer expert group data to target node."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{target_node.endpoint}/cache/store_expert_group",
                json={"group_id": group_id, "data": data},
                timeout=aiohttp.ClientTimeout(total=60.0)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to store on target: HTTP {response.status}")
    
    def get_replication_stats(self) -> Dict[str, Any]:
        """Get replication statistics."""
        return {
            "queued_replications": len(self.replication_queue),
            "active_replications": len(self.active_replications),
            "active_tasks": [
                {
                    "source": task.source_node,
                    "target": task.target_node,
                    "group_id": task.expert_group.group_id,
                    "status": task.status,
                    "priority": task.priority
                }
                for task in self.active_replications.values()
            ]
        }