"""
Conversation Context Management for Distributed LLM
Supports multiple strategies: full context, KV cache, and hybrid approaches
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import pickle
import torch
from pathlib import Path

class ContextStrategy(Enum):
    FULL_CONTEXT = "full_context"  # Always send full conversation
    KV_CACHE = "kv_cache"          # Use attention key-value cache
    HYBRID = "hybrid"              # Recent + summary + KV cache
    SLIDING_WINDOW = "sliding"     # Keep only recent N turns

@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    token_count: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class KVCacheEntry:
    """Represents cached attention states"""
    cache_id: str
    turn_count: int
    key_states: torch.Tensor = None  # Cached key states
    value_states: torch.Tensor = None  # Cached value states
    position_offset: int = 0
    expires_at: float = 0
    
    def to_bytes(self) -> bytes:
        """Serialize for Redis storage"""
        data = {
            'cache_id': self.cache_id,
            'turn_count': self.turn_count,
            'position_offset': self.position_offset,
            'expires_at': self.expires_at,
            'key_states': self.key_states.cpu() if self.key_states is not None else None,
            'value_states': self.value_states.cpu() if self.value_states is not None else None
        }
        return pickle.dumps(data)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'KVCacheEntry':
        """Deserialize from Redis"""
        obj_data = pickle.loads(data)
        return cls(
            cache_id=obj_data['cache_id'],
            turn_count=obj_data['turn_count'],
            position_offset=obj_data['position_offset'],
            expires_at=obj_data['expires_at'],
            key_states=obj_data['key_states'],
            value_states=obj_data['value_states']
        )

class ConversationManager:
    def __init__(
        self,
        strategy: ContextStrategy = ContextStrategy.HYBRID,
        max_context_length: int = 4096,
        sliding_window_size: int = 10,
        kv_cache_ttl: int = 3600,  # 1 hour
        redis_client: Optional[redis.Redis] = None
    ):
        self.strategy = strategy
        self.max_context_length = max_context_length
        self.sliding_window_size = sliding_window_size
        self.kv_cache_ttl = kv_cache_ttl
        
        # Redis for distributed KV cache storage
        self.redis = redis_client or redis.Redis(
            host='localhost', port=6379, decode_responses=False
        )
        
        # In-memory conversation storage
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        
        # KV cache management
        self.kv_caches: Dict[str, KVCacheEntry] = {}
    
    def start_conversation(self, conversation_id: str) -> str:
        """Initialize a new conversation"""
        self.conversations[conversation_id] = []
        return conversation_id
    
    def add_turn(self, conversation_id: str, role: str, content: str, 
                 metadata: Dict[str, Any] = None) -> ConversationTurn:
        """Add a new turn to the conversation"""
        if conversation_id not in self.conversations:
            self.start_conversation(conversation_id)
        
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            token_count=len(content.split()),  # Simple approximation
            metadata=metadata or {}
        )
        
        self.conversations[conversation_id].append(turn)
        return turn
    
    def get_context_for_inference(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get the appropriate context based on the chosen strategy
        Returns: {
            'messages': List[Dict],  # Chat format messages
            'kv_cache': Optional[KVCacheEntry],  # Cached attention states
            'strategy_used': str,
            'token_estimate': int
        }
        """
        if conversation_id not in self.conversations:
            return {
                'messages': [],
                'kv_cache': None,
                'strategy_used': self.strategy.value,
                'token_estimate': 0
            }
        
        conversation = self.conversations[conversation_id]
        
        if self.strategy == ContextStrategy.FULL_CONTEXT:
            return self._get_full_context(conversation)
        elif self.strategy == ContextStrategy.KV_CACHE:
            return self._get_kv_cached_context(conversation_id, conversation)
        elif self.strategy == ContextStrategy.HYBRID:
            return self._get_hybrid_context(conversation_id, conversation)
        elif self.strategy == ContextStrategy.SLIDING_WINDOW:
            return self._get_sliding_window_context(conversation)
        
        # Fallback to full context
        return self._get_full_context(conversation)
    
    def _get_full_context(self, conversation: List[ConversationTurn]) -> Dict[str, Any]:
        """Always send the complete conversation history"""
        messages = []
        total_tokens = 0
        
        for turn in conversation:
            messages.append({
                'role': turn.role,
                'content': turn.content
            })
            total_tokens += turn.token_count
        
        return {
            'messages': messages,
            'kv_cache': None,
            'strategy_used': ContextStrategy.FULL_CONTEXT.value,
            'token_estimate': total_tokens
        }
    
    def _get_sliding_window_context(self, conversation: List[ConversationTurn]) -> Dict[str, Any]:
        """Keep only the most recent N turns"""
        recent_turns = conversation[-self.sliding_window_size:] if len(conversation) > self.sliding_window_size else conversation
        
        messages = []
        total_tokens = 0
        
        for turn in recent_turns:
            messages.append({
                'role': turn.role,
                'content': turn.content
            })
            total_tokens += turn.token_count
        
        return {
            'messages': messages,
            'kv_cache': None,
            'strategy_used': ContextStrategy.SLIDING_WINDOW.value,
            'token_estimate': total_tokens
        }
    
    def _get_kv_cached_context(self, conversation_id: str, conversation: List[ConversationTurn]) -> Dict[str, Any]:
        """Use KV cache for efficiency - only send new turns"""
        cache_key = f"kv_cache:{conversation_id}"
        
        try:
            # Try to load existing KV cache
            cached_data = self.redis.get(cache_key)
            if cached_data:
                kv_cache = KVCacheEntry.from_bytes(cached_data)
                
                # Check if cache is still valid and up-to-date
                if (kv_cache.expires_at > time.time() and 
                    kv_cache.turn_count < len(conversation)):
                    
                    # Send only new turns since cache
                    new_turns = conversation[kv_cache.turn_count:]
                    messages = [{'role': turn.role, 'content': turn.content} for turn in new_turns]
                    
                    return {
                        'messages': messages,
                        'kv_cache': kv_cache,
                        'strategy_used': ContextStrategy.KV_CACHE.value,
                        'token_estimate': sum(turn.token_count for turn in new_turns)
                    }
        
        except Exception as e:
            print(f"KV cache retrieval failed: {e}")
        
        # Fallback to full context if cache miss/invalid
        return self._get_full_context(conversation)
    
    def _get_hybrid_context(self, conversation_id: str, conversation: List[ConversationTurn]) -> Dict[str, Any]:
        """
        Hybrid strategy: 
        - Use KV cache for older context
        - Send recent turns in full
        - Summarize very old context
        """
        if len(conversation) <= self.sliding_window_size:
            return self._get_full_context(conversation)
        
        # Try KV cache first
        kv_result = self._get_kv_cached_context(conversation_id, conversation)
        
        # If KV cache worked, enhance with recent context
        if kv_result['kv_cache'] is not None:
            # Add recent turns for better context
            recent_turns = conversation[-3:]  # Always include last 3 turns
            enhanced_messages = [{'role': turn.role, 'content': turn.content} for turn in recent_turns]
            
            return {
                'messages': enhanced_messages,
                'kv_cache': kv_result['kv_cache'],
                'strategy_used': ContextStrategy.HYBRID.value,
                'token_estimate': kv_result['token_estimate'] + sum(turn.token_count for turn in recent_turns)
            }
        
        # Fallback: sliding window + summary
        return self._get_sliding_window_context(conversation)
    
    def store_kv_cache(self, conversation_id: str, key_states: torch.Tensor, 
                      value_states: torch.Tensor, turn_count: int) -> None:
        """Store KV cache after model inference"""
        cache_id = f"kv_cache:{conversation_id}"
        
        kv_cache = KVCacheEntry(
            cache_id=cache_id,
            turn_count=turn_count,
            key_states=key_states,
            value_states=value_states,
            expires_at=time.time() + self.kv_cache_ttl
        )
        
        try:
            self.redis.setex(
                cache_id,
                self.kv_cache_ttl,
                kv_cache.to_bytes()
            )
            self.kv_caches[conversation_id] = kv_cache
        except Exception as e:
            print(f"Failed to store KV cache: {e}")
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics about a conversation"""
        if conversation_id not in self.conversations:
            return {}
        
        conversation = self.conversations[conversation_id]
        total_tokens = sum(turn.token_count for turn in conversation)
        
        return {
            'turn_count': len(conversation),
            'total_tokens': total_tokens,
            'strategy': self.strategy.value,
            'has_kv_cache': conversation_id in self.kv_caches,
            'conversation_age': time.time() - conversation[0].timestamp if conversation else 0
        }
    
    def cleanup_expired_conversations(self, max_age_hours: int = 24) -> int:
        """Remove old conversations and caches"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned = 0
        
        # Clean up conversations
        to_remove = []
        for conv_id, turns in self.conversations.items():
            if turns and turns[0].timestamp < cutoff_time:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.conversations[conv_id]
            if conv_id in self.kv_caches:
                del self.kv_caches[conv_id]
            # Redis keys expire automatically
            cleaned += 1
        
        return cleaned

# Global conversation manager instance
conversation_manager = ConversationManager(
    strategy=ContextStrategy.HYBRID,  # Start with hybrid approach
    max_context_length=4096,
    sliding_window_size=10,
    kv_cache_ttl=3600
)