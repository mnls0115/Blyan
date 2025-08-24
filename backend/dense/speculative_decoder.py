"""
Speculative Decoding with Model Size Awareness
===============================================
Accelerates inference using draft models matched to main model size.
Supports heterogeneous GPU configurations and dynamic model profiles.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import asyncio
import numpy as np
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class DraftModelProfile:
    """Profile for a draft model used in speculative decoding."""
    model_name: str
    model_size: str  # "tiny", "small", "medium"
    num_params: float  # In billions
    num_layers: int
    hidden_size: int
    
    # Performance characteristics
    tokens_per_second: float  # Expected throughput
    memory_requirement_gb: float
    
    # Compatibility
    compatible_with: List[str]  # List of main model names
    vocab_size: int
    max_sequence_length: int
    
    # Quality metrics
    acceptance_rate: float = 0.0  # Historical acceptance rate
    avg_draft_length: float = 4.0  # Average accepted draft tokens


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    # Draft parameters
    draft_length: int = 4  # Number of tokens to draft
    temperature: float = 0.8  # Draft temperature (slightly higher for diversity)
    top_k: int = 10  # Draft top-k sampling
    
    # Verification parameters
    verify_temperature: float = 0.6  # Main model temperature
    verify_top_p: float = 0.95
    
    # Adaptive parameters
    min_acceptance_rate: float = 0.5  # Minimum acceptance to continue
    adaptive_draft_length: bool = True  # Adjust draft length based on acceptance
    
    # Performance
    max_batch_size: int = 8
    cache_draft_kvs: bool = True  # Cache draft model KV states
    
    # Model size thresholds (in billions of parameters)
    tiny_model_threshold: float = 0.5  # < 0.5B
    small_model_threshold: float = 2.0  # 0.5B - 2B
    medium_model_threshold: float = 7.0  # 2B - 7B


class DraftModelSelector:
    """Selects appropriate draft model based on main model size."""
    
    # Predefined draft model profiles for different sizes
    DRAFT_PROFILES = {
        "tiny": [
            DraftModelProfile(
                model_name="gpt2",
                model_size="tiny",
                num_params=0.124,
                num_layers=12,
                hidden_size=768,
                tokens_per_second=1000,
                memory_requirement_gb=0.5,
                compatible_with=["*"],  # Universal
                vocab_size=50257,
                max_sequence_length=1024
            ),
            DraftModelProfile(
                model_name="pythia-70m",
                model_size="tiny",
                num_params=0.07,
                num_layers=6,
                hidden_size=512,
                tokens_per_second=1500,
                memory_requirement_gb=0.3,
                compatible_with=["*"],
                vocab_size=50304,
                max_sequence_length=2048
            )
        ],
        "small": [
            DraftModelProfile(
                model_name="pythia-410m",
                model_size="small",
                num_params=0.41,
                num_layers=24,
                hidden_size=1024,
                tokens_per_second=500,
                memory_requirement_gb=1.5,
                compatible_with=["*"],
                vocab_size=50304,
                max_sequence_length=2048
            ),
            DraftModelProfile(
                model_name="gpt2-medium",
                model_size="small",
                num_params=0.355,
                num_layers=24,
                hidden_size=1024,
                tokens_per_second=600,
                memory_requirement_gb=1.3,
                compatible_with=["*"],
                vocab_size=50257,
                max_sequence_length=1024
            )
        ],
        "medium": [
            DraftModelProfile(
                model_name="pythia-1.4b",
                model_size="medium",
                num_params=1.4,
                num_layers=24,
                hidden_size=2048,
                tokens_per_second=200,
                memory_requirement_gb=4.0,
                compatible_with=["Qwen", "Llama", "Mistral"],
                vocab_size=50304,
                max_sequence_length=2048
            ),
            DraftModelProfile(
                model_name="qwen-1.8b",
                model_size="medium",
                num_params=1.8,
                num_layers=24,
                hidden_size=2048,
                tokens_per_second=180,
                memory_requirement_gb=4.5,
                compatible_with=["Qwen"],
                vocab_size=152064,
                max_sequence_length=8192
            )
        ]
    }
    
    @classmethod
    def select_draft_model(
        cls,
        main_model_name: str,
        main_model_params: float,  # In billions
        available_memory_gb: float,
        config: SpeculativeConfig
    ) -> Optional[DraftModelProfile]:
        """Select best draft model for the main model."""
        # Determine size category
        if main_model_params < 7:
            # Small main model - use tiny draft
            size_category = "tiny"
            max_draft_params = 0.2
        elif main_model_params < 20:
            # Medium main model - use small draft
            size_category = "small"
            max_draft_params = 1.0
        elif main_model_params < 70:
            # Large main model - use medium draft
            size_category = "medium"
            max_draft_params = 3.0
        else:
            # Very large main model - use medium draft but be selective
            size_category = "medium"
            max_draft_params = 2.0
        
        # Get candidates
        candidates = cls.DRAFT_PROFILES.get(size_category, [])
        
        # Filter by compatibility and resources
        valid_candidates = []
        for profile in candidates:
            # Check memory
            if profile.memory_requirement_gb > available_memory_gb * 0.2:  # Max 20% for draft
                continue
            
            # Check parameter ratio
            if profile.num_params > max_draft_params:
                continue
            
            # Check compatibility
            if "*" not in profile.compatible_with:
                # Check specific compatibility
                if not any(compat in main_model_name for compat in profile.compatible_with):
                    continue
            
            valid_candidates.append(profile)
        
        if not valid_candidates:
            return None
        
        # Select based on expected speedup
        # Speedup = draft_speed / main_speed * acceptance_rate
        main_speed_estimate = 1000 / main_model_params  # Rough estimate
        
        best_candidate = None
        best_speedup = 0
        
        for candidate in valid_candidates:
            expected_acceptance = 0.6  # Default expectation
            if candidate.acceptance_rate > 0:
                expected_acceptance = candidate.acceptance_rate
            
            speedup = (candidate.tokens_per_second / main_speed_estimate) * expected_acceptance
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_candidate = candidate
        
        return best_candidate


class SpeculativeDecoder:
    """Speculative decoding with model-aware optimization."""
    
    def __init__(
        self,
        main_model_name: str,
        main_model_params: float,
        config: Optional[SpeculativeConfig] = None
    ):
        self.main_model_name = main_model_name
        self.main_model_params = main_model_params
        self.config = config or SpeculativeConfig()
        
        # State
        self.draft_model: Optional[Any] = None
        self.draft_profile: Optional[DraftModelProfile] = None
        self.main_model: Optional[Any] = None
        
        # Caching
        self.kv_cache: Dict[str, torch.Tensor] = {}
        self.draft_cache: deque = deque(maxlen=100)
        
        # Statistics
        self.total_drafted = 0
        self.total_accepted = 0
        self.acceptance_history = deque(maxlen=100)
        
        # Adaptive parameters
        self.current_draft_length = self.config.draft_length
    
    async def initialize(
        self,
        main_model: Any,
        available_memory_gb: float
    ) -> bool:
        """Initialize speculative decoding with appropriate draft model."""
        self.main_model = main_model
        
        # Select draft model
        self.draft_profile = DraftModelSelector.select_draft_model(
            self.main_model_name,
            self.main_model_params,
            available_memory_gb,
            self.config
        )
        
        if not self.draft_profile:
            logger.warning(f"No suitable draft model for {self.main_model_name}")
            return False
        
        # Load draft model (placeholder - actual loading would happen here)
        logger.info(f"Selected draft model: {self.draft_profile.model_name}")
        # self.draft_model = await self._load_draft_model(self.draft_profile)
        
        return True
    
    async def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate tokens using speculative decoding."""
        if not self.draft_model:
            # Fallback to regular generation
            return await self._generate_regular(input_ids, max_new_tokens, **kwargs)
        
        device = input_ids.device
        generated_ids = input_ids.clone()
        
        stats = {
            "total_drafted": 0,
            "total_accepted": 0,
            "acceptance_rate": 0.0,
            "speedup": 1.0,
            "draft_model": self.draft_profile.model_name if self.draft_profile else None
        }
        
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Adaptive draft length based on recent acceptance
            draft_length = self._get_adaptive_draft_length()
            
            # Generate draft tokens
            draft_tokens = await self._generate_draft(
                generated_ids,
                draft_length
            )
            
            # Verify with main model
            accepted_tokens, acceptance_mask = await self._verify_draft(
                generated_ids,
                draft_tokens
            )
            
            # Update statistics
            num_accepted = acceptance_mask.sum().item()
            self.total_drafted += len(draft_tokens)
            self.total_accepted += num_accepted
            self.acceptance_history.append(num_accepted / len(draft_tokens))
            
            stats["total_drafted"] += len(draft_tokens)
            stats["total_accepted"] += num_accepted
            
            # Append accepted tokens
            if num_accepted > 0:
                generated_ids = torch.cat([
                    generated_ids,
                    accepted_tokens[:num_accepted]
                ], dim=-1)
                tokens_generated += num_accepted
            else:
                # No tokens accepted - generate one with main model
                next_token = await self._generate_single_main(generated_ids)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                tokens_generated += 1
            
            # Check for EOS or max length
            if self._check_stopping_criteria(generated_ids):
                break
        
        # Calculate final statistics
        if stats["total_drafted"] > 0:
            stats["acceptance_rate"] = stats["total_accepted"] / stats["total_drafted"]
            
            # Estimate speedup
            draft_time = stats["total_drafted"] / self.draft_profile.tokens_per_second
            main_time = stats["total_accepted"] / (1000 / self.main_model_params)
            baseline_time = tokens_generated / (1000 / self.main_model_params)
            
            stats["speedup"] = baseline_time / (draft_time + main_time)
        
        return generated_ids, stats
    
    async def _generate_draft(
        self,
        input_ids: torch.Tensor,
        draft_length: int
    ) -> torch.Tensor:
        """Generate draft tokens using the draft model."""
        # Placeholder for actual draft generation
        # In production, this would use the actual draft model
        
        # Simulate draft generation
        draft_tokens = torch.randint(
            0, 50000,
            (draft_length,),
            device=input_ids.device
        )
        
        return draft_tokens
    
    async def _verify_draft(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify draft tokens with main model."""
        # Placeholder for actual verification
        # In production, this would:
        # 1. Run main model on input_ids + draft_tokens
        # 2. Compare logits with draft model predictions
        # 3. Accept/reject based on probability ratios
        
        # Simulate verification with decreasing acceptance
        acceptance_probs = torch.tensor(
            [0.9 ** i for i in range(len(draft_tokens))],
            device=input_ids.device
        )
        acceptance_mask = torch.rand(len(draft_tokens), device=input_ids.device) < acceptance_probs
        
        return draft_tokens, acceptance_mask
    
    async def _generate_single_main(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate single token with main model."""
        # Placeholder for actual generation
        return torch.randint(0, 50000, (1,), device=input_ids.device)
    
    async def _generate_regular(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Regular generation without speculation."""
        # Placeholder for regular generation
        generated = torch.randint(
            0, 50000,
            (max_new_tokens,),
            device=input_ids.device
        )
        
        output = torch.cat([input_ids, generated], dim=-1)
        stats = {"speedup": 1.0, "mode": "regular"}
        
        return output, stats
    
    def _get_adaptive_draft_length(self) -> int:
        """Get adaptive draft length based on recent acceptance."""
        if not self.config.adaptive_draft_length:
            return self.current_draft_length
        
        if len(self.acceptance_history) < 10:
            return self.current_draft_length
        
        # Calculate recent acceptance rate
        recent_acceptance = np.mean(list(self.acceptance_history)[-10:])
        
        # Adjust draft length
        if recent_acceptance > 0.8:
            # High acceptance - increase draft length
            self.current_draft_length = min(
                self.current_draft_length + 1,
                self.config.draft_length * 2
            )
        elif recent_acceptance < 0.4:
            # Low acceptance - decrease draft length
            self.current_draft_length = max(
                self.current_draft_length - 1,
                2
            )
        
        return self.current_draft_length
    
    def _check_stopping_criteria(self, generated_ids: torch.Tensor) -> bool:
        """Check if generation should stop."""
        # Check for EOS token (model-specific)
        # This is a placeholder
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get speculative decoding statistics."""
        acceptance_rate = 0.0
        if self.total_drafted > 0:
            acceptance_rate = self.total_accepted / self.total_drafted
        
        return {
            "draft_model": self.draft_profile.model_name if self.draft_profile else None,
            "draft_model_params": self.draft_profile.num_params if self.draft_profile else 0,
            "main_model_params": self.main_model_params,
            "param_ratio": self.draft_profile.num_params / self.main_model_params if self.draft_profile else 0,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": acceptance_rate,
            "current_draft_length": self.current_draft_length,
            "recent_acceptance": np.mean(list(self.acceptance_history)) if self.acceptance_history else 0
        }


class DistributedSpeculativeDecoder:
    """Distributed speculative decoding across multiple nodes."""
    
    def __init__(
        self,
        main_model_name: str,
        main_model_params: float,
        num_draft_nodes: int = 2
    ):
        self.main_model_name = main_model_name
        self.main_model_params = main_model_params
        self.num_draft_nodes = num_draft_nodes
        
        # Draft nodes
        self.draft_nodes: List[Dict[str, Any]] = []
        self.node_performance: Dict[str, float] = {}
        
        # Load balancing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.node_loads: Dict[str, int] = {}
    
    async def register_draft_node(
        self,
        node_id: str,
        profile: DraftModelProfile,
        endpoint: str
    ):
        """Register a draft node for distributed speculation."""
        self.draft_nodes.append({
            "node_id": node_id,
            "profile": profile,
            "endpoint": endpoint,
            "status": "active"
        })
        
        self.node_performance[node_id] = profile.tokens_per_second
        self.node_loads[node_id] = 0
        
        logger.info(f"Registered draft node {node_id} with model {profile.model_name}")
    
    async def generate_distributed(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate using distributed draft nodes."""
        if not self.draft_nodes:
            # No draft nodes - fallback to regular
            return input_ids, {"mode": "regular", "reason": "no_draft_nodes"}
        
        # Select best draft node based on load
        draft_node = self._select_draft_node()
        if not draft_node:
            return input_ids, {"mode": "regular", "reason": "all_nodes_busy"}
        
        # Track load
        self.node_loads[draft_node["node_id"]] += 1
        
        try:
            # Generate with selected node
            result = await self._generate_with_node(
                draft_node,
                input_ids,
                max_new_tokens
            )
            
            # Update performance metrics
            self._update_node_performance(draft_node["node_id"], result)
            
            return result
            
        finally:
            # Release load
            self.node_loads[draft_node["node_id"]] -= 1
    
    def _select_draft_node(self) -> Optional[Dict[str, Any]]:
        """Select best draft node based on load and performance."""
        active_nodes = [n for n in self.draft_nodes if n["status"] == "active"]
        if not active_nodes:
            return None
        
        # Score nodes by performance/load ratio
        best_node = None
        best_score = -1
        
        for node in active_nodes:
            node_id = node["node_id"]
            load = self.node_loads.get(node_id, 0)
            performance = self.node_performance.get(node_id, 1.0)
            
            # Skip overloaded nodes
            if load > 10:
                continue
            
            # Score = performance / (load + 1)
            score = performance / (load + 1)
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    async def _generate_with_node(
        self,
        node: Dict[str, Any],
        input_ids: torch.Tensor,
        max_new_tokens: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate using specific draft node."""
        # Placeholder for actual distributed generation
        # Would make RPC call to draft node
        
        stats = {
            "mode": "distributed_speculative",
            "draft_node": node["node_id"],
            "draft_model": node["profile"].model_name
        }
        
        # Simulate generation
        generated = torch.randint(
            0, 50000,
            (max_new_tokens,),
            device=input_ids.device
        )
        
        output = torch.cat([input_ids, generated], dim=-1)
        
        return output, stats
    
    def _update_node_performance(self, node_id: str, result: Tuple):
        """Update node performance metrics."""
        # Extract performance from result
        # Update exponential moving average
        alpha = 0.1
        
        # Placeholder - would extract actual performance
        new_performance = 100.0  # tokens/sec
        
        old_performance = self.node_performance.get(node_id, new_performance)
        self.node_performance[node_id] = alpha * new_performance + (1 - alpha) * old_performance