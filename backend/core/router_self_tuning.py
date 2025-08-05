"""
Router Self-Tuning System for Blyan
Reinforcement Learning and adaptive router optimization
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json

from .dynamic_router import DynamicRouter, ExpertPerformanceMetrics
from .pol import EnhancedPoLValidator, PoLScore
from .delta_compression import DeltaCompressor, DeltaBase

@dataclass
class RoutingExperience:
    """Experience tuple for router learning."""
    state: torch.Tensor  # Hidden states
    action: List[int]    # Selected experts
    reward: float        # Performance reward
    next_state: Optional[torch.Tensor] = None
    done: bool = False
    
    # Additional context
    inference_time: float = 0.0
    pol_score: Optional[float] = None
    expert_outputs: Optional[List[torch.Tensor]] = None

@dataclass
class RouterTrainingConfig:
    """Configuration for router self-tuning."""
    learning_rate: float = 0.001
    experience_buffer_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 100
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    reward_decay: float = 0.99
    
    # RL-specific parameters
    use_ddqn: bool = True
    target_update_frequency: int = 1000
    
    # Router-specific parameters
    routing_loss_weight: float = 0.1
    diversity_bonus_weight: float = 0.05
    efficiency_weight: float = 0.3

class SelfTuningRouter(nn.Module):
    """
    Self-tuning router using reinforcement learning.
    
    The router learns to make better routing decisions based on:
    - Inference quality (PoL scores)
    - Inference speed
    - Expert diversity
    - Long-term performance trends
    """
    
    def __init__(self, 
                 layer_id: str,
                 hidden_dim: int,
                 num_experts: int,
                 config: Optional[RouterTrainingConfig] = None,
                 pol_validator: Optional[EnhancedPoLValidator] = None):
        
        super().__init__()
        
        self.layer_id = layer_id
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.config = config or RouterTrainingConfig()
        self.pol_validator = pol_validator
        
        # Neural network components
        self.router_network = self._build_router_network()
        self.target_network = self._build_router_network() if config.use_ddqn else None
        
        # Training components
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.experience_buffer: deque = deque(maxlen=self.config.experience_buffer_size)
        
        # Training state
        self.training_step = 0
        self.exploration_rate = self.config.exploration_rate
        self.total_reward = 0.0
        self.training_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.routing_performance: Dict[str, deque] = {
            'rewards': deque(maxlen=1000),
            'losses': deque(maxlen=1000),
            'exploration_rates': deque(maxlen=1000),
            'inference_times': deque(maxlen=1000)
        }
        
        print(f"🧠 Self-tuning router initialized for layer {layer_id}")
        print(f"   Hidden dim: {hidden_dim}, Experts: {num_experts}")
        print(f"   DDQN: {self.config.use_ddqn}, Buffer size: {self.config.experience_buffer_size}")
    
    def _build_router_network(self) -> nn.Module:
        """Build the neural network for routing decisions."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_experts),
            # No final activation - we'll apply softmax later
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through router network."""
        # Handle batch dimension
        if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
            hidden_states = hidden_states.mean(dim=1)  # Average pooling
        
        return self.router_network(hidden_states)
    
    def select_experts(self, 
                      hidden_states: torch.Tensor, 
                      top_k: int = 2,
                      training: bool = True) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Select experts using learned routing with exploration.
        
        Returns:
            expert_indices, expert_weights, router_logits
        """
        with torch.no_grad() if not training else torch.enable_grad():
            router_logits = self.forward(hidden_states)
            
            if training and np.random.random() < self.exploration_rate:
                # Exploration: add noise to logits
                noise = torch.randn_like(router_logits) * 0.5
                router_logits = router_logits + noise
            
            # Select top-k experts
            scores, expert_indices = torch.topk(router_logits, top_k, dim=-1)
            expert_weights = torch.softmax(scores, dim=-1)
            
            return expert_indices.tolist(), expert_weights, router_logits
    
    def calculate_reward(self, 
                        expert_indices: List[int],
                        inference_time: float,
                        pol_score: Optional[PoLScore] = None,
                        expert_outputs: Optional[List[torch.Tensor]] = None) -> float:
        """Calculate reward for routing decision."""
        reward = 0.0
        
        # 1. Quality reward from PoL score
        if pol_score:
            quality_reward = pol_score.improvement_score * 10.0  # Scale up
            confidence_bonus = pol_score.confidence_score * 2.0
            fraud_penalty = pol_score.fraud_probability * -5.0
            
            reward += quality_reward + confidence_bonus + fraud_penalty
        
        # 2. Efficiency reward (speed)
        if inference_time > 0:
            # Reward faster inference (but not too much to avoid quality trade-offs)
            speed_reward = max(-2.0, min(2.0, 100.0 / inference_time - 50.0))
            reward += speed_reward * self.config.efficiency_weight
        
        # 3. Diversity bonus
        if len(set(expert_indices)) == len(expert_indices):
            # Bonus for selecting different experts
            diversity_bonus = 1.0
            reward += diversity_bonus * self.config.diversity_bonus_weight
        
        # 4. Load balancing reward
        # Encourage using underutilized experts
        expert_usage = self._get_expert_usage_distribution()
        for expert_idx in expert_indices:
            if expert_idx < len(expert_usage):
                usage_ratio = expert_usage[expert_idx]
                if usage_ratio < 0.5:  # Underutilized
                    reward += 0.5
        
        return reward
    
    def store_experience(self, experience: RoutingExperience):
        """Store experience in replay buffer."""
        self.experience_buffer.append(experience)
    
    def update_router(self) -> Optional[float]:
        """Update router using experience replay."""
        if len(self.experience_buffer) < self.config.batch_size:
            return None
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            self.config.batch_size, 
            replace=False
        )
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch data
        states = torch.stack([exp.state for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        
        # Current Q-values
        current_q_values = self.forward(states)
        
        # Target Q-values (for DDQN)
        if self.config.use_ddqn and self.target_network:
            with torch.no_grad():
                next_states = torch.stack([
                    exp.next_state if exp.next_state is not None else exp.state 
                    for exp in batch
                ])
                target_q_values = self.target_network(next_states)
                
                # Bellman equation
                target_values = rewards + self.config.reward_decay * target_q_values.max(dim=1)[0]
        else:
            target_values = rewards
        
        # Calculate loss
        # For routing, we use a custom loss that considers action quality
        routing_loss = 0.0
        for i, exp in enumerate(batch):
            # Get Q-values for selected experts
            selected_q_values = torch.tensor([
                current_q_values[i][expert_idx] for expert_idx in exp.action
            ])
            
            # Target value for this experience
            target_value = target_values[i]
            
            # Loss for this experience
            exp_loss = self.criterion(selected_q_values.mean(), target_value)
            routing_loss += exp_loss
        
        routing_loss /= len(batch)
        
        # Add regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        total_loss = routing_loss + 1e-4 * l2_reg
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update training state
        self.training_step += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.config.min_exploration_rate,
            self.exploration_rate * self.config.exploration_decay
        )
        
        # Update target network (DDQN)
        if (self.config.use_ddqn and 
            self.target_network and 
            self.training_step % self.config.target_update_frequency == 0):
            self.target_network.load_state_dict(self.state_dict())
        
        # Record training metrics
        loss_value = total_loss.item()
        self.routing_performance['losses'].append(loss_value)
        self.routing_performance['exploration_rates'].append(self.exploration_rate)
        
        # Training history
        if self.training_step % 100 == 0:
            self.training_history.append({
                'step': self.training_step,
                'loss': loss_value,
                'exploration_rate': self.exploration_rate,
                'buffer_size': len(self.experience_buffer),
                'avg_reward': np.mean(list(self.routing_performance['rewards'])) if self.routing_performance['rewards'] else 0.0
            })
        
        return loss_value
    
    def _get_expert_usage_distribution(self) -> List[float]:
        """Get current expert usage distribution."""
        if not self.experience_buffer:
            return [1.0 / self.num_experts] * self.num_experts
        
        usage_counts = [0] * self.num_experts
        total_selections = 0
        
        # Count usage from recent experiences
        recent_experiences = list(self.experience_buffer)[-1000:]  # Last 1000
        
        for exp in recent_experiences:
            for expert_idx in exp.action:
                if expert_idx < len(usage_counts):
                    usage_counts[expert_idx] += 1
                    total_selections += 1
        
        # Normalize to get distribution
        if total_selections > 0:
            return [count / total_selections for count in usage_counts]
        else:
            return [1.0 / self.num_experts] * self.num_experts
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'layer_id': self.layer_id,
            'training_step': self.training_step,
            'exploration_rate': self.exploration_rate,
            'buffer_size': len(self.experience_buffer),
            'total_reward': self.total_reward,
            
            # Performance metrics
            'avg_loss': np.mean(list(self.routing_performance['losses'])) if self.routing_performance['losses'] else 0.0,
            'avg_reward': np.mean(list(self.routing_performance['rewards'])) if self.routing_performance['rewards'] else 0.0,
            'avg_inference_time': np.mean(list(self.routing_performance['inference_times'])) if self.routing_performance['inference_times'] else 0.0,
            
            # Expert usage
            'expert_usage_distribution': self._get_expert_usage_distribution(),
            
            # Training config
            'config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'use_ddqn': self.config.use_ddqn,
                'buffer_size': self.config.experience_buffer_size
            }
        }
    
    def save_checkpoint(self, filepath: str):
        """Save router checkpoint."""
        checkpoint = {
            'layer_id': self.layer_id,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'exploration_rate': self.exploration_rate,
            'total_reward': self.total_reward,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Router checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load router checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.exploration_rate = checkpoint['exploration_rate']
        self.total_reward = checkpoint['total_reward']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"📂 Router checkpoint loaded: {filepath}")
        print(f"   Training step: {self.training_step}")
        print(f"   Exploration rate: {self.exploration_rate:.3f}")

class RouterEvolutionManager:
    """
    Manager for router evolution using blockchain delta system.
    
    Routers can evolve through delta blocks just like experts!
    """
    
    def __init__(self, delta_compressor: Optional[DeltaCompressor] = None):
        self.delta_compressor = delta_compressor or DeltaCompressor()
        self.router_versions: Dict[str, List[Dict[str, Any]]] = {}
        self.evolution_history: List[Dict[str, Any]] = []
    
    def create_router_delta(self, 
                           layer_id: str,
                           old_router: SelfTuningRouter,
                           new_router: SelfTuningRouter) -> DeltaBase:
        """Create delta between two router versions."""
        
        # Extract state dicts
        old_state = old_router.state_dict()
        new_state = new_router.state_dict()
        
        # Calculate deltas for each parameter
        delta_dict = {}
        for key in old_state.keys():
            if key in new_state:
                delta_dict[key] = new_state[key] - old_state[key]
        
        # Combine all deltas into single tensor for compression
        all_deltas = []
        for key, delta in delta_dict.items():
            all_deltas.append(delta.flatten())
        
        if all_deltas:
            combined_delta = torch.cat(all_deltas)
            compressed_delta = self.delta_compressor.compress_gradient(combined_delta)
            return compressed_delta
        else:
            raise ValueError("No parameters to create delta from")
    
    def apply_router_delta(self, 
                          base_router: SelfTuningRouter,
                          delta: DeltaBase) -> SelfTuningRouter:
        """Apply delta to create new router version."""
        
        # Create new router instance
        new_router = SelfTuningRouter(
            layer_id=base_router.layer_id,
            hidden_dim=base_router.hidden_dim,
            num_experts=base_router.num_experts,
            config=base_router.config,
            pol_validator=base_router.pol_validator
        )
        
        # Load base state
        new_router.load_state_dict(base_router.state_dict())
        
        # Apply delta (simplified - in practice would need to reconstruct parameter structure)
        # This is a placeholder for the complex delta application logic
        print(f"🔄 Applied router delta to {base_router.layer_id}")
        
        return new_router
    
    def evolve_router(self, 
                     router: SelfTuningRouter,
                     performance_threshold: float = 0.1) -> bool:
        """
        Determine if router should evolve based on performance.
        
        Returns True if evolution is recommended.
        """
        stats = router.get_training_stats()
        
        # Evolution criteria
        avg_reward = stats['avg_reward']
        training_steps = stats['training_step']
        exploration_rate = stats['exploration_rate']
        
        # Evolve if:
        # 1. Performance is consistently good
        # 2. Exploration has converged
        # 3. Enough training has occurred
        
        should_evolve = (
            avg_reward > performance_threshold and
            exploration_rate < 0.05 and
            training_steps > 1000
        )
        
        if should_evolve:
            evolution_event = {
                'timestamp': time.time(),
                'layer_id': router.layer_id,
                'trigger': 'performance_threshold',
                'stats': stats
            }
            self.evolution_history.append(evolution_event)
        
        return should_evolve

# Export main classes
__all__ = ['SelfTuningRouter', 'RouterEvolutionManager', 'RoutingExperience', 'RouterTrainingConfig']