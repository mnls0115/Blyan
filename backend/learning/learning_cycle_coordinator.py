"""
Complete Learning Cycle Coordinator for Blyan Network
Manages the entire learning flow from trigger to delta block creation
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class LearningCyclePhase(Enum):
    """Phases of the learning cycle"""
    IDLE = "idle"                          # Waiting for trigger
    TRIGGERED = "triggered"                # BLY burn threshold reached
    NOTIFYING = "notifying"                # Notifying GPU nodes
    DATA_ALLOCATION = "data_allocation"    # Allocating data from pool
    TRAINING = "training"                  # Nodes training
    CONSENSUS_BUILDING = "consensus"       # Building consensus
    DELTA_CREATION = "delta_creation"      # Creating delta block
    REWARD_DISTRIBUTION = "reward_distribution"  # Distributing rewards


@dataclass
class LearningRound:
    """Represents a complete learning round"""
    round_id: str
    trigger_time: float
    bly_burned_trigger: Decimal  # Amount of BLY burned that triggered learning
    participating_nodes: List[str]
    data_allocation: Dict[str, List[str]]  # node_id -> dataset_ids
    target_expert: str
    base_version: str
    consensus_delta: Optional[Any] = None
    rewards_distributed: Dict[str, Decimal] = None
    phase: LearningCyclePhase = LearningCyclePhase.IDLE
    completion_time: Optional[float] = None


class LearningTriggerMonitor:
    """Monitors BLY burn and triggers learning when threshold reached"""
    
    def __init__(self, threshold_bly: Decimal = Decimal("1000")):
        self.threshold_bly = threshold_bly  # BLY burned to trigger learning
        self.accumulated_burn = Decimal("0")
        self.last_trigger_time = 0
        self.min_interval = 3600  # Minimum 1 hour between rounds
        
    def record_burn(self, amount: Decimal) -> bool:
        """
        Record BLY burn from inference and check if learning should trigger.
        
        Returns:
            True if learning should be triggered
        """
        self.accumulated_burn += amount
        
        # Check trigger conditions
        current_time = time.time()
        time_since_last = current_time - self.last_trigger_time
        
        if (self.accumulated_burn >= self.threshold_bly and 
            time_since_last >= self.min_interval):
            logger.info(f"🔥 Learning triggered! {self.accumulated_burn} BLY burned")
            self.accumulated_burn = Decimal("0")
            self.last_trigger_time = current_time
            return True
            
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress toward learning trigger"""
        return {
            "accumulated_burn": float(self.accumulated_burn),
            "threshold": float(self.threshold_bly),
            "progress_percent": float(self.accumulated_burn / self.threshold_bly * 100),
            "time_until_eligible": max(0, self.min_interval - (time.time() - self.last_trigger_time))
        }


class DataPoolAllocator:
    """Allocates training data from the data pool to GPU nodes"""
    
    def __init__(self, dataset_chain):
        self.dataset_chain = dataset_chain
        
    async def allocate_data_to_nodes(
        self,
        node_ids: List[str],
        round_id: str
    ) -> Dict[str, List[str]]:
        """
        Allocate training data from pool to nodes.
        
        Strategy:
        1. Get GOLD tier datasets first
        2. Distribute evenly across nodes
        3. Ensure no overlap (each node gets unique data)
        4. Track allocation for verification
        """
        from backend.core.dataset_block import DatasetQualityTier
        
        # Get available datasets
        gold_datasets = self.dataset_chain.get_datasets_by_tier(DatasetQualityTier.GOLD)
        silver_datasets = self.dataset_chain.get_datasets_by_tier(DatasetQualityTier.SILVER)
        
        all_datasets = gold_datasets + silver_datasets[:len(silver_datasets)//2]  # Use half of silver
        
        if not all_datasets:
            logger.error("No datasets available in pool")
            return {}
        
        # Allocate evenly
        allocation = {}
        datasets_per_node = max(1, len(all_datasets) // len(node_ids))
        
        for i, node_id in enumerate(node_ids):
            start_idx = i * datasets_per_node
            end_idx = start_idx + datasets_per_node
            
            # Last node gets remaining datasets
            if i == len(node_ids) - 1:
                end_idx = len(all_datasets)
            
            node_datasets = all_datasets[start_idx:end_idx]
            allocation[node_id] = [d.dataset_id for d in node_datasets]
            
            logger.info(f"Allocated {len(node_datasets)} datasets to {node_id}")
        
        return allocation


class LearningCycleCoordinator:
    """
    Main coordinator for the complete learning cycle.
    Manages the flow from BLY burn trigger to delta block creation.
    """
    
    def __init__(self,
                 service_node_url: str,
                 dataset_chain,
                 param_chain,
                 distributed_coordinator,
                 budget_controller):
        
        self.service_node_url = service_node_url
        self.dataset_chain = dataset_chain
        self.param_chain = param_chain
        self.distributed_coordinator = distributed_coordinator
        self.budget_controller = budget_controller
        
        # Components
        self.trigger_monitor = LearningTriggerMonitor()
        self.data_allocator = DataPoolAllocator(dataset_chain)
        
        # State
        self.current_round: Optional[LearningRound] = None
        self.round_history: List[LearningRound] = []
        self.is_running = False
        
    async def start(self):
        """Start the learning cycle coordinator"""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("🚀 Learning Cycle Coordinator started")
        
        # Start monitoring loop
        asyncio.create_task(self._monitor_loop())
        
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check if we're in a round
                if self.current_round:
                    await self._process_current_round()
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(30)
    
    def record_inference_burn(self, bly_amount: Decimal):
        """
        Record BLY burned from inference.
        Called by billing gateway after each inference payment.
        """
        if self.trigger_monitor.record_burn(bly_amount):
            # Learning triggered!
            asyncio.create_task(self._start_learning_round())
    
    async def _start_learning_round(self):
        """Start a new learning round"""
        if self.current_round:
            logger.warning("Learning round already in progress")
            return
        
        # Create new round
        round_id = hashlib.sha256(f"round_{time.time()}".encode()).hexdigest()[:16]
        
        self.current_round = LearningRound(
            round_id=round_id,
            trigger_time=time.time(),
            bly_burned_trigger=self.trigger_monitor.threshold_bly,
            participating_nodes=[],
            data_allocation={},
            target_expert="layer0.expert0",  # TODO: Dynamic selection
            base_version="",
            phase=LearningCyclePhase.TRIGGERED
        )
        
        logger.info(f"📚 Starting learning round {round_id}")
        
        # Move to notification phase
        await self._notify_gpu_nodes()
    
    async def _notify_gpu_nodes(self):
        """Notify GPU nodes to start learning"""
        if not self.current_round:
            return
            
        self.current_round.phase = LearningCyclePhase.NOTIFYING
        
        # Get available GPU nodes
        available_nodes = []
        if self.distributed_coordinator:
            for node_id, node in self.distributed_coordinator.registry.nodes.items():
                if node.status == "healthy":
                    available_nodes.append(node_id)
        
        if not available_nodes:
            logger.error("No GPU nodes available for learning")
            self.current_round = None
            return
        
        self.current_round.participating_nodes = available_nodes
        logger.info(f"Notifying {len(available_nodes)} GPU nodes")
        
        # Send notification to nodes
        notifications = []
        for node_id in available_nodes:
            notification = self._send_learning_notification(node_id)
            notifications.append(notification)
        
        await asyncio.gather(*notifications)
        
        # Move to data allocation
        await self._allocate_training_data()
    
    async def _send_learning_notification(self, node_id: str):
        """Send learning start notification to a GPU node"""
        try:
            import httpx
            
            # Get node info from registry
            node = self.distributed_coordinator.registry.nodes.get(node_id)
            if not node:
                logger.error(f"Node {node_id} not found in registry")
                return
            
            # Build URL
            node_url = f"http://{node.host}:{node.port}/learning/start"
            
            # Send notification
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    node_url,
                    json={
                        "round_id": self.current_round.round_id,
                        "target_expert": self.current_round.target_expert,
                        "base_version": self.current_round.base_version
                    }
                )
                
                if resp.status_code == 200:
                    logger.info(f"✅ Notified {node_id} to start learning")
                else:
                    logger.warning(f"Failed to notify {node_id}: {resp.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to notify {node_id}: {e}")
    
    async def _allocate_training_data(self):
        """Allocate training data from pool to nodes"""
        if not self.current_round:
            return
            
        self.current_round.phase = LearningCyclePhase.DATA_ALLOCATION
        
        # Allocate data
        allocation = await self.data_allocator.allocate_data_to_nodes(
            self.current_round.participating_nodes,
            self.current_round.round_id
        )
        
        self.current_round.data_allocation = allocation
        
        # Send allocations to nodes
        for node_id, dataset_ids in allocation.items():
            await self._send_data_allocation(node_id, dataset_ids)
        
        # Move to training phase
        self.current_round.phase = LearningCyclePhase.TRAINING
        logger.info(f"Training started for round {self.current_round.round_id}")
    
    async def _send_data_allocation(self, node_id: str, dataset_ids: List[str]):
        """Send data allocation to a GPU node"""
        try:
            # In production: Send dataset IDs to node
            logger.info(f"Sent {len(dataset_ids)} datasets to {node_id}")
            
        except Exception as e:
            logger.error(f"Failed to send allocation to {node_id}: {e}")
    
    async def _process_current_round(self):
        """Process the current learning round based on its phase"""
        if not self.current_round:
            return
        
        phase = self.current_round.phase
        
        if phase == LearningCyclePhase.TRAINING:
            # Check if nodes have finished training
            if await self._check_training_complete():
                await self._build_consensus()
                
        elif phase == LearningCyclePhase.CONSENSUS_BUILDING:
            # Check if consensus is reached
            if await self._check_consensus_complete():
                await self._create_delta_block()
                
        elif phase == LearningCyclePhase.DELTA_CREATION:
            # Delta block created, distribute rewards
            await self._distribute_rewards()
    
    async def _check_training_complete(self) -> bool:
        """Check if all nodes have completed training"""
        # In production: Query nodes for training status
        # For now, simulate with timeout
        
        elapsed = time.time() - self.current_round.trigger_time
        if elapsed > 300:  # 5 minutes for training
            logger.info("Training phase complete")
            return True
        return False
    
    async def _build_consensus(self):
        """Build consensus on the trained delta"""
        if not self.current_round:
            return
            
        self.current_round.phase = LearningCyclePhase.CONSENSUS_BUILDING
        logger.info("Building consensus on trained deltas")
        
        # In production: Collect deltas from nodes and aggregate
        # Use Byzantine fault-tolerant aggregation
        
        # Simulate consensus
        await asyncio.sleep(10)
        
        # Store consensus delta
        self.current_round.consensus_delta = {"mock": "delta"}
    
    async def _check_consensus_complete(self) -> bool:
        """Check if consensus building is complete"""
        return self.current_round.consensus_delta is not None
    
    async def _create_delta_block(self):
        """Create delta block on blockchain"""
        if not self.current_round:
            return
            
        self.current_round.phase = LearningCyclePhase.DELTA_CREATION
        logger.info("Creating delta block on chain")
        
        # Create block metadata
        metadata = {
            "round_id": self.current_round.round_id,
            "expert": self.current_round.target_expert,
            "base_version": self.current_round.base_version,
            "nodes": self.current_round.participating_nodes,
            "datasets_used": sum(len(d) for d in self.current_round.data_allocation.values()),
            "timestamp": time.time()
        }
        
        # Add delta to blockchain
        if self.current_round.consensus_delta:
            block_hash = self.param_chain.add_block(
                json.dumps(self.current_round.consensus_delta).encode(),
                block_type="delta",
                metadata=json.dumps(metadata)
            )
            logger.info(f"✅ Delta block created: {block_hash}")
    
    async def _distribute_rewards(self):
        """Distribute learning rewards to participating nodes"""
        if not self.current_round:
            return
            
        self.current_round.phase = LearningCyclePhase.REWARD_DISTRIBUTION
        logger.info("Distributing learning rewards")
        
        # Calculate rewards based on contribution
        total_reward = Decimal("1000")  # Base learning reward
        rewards = {}
        
        num_nodes = len(self.current_round.participating_nodes)
        if num_nodes > 0:
            reward_per_node = total_reward / num_nodes
            
            for node_id in self.current_round.participating_nodes:
                rewards[node_id] = reward_per_node
                
                # Request reward from budget controller
                await self.budget_controller.request_reward(
                    request_type="learning",
                    recipient=node_id,
                    amount=reward_per_node,
                    metadata={
                        "round_id": self.current_round.round_id,
                        "datasets_processed": len(self.current_round.data_allocation.get(node_id, []))
                    }
                )
        
        self.current_round.rewards_distributed = rewards
        self.current_round.completion_time = time.time()
        
        # Archive round
        self.round_history.append(self.current_round)
        
        # Clear current round
        logger.info(f"✅ Learning round {self.current_round.round_id} complete")
        self.current_round = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current learning cycle status"""
        status = {
            "is_running": self.is_running,
            "trigger_progress": self.trigger_monitor.get_progress(),
            "rounds_completed": len(self.round_history)
        }
        
        if self.current_round:
            status["current_round"] = {
                "round_id": self.current_round.round_id,
                "phase": self.current_round.phase.value,
                "participating_nodes": len(self.current_round.participating_nodes),
                "elapsed_time": time.time() - self.current_round.trigger_time
            }
        
        return status


# Singleton instance
_learning_coordinator = None

def get_learning_coordinator(service_node_url: str,
                            dataset_chain,
                            param_chain,
                            distributed_coordinator,
                            budget_controller) -> LearningCycleCoordinator:
    """Get or create learning coordinator singleton"""
    global _learning_coordinator
    if _learning_coordinator is None:
        _learning_coordinator = LearningCycleCoordinator(
            service_node_url,
            dataset_chain,
            param_chain,
            distributed_coordinator,
            budget_controller
        )
    return _learning_coordinator