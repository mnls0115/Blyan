#!/usr/bin/env python3
"""
Demonstration of the complete learning cycle
Shows how BLY burn triggers learning and rewards distribution
"""

import asyncio
import sys
from pathlib import Path
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.learning.learning_cycle_coordinator import (
    get_learning_coordinator,
    LearningCyclePhase
)
from backend.economics.budget_controller import get_budget_controller
from backend.core.dataset_chain import DatasetChain
from backend.core.chain import Chain


async def simulate_inference_activity(coordinator, num_inferences=20):
    """Simulate inference activity that burns BLY"""
    print("\n🔥 Simulating inference activity...")
    
    for i in range(num_inferences):
        # Each inference burns some BLY (simulated)
        bly_burned = Decimal("100")  # 100 BLY per inference
        
        print(f"  Inference {i+1}: Burned {bly_burned} BLY")
        coordinator.record_inference_burn(bly_burned)
        
        # Check progress
        progress = coordinator.trigger_monitor.get_progress()
        print(f"    Progress: {progress['accumulated_burn']}/{progress['threshold']} BLY "
              f"({progress['progress_percent']:.1f}%)")
        
        # Small delay between inferences
        await asyncio.sleep(0.5)
        
        # Check if learning was triggered
        if coordinator.current_round:
            print(f"\n✅ Learning triggered at {progress['accumulated_burn']} BLY burned!")
            break


async def main():
    print("=" * 60)
    print("🎓 Blyan Learning Cycle Demonstration")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    # Chains
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    dataset_chain = DatasetChain(data_dir, "D")
    param_chain = Chain(data_dir, "B")
    
    # Budget controller
    budget_controller = get_budget_controller()
    budget_controller.add_revenue(Decimal("10000"))  # Add initial pool balance
    
    # Mock distributed coordinator
    class MockDistributedCoordinator:
        def __init__(self):
            self.registry = MockRegistry()
    
    class MockRegistry:
        def __init__(self):
            self.nodes = {
                "gpu_node_1": MockNode("gpu_node_1"),
                "gpu_node_2": MockNode("gpu_node_2"),
                "gpu_node_3": MockNode("gpu_node_3"),
            }
    
    class MockNode:
        def __init__(self, node_id):
            self.node_id = node_id
            self.status = "healthy"
    
    distributed_coordinator = MockDistributedCoordinator()
    
    # Initialize learning coordinator
    coordinator = get_learning_coordinator(
        service_node_url="http://localhost:8000",
        dataset_chain=dataset_chain,
        param_chain=param_chain,
        distributed_coordinator=distributed_coordinator,
        budget_controller=budget_controller
    )
    
    # Start coordinator
    await coordinator.start()
    print("✅ Learning coordinator started")
    
    # Show initial status
    print("\n📊 Initial Status:")
    status = coordinator.get_status()
    print(f"  Trigger threshold: {status['trigger_progress']['threshold']} BLY")
    print(f"  Current burn: {status['trigger_progress']['accumulated_burn']} BLY")
    print(f"  Rounds completed: {status['rounds_completed']}")
    
    # Phase 1: Accumulate BLY burn through inference
    print("\n" + "=" * 60)
    print("PHASE 1: BLY Burn Accumulation")
    print("=" * 60)
    
    await simulate_inference_activity(coordinator)
    
    # Phase 2: Learning Round Execution
    if coordinator.current_round:
        print("\n" + "=" * 60)
        print("PHASE 2: Learning Round Execution")
        print("=" * 60)
        
        round_id = coordinator.current_round.round_id
        print(f"\n🔄 Learning Round: {round_id}")
        
        # Monitor phases
        phases_seen = set()
        timeout = 60  # 60 second timeout for demo
        start_time = asyncio.get_event_loop().time()
        
        while coordinator.current_round:
            current_phase = coordinator.current_round.phase
            
            if current_phase not in phases_seen:
                phases_seen.add(current_phase)
                print(f"\n📍 Phase: {current_phase.value}")
                
                if current_phase == LearningCyclePhase.NOTIFYING:
                    nodes = coordinator.current_round.participating_nodes
                    print(f"  Notifying {len(nodes)} GPU nodes: {nodes}")
                    
                elif current_phase == LearningCyclePhase.DATA_ALLOCATION:
                    allocation = coordinator.current_round.data_allocation
                    total_datasets = sum(len(d) for d in allocation.values())
                    print(f"  Allocated {total_datasets} datasets across nodes")
                    
                elif current_phase == LearningCyclePhase.TRAINING:
                    print(f"  Training in progress...")
                    print(f"  Participating nodes: {coordinator.current_round.participating_nodes}")
                    
                elif current_phase == LearningCyclePhase.CONSENSUS_BUILDING:
                    print(f"  Building consensus on trained deltas...")
                    
                elif current_phase == LearningCyclePhase.DELTA_CREATION:
                    print(f"  Creating delta block on blockchain...")
                    
                elif current_phase == LearningCyclePhase.REWARD_DISTRIBUTION:
                    print(f"  Distributing rewards to nodes...")
            
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                print("\n⏱️ Demo timeout reached")
                break
            
            await asyncio.sleep(1)
    
    # Phase 3: Results
    print("\n" + "=" * 60)
    print("PHASE 3: Learning Cycle Results")
    print("=" * 60)
    
    final_status = coordinator.get_status()
    
    print(f"\n📊 Final Status:")
    print(f"  Rounds completed: {final_status['rounds_completed']}")
    print(f"  Trigger progress reset: {final_status['trigger_progress']['accumulated_burn']} BLY")
    
    if coordinator.round_history:
        last_round = coordinator.round_history[-1]
        print(f"\n✅ Last Round Summary:")
        print(f"  Round ID: {last_round.round_id}")
        print(f"  BLY Burned (trigger): {last_round.bly_burned_trigger}")
        print(f"  Nodes participated: {len(last_round.participating_nodes)}")
        
        if last_round.rewards_distributed:
            print(f"  Rewards distributed:")
            for node_id, reward in last_round.rewards_distributed.items():
                print(f"    {node_id}: {reward} BLY")
        
        if last_round.completion_time:
            duration = last_round.completion_time - last_round.trigger_time
            print(f"  Duration: {duration:.1f} seconds")
    
    # Budget status
    budget_status = budget_controller.get_budget_status()
    print(f"\n💰 Budget Status:")
    print(f"  Pool balance: {budget_status['pool_balance']:.2f} BLY")
    print(f"  Pending rewards: {budget_status['pending_queue_size']}")
    print(f"  Demand ratio: {budget_status['demand_ratio']:.2f}")
    
    print("\n" + "=" * 60)
    print("✨ Learning Cycle Demo Complete!")
    print("=" * 60)
    
    print("\n📝 Key Insights:")
    print("1. BLY burn from inference accumulates until threshold")
    print("2. Service node triggers learning when threshold reached")
    print("3. GPU nodes are notified and allocated data from pool")
    print("4. Training occurs in parallel across nodes")
    print("5. Consensus is built on the trained deltas")
    print("6. Delta block is created on blockchain")
    print("7. Rewards are distributed proportionally")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted")