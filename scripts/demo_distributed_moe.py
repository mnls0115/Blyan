#!/usr/bin/env python
"""Demo script for distributed MoE blockchain inference.

This script demonstrates the complete workflow:
1. Start expert nodes
2. Upload MoE experts to blockchain
3. Perform distributed inference
4. Track expert usage and rewards
"""

import asyncio
import json
import subprocess
import time
import requests
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.p2p.distributed_inference import ExpertNodeServer
from miner.upload_moe_parameters import MoEExpertExtractor


class DistributedMoEDemo:
    """Demo orchestrator for distributed MoE inference."""
    
    def __init__(self):
        self.api_base = "http://127.0.0.1:8000"
        self.expert_nodes = []
        self.node_processes = []
    
    def start_api_server(self):
        """Start the main API server."""
        print("🚀 Starting API server...")
        try:
            # Check if server is already running
            response = requests.get(f"{self.api_base}/docs")
            if response.status_code == 200:
                print("✓ API server already running")
                return True
        except:
            pass
        
        # Start server in background
        process = subprocess.Popen([
            "uvicorn", "api.server:app", "--reload", "--host", "127.0.0.1", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{self.api_base}/docs", timeout=1)
                if response.status_code == 200:
                    print("✓ API server started successfully")
                    return True
            except:
                time.sleep(1)
        
        print("❌ Failed to start API server")
        return False
    
    def start_expert_nodes(self):
        """Start multiple expert nodes."""
        print("\n🔧 Starting expert nodes...")
        
        # Define expert nodes with different specializations
        node_configs = [
            {
                "node_id": "node1",
                "port": 8001,
                "experts": ["layer0.expert0", "layer0.expert1", "layer1.expert0"]
            },
            {
                "node_id": "node2", 
                "port": 8002,
                "experts": ["layer1.expert1", "layer2.expert0", "layer2.expert1"]
            },
            {
                "node_id": "node3",
                "port": 8003,
                "experts": ["layer3.expert0", "layer3.expert1", "layer0.expert2"]
            }
        ]
        
        for config in node_configs:
            try:
                # Start expert node server
                server = ExpertNodeServer(
                    node_id=config["node_id"],
                    available_experts=config["experts"],
                    port=config["port"]
                )
                
                # Start server in separate process (mock for demo)
                print(f"✓ Expert node {config['node_id']} configured on port {config['port']}")
                self.expert_nodes.append(config)
                
                # Register with coordinator
                self.register_expert_node(config)
                
            except Exception as e:
                print(f"❌ Failed to start expert node {config['node_id']}: {e}")
    
    def register_expert_node(self, config):
        """Register an expert node with the coordinator."""
        try:
            response = requests.post(f"{self.api_base}/p2p/register", json={
                "node_id": config["node_id"],
                "host": "localhost",
                "port": config["port"],
                "available_experts": config["experts"]
            })
            
            if response.status_code == 200:
                print(f"✓ Registered {config['node_id']} with {len(config['experts'])} experts")
            else:
                print(f"❌ Failed to register {config['node_id']}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error registering {config['node_id']}: {e}")
    
    def initialize_blockchain(self):
        """Initialize the blockchain with meta chain."""
        print("\n⛓️  Initializing blockchain...")
        
        try:
            # Check if meta chain exists
            response = requests.get(f"{self.api_base}/chain/A/blocks")
            if response.status_code == 200 and response.json()["blocks"]:
                print("✓ Blockchain already initialized")
                return True
        except:
            pass
        
        # Initialize meta chain with MoE model spec
        print("Creating genesis block for MoE model...")
        # This would typically be done via the genesis script from README
        # For demo purposes, we'll assume it's already done
        return True
    
    def simulate_expert_upload(self):
        """Simulate uploading MoE experts to the blockchain."""
        print("\n📤 Simulating expert uploads...")
        
        # Mock expert upload for each node's experts
        mock_meta_hash = "abc123def456"  # This would be the actual meta block hash
        
        uploaded_experts = []
        for node_config in self.expert_nodes:
            for expert_name in node_config["experts"]:
                try:
                    # Mock upload request
                    upload_data = {
                        "miner_address": f"miner_{node_config['node_id']}",
                        "miner_pub": "mock_pubkey_" + expert_name.replace(".", "_"),
                        "payload_sig": "mock_signature",
                        "expert_name": expert_name,
                        "layer_id": expert_name.split(".")[0],
                        "block_type": "expert",
                        "depends_on": [mock_meta_hash],
                        "tensor_data_b64": "bW9ja190ZW5zb3JfZGF0YQ==",  # "mock_tensor_data" in base64
                        "candidate_loss": 0.8,
                        "previous_loss": 1.0
                    }
                    
                    # Note: This would fail with actual API due to signature verification
                    # In a real demo, you'd use the upload_moe_parameters.py script
                    print(f"✓ Would upload expert {expert_name} to blockchain")
                    uploaded_experts.append(expert_name)
                    
                except Exception as e:
                    print(f"❌ Failed to upload {expert_name}: {e}")
        
        print(f"✓ Simulated upload of {len(uploaded_experts)} experts")
        return uploaded_experts
    
    def test_inference_modes(self):
        """Test different inference modes."""
        print("\n🧠 Testing inference modes...")
        
        test_prompt = "Explain quantum computing in simple terms"
        
        # Test 1: Standard inference
        print("\n1. Standard Inference:")
        try:
            response = requests.post(f"{self.api_base}/chat", json={
                "prompt": test_prompt,
                "use_moe": False,
                "max_new_tokens": 32
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Response: {result['response']}")
                print(f"✓ Time: {result['inference_time']:.3f}s")
            else:
                print(f"❌ Standard inference failed: {response.text}")
        except Exception as e:
            print(f"❌ Standard inference error: {e}")
        
        # Test 2: MoE inference
        print("\n2. MoE Inference:")
        try:
            response = requests.post(f"{self.api_base}/chat", json={
                "prompt": test_prompt,
                "use_moe": True,
                "top_k_experts": 2,
                "max_new_tokens": 32
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Response: {result['response']}")
                print(f"✓ Time: {result['inference_time']:.3f}s")
                print(f"✓ Expert usage: {result['expert_usage']}")
            else:
                print(f"❌ MoE inference failed: {response.text}")
        except Exception as e:
            print(f"❌ MoE inference error: {e}")
        
        # Test 3: Distributed inference (would require actual expert nodes)
        print("\n3. Distributed Inference:")
        try:
            response = requests.post(f"{self.api_base}/chat/distributed", json={
                "prompt": test_prompt,
                "top_k_experts": 2,
                "max_new_tokens": 32
            })
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Response: {result['response']}")
                print(f"✓ Time: {result['inference_time']:.3f}s")
                print(f"✓ Expert usage: {result['expert_usage']}")
            else:
                print(f"❌ Distributed inference failed: {response.text}")
        except Exception as e:
            print(f"❌ Distributed inference error: {e}")
    
    def check_expert_analytics(self):
        """Check expert usage analytics."""
        print("\n📊 Checking expert analytics...")
        
        try:
            # Get top experts
            response = requests.get(f"{self.api_base}/experts/top?limit=5")
            if response.status_code == 200:
                top_experts = response.json()["experts"]
                print(f"✓ Found {len(top_experts)} experts with usage data")
                
                for expert in top_experts[:3]:  # Show top 3
                    print(f"  - {expert['expert_name']}: {expert['call_count']} calls, "
                          f"{expert['average_response_time']:.3f}s avg, "
                          f"{expert['current_reward_multiplier']:.2f}x reward")
            else:
                print(f"❌ Failed to get expert analytics: {response.text}")
                
        except Exception as e:
            print(f"❌ Analytics error: {e}")
    
    def check_p2p_network(self):
        """Check P2P network status."""
        print("\n🌐 Checking P2P network...")
        
        try:
            response = requests.get(f"{self.api_base}/p2p/nodes")
            if response.status_code == 200:
                nodes = response.json()["nodes"]
                print(f"✓ Found {len(nodes)} registered expert nodes")
                
                for node in nodes:
                    print(f"  - {node['node_id']}: {len(node['available_experts'])} experts, "
                          f"load={node['load_factor']:.2f}")
            else:
                print(f"❌ Failed to get P2P status: {response.text}")
                
        except Exception as e:
            print(f"❌ P2P network error: {e}")
    
    def run_demo(self):
        """Run the complete demo."""
        print("🎯 Starting Distributed MoE Blockchain Demo")
        print("=" * 50)
        
        # Step 1: Start API server
        if not self.start_api_server():
            return
        
        # Step 2: Start expert nodes
        self.start_expert_nodes()
        
        # Step 3: Initialize blockchain
        self.initialize_blockchain()
        
        # Step 4: Simulate expert uploads
        self.simulate_expert_upload()
        
        # Wait a moment for everything to settle
        time.sleep(2)
        
        # Step 5: Test inference modes
        self.test_inference_modes()
        
        # Step 6: Check analytics
        self.check_expert_analytics()
        
        # Step 7: Check P2P network
        self.check_p2p_network()
        
        print("\n🎉 Demo completed!")
        print("\nNext steps:")
        print("1. Visit http://127.0.0.1:8000/docs for API documentation")
        print("2. Try uploading real MoE models with upload_moe_parameters.py")
        print("3. Start actual expert node servers for true distributed inference")
        print("4. Monitor expert performance and rewards in real-time")
    
    def cleanup(self):
        """Clean up processes."""
        for process in self.node_processes:
            process.terminate()


if __name__ == "__main__":
    demo = DistributedMoEDemo()
    
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    finally:
        demo.cleanup()