#!/usr/bin/env python
"""Quick inference test script for debugging MoE selective loading.

This script focuses only on testing the inference part,
assuming the blockchain is already set up with expert blocks.
"""

import requests
import json
import time
from typing import Dict, Any


def test_all_inference_modes():
    """Test all three inference modes with detailed logging."""
    
    api_base = "http://127.0.0.1:8000"
    test_prompt = "What is machine learning?"
    
    print("🧠 Testing AI-Block Inference Modes")
    print("=" * 50)
    print(f"Test prompt: '{test_prompt}'")
    print()
    
    # Test 1: Standard inference
    print("1️⃣ Standard Inference (no MoE)")
    try:
        start_time = time.time()
        response = requests.post(f"{api_base}/chat", json={
            "prompt": test_prompt,
            "use_moe": False,
            "max_new_tokens": 32
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ SUCCESS ({elapsed:.2f}s)")
            print(f"Response: {result['response']}")
            print(f"Inference time: {result.get('inference_time', 'N/A')}s")
            print(f"Expert usage: {result.get('expert_usage', 'None')}")
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test 2: MoE inference
    print("2️⃣ MoE Inference (selective expert loading)")
    try:
        start_time = time.time()
        response = requests.post(f"{api_base}/chat", json={
            "prompt": test_prompt,
            "use_moe": True,
            "top_k_experts": 2,
            "max_new_tokens": 32
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ SUCCESS ({elapsed:.2f}s)")
            print(f"Response: {result['response']}")
            print(f"Inference time: {result.get('inference_time', 'N/A')}s")
            
            expert_usage = result.get('expert_usage', {})
            if expert_usage:
                print("🎯 Expert Usage Details:")
                for expert_name, usage_info in expert_usage.items():
                    if isinstance(usage_info, dict):
                        print(f"  - {expert_name}: {usage_info}")
                    else:
                        print(f"  - {expert_name}: {usage_info}s")
            else:
                print("⚠️  No expert usage information available")
                
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test 3: Distributed inference
    print("3️⃣ Distributed Inference (P2P expert nodes)")
    try:
        start_time = time.time()
        response = requests.post(f"{api_base}/chat/distributed", json={
            "prompt": test_prompt,
            "top_k_experts": 2,
            "max_new_tokens": 32
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ SUCCESS ({elapsed:.2f}s)")
            print(f"Response: {result['response']}")
            print(f"Inference time: {result.get('inference_time', 'N/A')}s")
            
            expert_usage = result.get('expert_usage', {})
            if expert_usage:
                print("🌐 Distributed Expert Usage:")
                for expert_name, usage_info in expert_usage.items():
                    print(f"  - {expert_name}: {usage_info}")
            else:
                print("⚠️  No distributed experts available (register nodes first)")
                
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 50)


def check_expert_status():
    """Check what experts are available in the system."""
    
    api_base = "http://127.0.0.1:8000"
    
    print("\n📊 Expert Status Check")
    print("-" * 30)
    
    # Check expert analytics
    try:
        response = requests.get(f"{api_base}/experts/top?limit=10")
        if response.status_code == 200:
            experts = response.json()["experts"]
            print(f"📈 Found {len(experts)} experts with usage data:")
            
            for expert in experts:
                print(f"  - {expert['expert_name']}: "
                      f"{expert['call_count']} calls, "
                      f"{expert['average_response_time']:.3f}s avg, "
                      f"{expert['current_reward_multiplier']:.2f}x reward")
        else:
            print("⚠️  No expert analytics available")
            
    except Exception as e:
        print(f"❌ Analytics error: {e}")
    
    # Check blockchain state
    try:
        print(f"\n⛓️  Blockchain State:")
        
        # Meta chain
        response = requests.get(f"{api_base}/chain/A/blocks?limit=5")
        if response.status_code == 200:
            meta_blocks = response.json()["blocks"]
            print(f"  Meta-chain (A): {len(meta_blocks)} blocks")
        
        # Parameter chain
        response = requests.get(f"{api_base}/chain/B/blocks?limit=20")
        if response.status_code == 200:
            param_blocks = response.json()["blocks"]
            print(f"  Parameter-chain (B): {len(param_blocks)} blocks")
            
            # Count by type (if available)
            expert_count = sum(1 for b in param_blocks if 'expert' in str(b))
            router_count = sum(1 for b in param_blocks if 'router' in str(b))
            
            if expert_count > 0:
                print(f"    └─ Expert blocks: {expert_count}")
            if router_count > 0:
                print(f"    └─ Router blocks: {router_count}")
                
    except Exception as e:
        print(f"❌ Blockchain state error: {e}")
    
    # Check P2P nodes
    try:
        print(f"\n🌐 P2P Network:")
        response = requests.get(f"{api_base}/p2p/nodes")
        if response.status_code == 200:
            nodes = response.json()["nodes"]
            print(f"  Registered nodes: {len(nodes)}")
            
            for node in nodes:
                experts_count = len(node.get('available_experts', []))
                print(f"    - {node['node_id']}: {experts_count} experts, "
                      f"load={node.get('load_factor', 0):.2f}")
        else:
            print("  No P2P nodes registered")
            
    except Exception as e:
        print(f"❌ P2P network error: {e}")


def quick_debug_test():
    """Quick test to debug specific issues."""
    
    api_base = "http://127.0.0.1:8000"
    
    print("🔧 Quick Debug Test")
    print("=" * 30)
    
    # Test server availability
    try:
        response = requests.get(f"{api_base}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API server is responding")
        else:
            print(f"⚠️  API server status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach API server: {e}")
        print("💡 Make sure to run: uvicorn api.server:app --reload")
        return
    
    # Test simple chat
    try:
        response = requests.post(f"{api_base}/chat", json={
            "prompt": "Hello",
            "use_moe": False,
            "max_new_tokens": 10
        }, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Basic chat works: '{result['response'][:30]}...'")
        else:
            print(f"❌ Basic chat failed: {response.status_code}")
            print(f"Error details: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Chat test error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        quick_debug_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        check_expert_status()
    else:
        test_all_inference_modes()
        check_expert_status()
        
    print(f"\n💡 Usage:")
    print(f"  python {sys.argv[0]}        # Full inference test")
    print(f"  python {sys.argv[0]} debug  # Quick debug test") 
    print(f"  python {sys.argv[0]} status # Expert status only")