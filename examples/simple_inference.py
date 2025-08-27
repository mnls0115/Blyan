#!/usr/bin/env python3
"""
Simple example for running inference with Blyan AI.
No web interface needed!
"""

import asyncio
import sys
import os
sys.path.append('..')  # Add parent directory to path

from client.blyan_client import BlyanClient


async def main():
    # API server URL (change this to your server)
    API_URL = os.getenv('BLYAN_API_URL', 'http://165.227.221.225:8000')
    
    print("🤖 Blyan AI Client Example")
    print(f"📡 Connecting to: {API_URL}")
    print("-" * 50)
    
    async with BlyanClient(API_URL) as client:
        # Check available nodes
        nodes = await client.list_nodes()
        print(f"✅ Connected! Found {len(nodes)} expert nodes")
        
        # Interactive chat loop
        print("\n💬 Chat with Blyan AI (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            # Get user input
            prompt = input("\nYou: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not prompt:
                continue
            
            try:
                # Run inference
                print("🤔 Thinking...", end='', flush=True)
                
                response = await client.chat(
                    prompt,
                    use_moe=True,  # Use Mixture of Experts
                    top_k_experts=2,
                    max_new_tokens=256
                )
                
                print(f"\r🤖 Blyan: {response}")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")