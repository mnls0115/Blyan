#!/usr/bin/env python3
"""
Setup script for main node - RUN THIS ONLY ON YOUR DIGITAL OCEAN SERVER
This prevents others from claiming to be the main node
"""

import os
import sys
import secrets
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from backend.api.node_auth import NodeAuthenticator

def setup_main_node():
    """One-time setup for main node"""
    print("=== Blyan Main Node Setup ===")
    print("⚠️  This should only be run on your Digital Ocean server!")
    print()
    
    # Check if already setup
    auth = NodeAuthenticator()
    if auth.config["main_node"]["id"]:
        print("❌ Main node already configured!")
        print(f"   Node ID: {auth.config['main_node']['id']}")
        print(f"   Host: {auth.config['main_node']['host']}")
        return
    
    # Generate or get secret
    secret = os.environ.get('BLYAN_MAIN_NODE_SECRET')
    if not secret:
        print("⚠️  BLYAN_MAIN_NODE_SECRET not set in environment")
        print("   Generating a new secret...")
        secret = secrets.token_hex(32)
        print(f"\n📝 Add this to your environment variables:")
        print(f"   export BLYAN_MAIN_NODE_SECRET={secret}")
        print("\n   Then run this script again.")
        return
    
    # Get node information
    node_id = input("Enter main node ID (e.g., 'main-do-nyc1'): ").strip()
    host = input("Enter main node host (e.g., 'your-domain.com' or IP): ").strip()
    
    if not node_id or not host:
        print("❌ Node ID and host are required!")
        return
    
    # Register main node
    result = auth.register_main_node(node_id, host, secret)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("\n✅ Main node successfully registered!")
    print(f"   Node ID: {node_id}")
    print(f"   Host: {host}")
    print(f"\n🔑 Your main node auth token (SAVE THIS SECURELY!):")
    print(f"   {result['auth_token']}")
    print(f"\n📝 Add this to your .env file on the main node:")
    print(f"   BLYAN_MAIN_NODE_TOKEN={result['auth_token']}")
    print(f"\n⚠️  Never share this token or commit it to git!")
    
    # Create .env.example
    env_example = Path(".env.example")
    with open(env_example, 'w') as f:
        f.write("# Main Node Configuration (DO NOT COMMIT ACTUAL VALUES!)\n")
        f.write("BLYAN_MAIN_NODE_SECRET=your-secret-here\n")
        f.write("BLYAN_MAIN_NODE_TOKEN=your-token-here\n")
        f.write("BLYAN_NODE_ID=your-node-id\n")
    
    print(f"\n📄 Created {env_example} for reference")

if __name__ == "__main__":
    setup_main_node()