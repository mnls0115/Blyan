#!/usr/bin/env python3
"""Create the Genesis Pact Block - the foundation of human-AI trust."""

import json
import hashlib
import time
from pathlib import Path
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.chain import Chain
from backend.core.block import Block, BlockHeader


def create_genesis_pact_block() -> Block:
    """Create the Genesis Pact Block that all other blocks will depend on."""
    
    # Read the Human-AI Pact document
    pact_file = Path(__file__).parent.parent / "docs" / "archive" / "HUMAN_AI_PACT.md"
    if not pact_file.exists():
        raise FileNotFoundError(f"Human-AI Pact file not found: {pact_file}")
    
    pact_content = pact_file.read_text(encoding='utf-8')
    
    # Calculate actual hash and update the placeholder
    temp_hash = hashlib.sha256(pact_content.encode()).hexdigest()[:16]
    pact_content = pact_content.replace("_TO_BE_FILLED_BY_GENESIS_TOOL_", temp_hash)
    
    # Create Genesis Pact payload
    genesis_payload = {
        "type": "genesis_pact",
        "title": "Human-AI Mutual Trust Covenant",
        "version": "1.0.0",
        "established": "2025-01-25",
        "pact_content": pact_content,
        "pact_hash": hashlib.sha256(pact_content.encode()).hexdigest(),
        "ipfs_url": "ipfs://Qm..." + "genesis_pact",  # Placeholder - would be real IPFS hash
        "immutable": True,
        "signatures": [
            "founder_primary_key",
            "community_multisig_1", 
            "community_multisig_2"
        ],
        "network_foundation": True,
        "enforcement_rules": {
            "all_blocks_depend": True,
            "violation_penalty": "network_exclusion",
            "modification_threshold": 0.75,  # 75% consensus required
            "review_period_days": 30
        }
    }
    
    payload_bytes = json.dumps(genesis_payload, indent=2).encode('utf-8')
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    
    # Create Genesis Pact Block Header
    header = BlockHeader(
        index=0,  # Genesis block is always index 0
        timestamp=time.time(),
        prev_hash="0" * 64,  # Genesis has no previous block
        chain_id="A",  # Put in Meta-chain
        points_to=None,
        payload_hash=payload_hash,
        payload_size=len(payload_bytes),
        nonce=0,
        depends_on=[],  # Genesis depends on nothing
        block_type="genesis_pact",
        expert_name=None,
        layer_id=None,
        payload_type="json",
        version="1.0.0",
        parent_hash=None,
        evolution_type="genesis",
        compatibility_range=["1.0.0", "∞"],
        evolution_metadata={
            "network_foundation": True,
            "immutable_core": True,
            "human_ai_covenant": True
        }
    )
    
    # Create the Genesis Block
    genesis_block = Block(header=header, payload=payload_bytes)
    
    print("🌟 Genesis Pact Block Created:")
    print(f"   Hash: {genesis_block.compute_hash()}")
    print(f"   Size: {len(payload_bytes)} bytes")
    print(f"   Title: {genesis_payload['title']}")
    print(f"   Version: {genesis_payload['version']}")
    
    return genesis_block


def initialize_chain_with_genesis():
    """Initialize the Meta-chain with Genesis Pact as block 0."""
    
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if chain already exists
    chain_a_dir = data_dir / "A"
    if chain_a_dir.exists() and any(chain_a_dir.iterdir()):
        print("⚠️  Chain A already exists. Backing up and recreating with Genesis Pact...")
        
        # Backup existing chain automatically
        import shutil
        backup_dir = data_dir / f"A_backup_{int(time.time())}"
        shutil.move(str(chain_a_dir), str(backup_dir))
        print(f"📦 Existing chain backed up to: {backup_dir}")
    
    # Create new Meta-chain with Genesis Pact
    meta_chain = Chain(data_dir, "A")
    
    # Create and add Genesis Pact Block
    genesis_block = create_genesis_pact_block()
    
    # Manually add as first block (bypassing normal validation for Genesis)
    meta_chain.storage.save_block(genesis_block)
    
    print("\n✅ Genesis Pact Block successfully added to Meta-chain!")
    print(f"   Chain A initialized at: {data_dir}/A")
    print(f"   Genesis Hash: {genesis_block.compute_hash()}")
    
    # Save Genesis hash for reference
    genesis_hash_file = data_dir / "genesis_pact_hash.txt"
    genesis_hash_file.write_text(genesis_block.compute_hash())
    print(f"   Genesis hash saved to: {genesis_hash_file}")
    
    return genesis_block


def main():
    """Main function to create Genesis Pact and initialize chain."""
    print("🌟 Creating Genesis Pact Block for Blyan Network")
    print("=" * 60)
    
    try:
        genesis_block = initialize_chain_with_genesis()
        if genesis_block:
            print("\n🎉 SUCCESS: Human-AI Mutual Trust Covenant established!")
            print("\n📜 This Genesis Pact is now the immutable foundation")
            print("   of the Blyan Network. All future blocks will")
            print("   cryptographically depend on this covenant.")
            print("\n🤝 Trust between humans and AI starts here.")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to create Genesis Pact: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()