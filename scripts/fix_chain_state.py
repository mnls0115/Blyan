#!/usr/bin/env python3
"""
Fix chain state issues by checking for duplicates and rebuilding indices.
Run this on the GPU node to fix validation errors.
"""
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_chain(data_dir: Path, chain_id: str):
    """Analyze a chain for issues."""
    chain_dir = data_dir / chain_id
    if not chain_dir.exists():
        logger.info(f"Chain {chain_id} does not exist")
        return
    
    blocks = []
    for block_file in sorted(chain_dir.glob("*.json")):
        try:
            with open(block_file, 'r') as f:
                block_data = json.load(f)
                if isinstance(block_data, list):
                    blocks.extend(block_data)
                else:
                    blocks.append(block_data)
        except Exception as e:
            logger.error(f"Error reading {block_file}: {e}")
    
    logger.info(f"\nChain {chain_id} Analysis:")
    logger.info(f"  Total blocks: {len(blocks)}")
    
    # Check for duplicate indices
    indices = {}
    for block in blocks:
        idx = block['header']['index']
        if idx in indices:
            logger.warning(f"  ⚠️ Duplicate index {idx} found!")
        indices[idx] = block
    
    # Check for duplicate hashes
    hashes = {}
    for block in blocks:
        if 'hash' in block:
            h = block['hash']
            if h in hashes:
                logger.warning(f"  ⚠️ Duplicate hash {h[:8]}... found!")
            hashes[h] = block
    
    # Check block types
    block_types = {}
    for block in blocks:
        bt = block['header'].get('block_type', 'unknown')
        block_types[bt] = block_types.get(bt, 0) + 1
    
    logger.info("  Block types:")
    for bt, count in block_types.items():
        logger.info(f"    {bt}: {count}")
    
    # Check layer blocks
    layer_blocks = {}
    for block in blocks:
        if block['header'].get('block_type') == 'dense_layer':
            layer_name = block['header'].get('layer_name', '')
            if layer_name in layer_blocks:
                logger.warning(f"  ⚠️ Duplicate layer {layer_name} found!")
            layer_blocks[layer_name] = block
    
    if layer_blocks:
        logger.info(f"  Layer blocks found: {list(layer_blocks.keys())}")
    
    return blocks, indices, hashes, layer_blocks

def fix_chain(data_dir: Path, chain_id: str):
    """Fix chain issues by removing duplicates and rebuilding."""
    blocks, indices, hashes, layer_blocks = analyze_chain(data_dir, chain_id)
    
    if not blocks:
        return
    
    # Check if we need to fix anything
    has_duplicates = False
    
    # Find duplicate indices
    seen_indices = set()
    for block in blocks:
        idx = block['header']['index']
        if idx in seen_indices:
            has_duplicates = True
            break
        seen_indices.add(idx)
    
    if has_duplicates:
        logger.info(f"\n🔧 Fixing chain {chain_id}...")
        
        # Keep only the first occurrence of each index
        clean_blocks = []
        seen = set()
        for block in blocks:
            idx = block['header']['index']
            if idx not in seen:
                clean_blocks.append(block)
                seen.add(idx)
        
        # Sort by index
        clean_blocks.sort(key=lambda b: b['header']['index'])
        
        # Backup old chain
        chain_dir = data_dir / chain_id
        backup_dir = data_dir / f"{chain_id}_backup"
        if chain_dir.exists():
            import shutil
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(chain_dir, backup_dir)
            logger.info(f"  Backed up to {backup_dir}")
        
        # Write clean chain
        chain_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove old files
        for old_file in chain_dir.glob("*.json"):
            old_file.unlink()
        
        # Write new chain file
        output_file = chain_dir / "00000000.json"
        with open(output_file, 'w') as f:
            json.dump(clean_blocks, f, indent=2)
        
        logger.info(f"  ✅ Fixed chain {chain_id}: {len(blocks)} -> {len(clean_blocks)} blocks")
    else:
        logger.info(f"  ✅ Chain {chain_id} is clean")

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        data_dir = Path("./data")
    
    logger.info(f"🔍 Analyzing chains in {data_dir}")
    
    # Analyze all chains
    for chain_id in ['A', 'B', 'D']:
        analyze_chain(data_dir, chain_id)
    
    # Fix if needed
    response = input("\nDo you want to fix chain issues? (y/n): ")
    if response.lower() == 'y':
        for chain_id in ['A', 'B', 'D']:
            fix_chain(data_dir, chain_id)
        logger.info("\n✅ Chain fixes complete. Please restart your node.")
    else:
        logger.info("No changes made.")

if __name__ == "__main__":
    main()