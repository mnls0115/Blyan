#!/usr/bin/env python3
"""
Check blockchain health and diagnose issues.
"""

import sys
import json
import shutil
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_disk_space():
    """Check available disk space."""
    total, used, free = shutil.disk_usage("/")
    
    # Convert to GB
    total_gb = total // (2**30)
    used_gb = used // (2**30)
    free_gb = free // (2**30)
    
    print(f"Disk Space:")
    print(f"  Total: {total_gb} GB")
    print(f"  Used: {used_gb} GB")
    print(f"  Free: {free_gb} GB")
    
    if free_gb < 10:
        print("  ⚠️ WARNING: Low disk space! Less than 10GB free")
    elif free_gb < 50:
        print("  ⚠️ CAUTION: Only {free_gb}GB free")
    else:
        print(f"  ✅ Sufficient space available")
    
    return free_gb

def check_blockchain_files():
    """Check blockchain file integrity."""
    data_dir = Path("./data")
    
    if not data_dir.exists():
        print("❌ Data directory does not exist")
        return
    
    for chain_id in ['A', 'B', 'D']:
        chain_dir = data_dir / chain_id
        print(f"\nChain {chain_id}:")
        
        if not chain_dir.exists():
            print(f"  Directory does not exist")
            continue
        
        json_files = list(chain_dir.glob("*.json"))
        print(f"  Files: {len(json_files)}")
        
        if json_files:
            # Check for empty files
            empty_files = []
            corrupted_files = []
            total_size = 0
            
            for file in json_files:
                size = file.stat().st_size
                total_size += size
                
                if size == 0:
                    empty_files.append(file.name)
                else:
                    # Try to parse JSON
                    try:
                        with file.open() as f:
                            content = f.read()
                            if content.strip() == "":
                                empty_files.append(file.name)
                            else:
                                json.loads(content)
                    except json.JSONDecodeError:
                        corrupted_files.append(file.name)
            
            print(f"  Total size: {total_size / (1024*1024):.1f} MB")
            
            if empty_files:
                print(f"  ❌ Empty files: {len(empty_files)}")
                for f in empty_files[:5]:
                    print(f"    - {f}")
                if len(empty_files) > 5:
                    print(f"    ... and {len(empty_files)-5} more")
            
            if corrupted_files:
                print(f"  ❌ Corrupted files: {len(corrupted_files)}")
                for f in corrupted_files[:5]:
                    print(f"    - {f}")
                if len(corrupted_files) > 5:
                    print(f"    ... and {len(corrupted_files)-5} more")
            
            if not empty_files and not corrupted_files:
                print(f"  ✅ All files appear valid")
            
            # Check sequence
            if json_files:
                indices = []
                for f in json_files:
                    try:
                        idx = int(f.stem)
                        indices.append(idx)
                    except ValueError:
                        pass
                
                if indices:
                    indices.sort()
                    missing = []
                    for i in range(indices[0], indices[-1]):
                        if i not in indices:
                            missing.append(i)
                    
                    if missing:
                        print(f"  ⚠️ Missing indices: {missing[:10]}")
                        if len(missing) > 10:
                            print(f"    ... and {len(missing)-10} more")

def check_last_blocks():
    """Check the last few blocks in detail."""
    from backend.core.chain import Chain
    
    print("\nLast Blocks Analysis:")
    
    for chain_id in ['A', 'B']:
        try:
            chain = Chain(Path("./data"), chain_id, skip_pol=True)
            blocks = chain.get_all_blocks()
            
            print(f"\nChain {chain_id}:")
            print(f"  Total blocks: {len(blocks)}")
            
            if blocks:
                # Check last 5 blocks
                for block in blocks[-5:]:
                    print(f"  Block {block.header.index}:")
                    print(f"    Type: {block.header.block_type}")
                    print(f"    Payload size: {len(block.payload)} bytes")
                    if hasattr(block.header, 'expert_name'):
                        print(f"    Expert: {block.header.expert_name}")
        except Exception as e:
            print(f"  ERROR: Cannot load chain {chain_id}: {e}")

if __name__ == "__main__":
    print("Blockchain Health Check")
    print("=" * 60)
    
    # Check disk space
    free_gb = check_disk_space()
    
    # Check blockchain files
    check_blockchain_files()
    
    # Check last blocks
    check_last_blocks()
    
    print("\n" + "=" * 60)
    
    if free_gb < 10:
        print("🔴 CRITICAL: Insufficient disk space!")
        print("   The upload is likely failing due to disk space")
        print("   Free up space or use a larger disk")
    else:
        print("💡 If seeing 'char 0' errors:")
        print("   1. Check if disk write permissions are OK")
        print("   2. Try restarting the upload")
        print("   3. Consider clearing ./data/B/ and starting fresh")