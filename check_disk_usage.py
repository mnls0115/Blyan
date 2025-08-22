#!/usr/bin/env python3
"""
Check disk usage in the project directory to find what's taking up space.
"""

import os
import sys
from pathlib import Path
import shutil

def get_size(path):
    """Get size of file or directory in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except:
                pass
    return total

def format_bytes(size):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def analyze_directory(root_path="."):
    """Analyze disk usage in the given directory."""
    root = Path(root_path)
    
    print("=" * 60)
    print(f"DISK USAGE ANALYSIS: {root.absolute()}")
    print("=" * 60)
    
    # Get total disk info
    total, used, free = shutil.disk_usage(root)
    print(f"\n📊 DISK OVERVIEW:")
    print(f"  Total: {format_bytes(total)}")
    print(f"  Used:  {format_bytes(used)} ({used*100/total:.1f}%)")
    print(f"  Free:  {format_bytes(free)} ({free*100/total:.1f}%)")
    
    # Analyze main directories
    dirs_to_check = {}
    
    # Check all subdirectories in root
    for item in root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            size = get_size(item)
            dirs_to_check[item.name] = size
    
    # Also check hidden directories that might be large
    hidden_dirs = ['.venv', '.cache', '.hf', '.git']
    for hidden in hidden_dirs:
        hidden_path = root / hidden
        if hidden_path.exists():
            size = get_size(hidden_path)
            dirs_to_check[hidden] = size
    
    # Sort by size
    sorted_dirs = sorted(dirs_to_check.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📁 TOP DIRECTORIES BY SIZE:")
    for dir_name, size in sorted_dirs[:10]:
        print(f"  {dir_name:20} {format_bytes(size):>10}")
    
    # Specific analysis for data directory (blockchain)
    data_dir = root / "data"
    if data_dir.exists():
        print(f"\n⛓️  BLOCKCHAIN DATA BREAKDOWN:")
        for chain in ['A', 'B', 'D']:
            chain_dir = data_dir / chain
            if chain_dir.exists():
                size = get_size(chain_dir)
                num_files = len(list(chain_dir.glob("*.json")))
                print(f"  Chain {chain}: {format_bytes(size):>10} ({num_files} blocks)")
    
    # Check for large model files
    print(f"\n🤖 LARGE FILES (>100MB):")
    large_files = []
    for ext in ['*.bin', '*.safetensors', '*.pth', '*.pt', '*.ckpt', '*.h5']:
        for file in root.rglob(ext):
            size = get_size(file)
            if size > 100 * 1024 * 1024:  # 100MB
                large_files.append((file.relative_to(root), size))
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    for file_path, size in large_files[:10]:
        print(f"  {str(file_path):50} {format_bytes(size):>10}")
    
    # Check HuggingFace cache
    hf_cache_dirs = [
        root / ".cache" / "huggingface",
        root / "data" / ".hf",
        Path.home() / ".cache" / "huggingface"
    ]
    
    print(f"\n🤗 HUGGINGFACE CACHE:")
    total_hf = 0
    for hf_dir in hf_cache_dirs:
        if hf_dir.exists():
            size = get_size(hf_dir)
            total_hf += size
            print(f"  {str(hf_dir):50} {format_bytes(size):>10}")
    
    if total_hf > 0:
        print(f"  {'TOTAL HF CACHE:':50} {format_bytes(total_hf):>10}")
    
    # Check for temporary files
    print(f"\n🗑️  TEMPORARY FILES:")
    temp_patterns = ['*.tmp', '*.temp', '*.log', '*.bak', '*.swp', '__pycache__']
    temp_size = 0
    temp_count = 0
    for pattern in temp_patterns:
        for file in root.rglob(pattern):
            if file.is_file():
                temp_size += get_size(file)
                temp_count += 1
            elif file.is_dir():  # for __pycache__
                temp_size += get_size(file)
                temp_count += 1
    
    print(f"  Found {temp_count} temporary files/dirs: {format_bytes(temp_size)}")
    
    # Recommendations
    print(f"\n💡 CLEANUP RECOMMENDATIONS:")
    
    if total_hf > 5 * 1024**3:  # More than 5GB
        print(f"  • HuggingFace cache is large ({format_bytes(total_hf)})")
        print(f"    Run: huggingface-cli delete-cache")
    
    if data_dir.exists():
        b_chain = data_dir / "B"
        if b_chain.exists():
            b_size = get_size(b_chain)
            if b_size > 10 * 1024**3:  # More than 10GB
                print(f"  • Chain B is large ({format_bytes(b_size)})")
                print(f"    Consider: rm -rf data/B/* (will delete blockchain)")
    
    if temp_size > 1024**3:  # More than 1GB
        print(f"  • Clean temporary files ({format_bytes(temp_size)})")
        print(f"    Run: find . -name '*.tmp' -delete")
        print(f"    Run: find . -type d -name '__pycache__' -exec rm -rf {{}} +")
    
    venv_dir = root / ".venv"
    if venv_dir.exists():
        venv_size = get_size(venv_dir)
        if venv_size > 5 * 1024**3:  # More than 5GB
            print(f"  • Virtual environment is large ({format_bytes(venv_size)})")
            print(f"    Consider reinstalling packages")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Check current directory or specified directory
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_directory(target_dir)