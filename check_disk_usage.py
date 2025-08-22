#!/usr/bin/env python3
"""
Check disk usage in the project directory and system to find what's taking up space.
"""

import os
import sys
from pathlib import Path
import shutil
import subprocess

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

def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return None

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
    print(f"DISK USAGE ANALYSIS")
    print("=" * 60)
    
    # Get system-wide disk usage using df
    print(f"\n📊 SYSTEM DISK USAGE (df -h):")
    df_output = run_command("df -h /")
    if df_output:
        print(df_output)
    
    # Get top 20 largest directories on system
    print(f"\n📂 TOP 20 LARGEST DIRECTORIES ON SYSTEM:")
    print("(This may take a moment...)")
    
    # Use du to find large directories
    du_cmd = "du -h / 2>/dev/null | sort -rh | head -20"
    du_output = run_command(du_cmd)
    if du_output:
        for line in du_output.split('\n'):
            print(f"  {line}")
    
    # Get total disk info for current directory
    total, used, free = shutil.disk_usage(root)
    print(f"\n📊 CURRENT DIRECTORY DISK INFO:")
    print(f"  Path: {root.absolute()}")
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
    
    # Check all possible cache locations
    print(f"\n🔍 CHECKING ALL COMMON CACHE LOCATIONS:")
    
    cache_locations = [
        Path.home() / ".cache",
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/workspace") if Path("/workspace").exists() else None,  # Common in containers
        Path("/root/.cache") if Path("/root/.cache").exists() else None,
        Path("/opt") if Path("/opt").exists() else None,
        root / ".cache",
        root / "data" / ".hf",
    ]
    
    for cache_dir in cache_locations:
        if cache_dir and cache_dir.exists():
            size = get_size(cache_dir)
            if size > 100 * 1024 * 1024:  # Only show if > 100MB
                print(f"  {str(cache_dir):40} {format_bytes(size):>10}")
                
                # Check subdirectories if it's a cache directory
                if "cache" in str(cache_dir).lower():
                    try:
                        subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]
                        for subdir in sorted(subdirs, key=lambda x: get_size(x), reverse=True)[:5]:
                            sub_size = get_size(subdir)
                            if sub_size > 100 * 1024 * 1024:
                                print(f"    └─ {subdir.name:36} {format_bytes(sub_size):>10}")
                    except:
                        pass
    
    # Specific HuggingFace cache check
    print(f"\n🤗 HUGGINGFACE SPECIFIC LOCATIONS:")
    hf_locations = [
        Path.home() / ".cache" / "huggingface",
        Path("/root/.cache/huggingface") if Path("/root/.cache/huggingface").exists() else None,
        Path("/workspace/.cache/huggingface") if Path("/workspace/.cache/huggingface").exists() else None,
        root / "data" / ".hf",
    ]
    
    total_hf = 0
    for hf_dir in hf_locations:
        if hf_dir and hf_dir.exists():
            size = get_size(hf_dir)
            total_hf += size
            print(f"  {str(hf_dir):50} {format_bytes(size):>10}")
            
            # Show what models are cached
            hub_dir = hf_dir / "hub"
            if hub_dir.exists():
                model_dirs = [d for d in hub_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
                for model_dir in model_dirs[:5]:
                    model_size = get_size(model_dir)
                    model_name = model_dir.name.replace("models--", "").replace("--", "/")
                    print(f"    └─ {model_name:46} {format_bytes(model_size):>10}")
    
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

def quick_check():
    """Quick check of major space consumers."""
    print("=" * 60)
    print("QUICK DISK CHECK")
    print("=" * 60)
    
    # System disk usage
    print("\n📊 DISK USAGE:")
    df_output = run_command("df -h / | tail -1")
    if df_output:
        parts = df_output.split()
        if len(parts) >= 5:
            print(f"  Total: {parts[1]}")
            print(f"  Used:  {parts[2]} ({parts[4]})")
            print(f"  Free:  {parts[3]}")
    
    # Find largest directories
    print("\n📁 LARGEST DIRECTORIES:")
    
    # Check common large directories
    dirs_to_check = [
        Path.home() / ".cache/huggingface",
        Path("/workspace") if Path("/workspace").exists() else None,
        Path("/root/.cache") if Path("/root/.cache").exists() else None,
        Path("./data"),
        Path("./.venv") if Path("./.venv").exists() else None,
    ]
    
    sizes = []
    for dir_path in dirs_to_check:
        if dir_path and dir_path.exists():
            size = get_size(dir_path)
            if size > 100 * 1024 * 1024:  # > 100MB
                sizes.append((str(dir_path), size))
    
    sizes.sort(key=lambda x: x[1], reverse=True)
    for path, size in sizes[:10]:
        print(f"  {path:50} {format_bytes(size):>10}")
    
    # Quick recommendation
    print("\n💡 TO FREE SPACE:")
    for path, size in sizes[:3]:
        if "huggingface" in path and size > 10 * 1024**3:
            print(f"  • Clear HuggingFace cache: rm -rf {path}/hub/models--*")
        elif "venv" in path and size > 5 * 1024**3:
            print(f"  • Reinstall virtual env: rm -rf {path} && python -m venv {path}")
        elif "/data" in path and size > 10 * 1024**3:
            print(f"  • Clear blockchain: rm -rf {path}/B/*")

if __name__ == "__main__":
    # Check for quick mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_check()
    else:
        # Check current directory or specified directory
        target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
        analyze_directory(target_dir)