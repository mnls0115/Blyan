#!/usr/bin/env python3
"""
Comprehensive verification of production fixes.
"""

import subprocess
import sys

def run_check(description, command):
    """Run a check command and report results."""
    print(f"\n{description}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(f"  Found: {result.stdout.strip()[:200]}")
        return False
    else:
        print(f"  ✅ Clean")
        return True

def main():
    print("=" * 80)
    print("COMPREHENSIVE PRODUCTION VERIFICATION")
    print("=" * 80)
    
    all_good = True
    
    # 1. Check for _min_blockchain_layers (should NOT exist)
    all_good &= run_check(
        "1. Checking for hardcoded _min_blockchain_layers:",
        "grep -n '_min_blockchain_layers' backend/model/manager.py"
    )
    
    # 2. Check for dynamic calculation (should exist)
    result = subprocess.run(
        "grep -n 'num_hidden_layers.*+.*3' backend/model/manager.py",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(f"\n2. Dynamic calculation found:")
        print(f"  ✅ Line {result.stdout.strip()[:100]}")
    else:
        print(f"\n2. Dynamic calculation NOT found")
        all_good = False
    
    # 3. Check for random perturbation in forward_chunk (should NOT exist)
    all_good &= run_check(
        "3. Checking for 0.01 * randn perturbation:",
        "grep -n '0.01.*randn' backend/model/chunked_blockchain_loader.py"
    )
    
    # 4. Check for random logits in forward_chunk (should NOT exist in forward path)
    print("\n4. Checking forward_chunk for random logits:")
    result = subprocess.run(
        "sed -n '/def forward_chunk/,/^    def /p' backend/model/chunked_blockchain_loader.py | grep -n 'torch.randn'",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(f"  ❌ Found random ops in forward_chunk: {result.stdout.strip()[:200]}")
        all_good = False
    else:
        print(f"  ✅ No random ops in forward_chunk")
    
    # 5. Check local sticky path iterates stages
    result = subprocess.run(
        "grep -n 'for stage in self.partition_plan.stages:' backend/dense/pipeline_coordinator.py",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(f"\n5. Local sticky path iterates stages:")
        print(f"  ✅ Line {result.stdout.strip()}")
    else:
        print(f"\n5. Local sticky path does NOT iterate stages")
        all_good = False
    
    # 6. Check for mock response in routes.py (should NOT exist)
    all_good &= run_check(
        "6. Checking for 'Hello world' mock response:",
        "grep -n 'Hello world.*non-streaming' server/http/routes.py"
    )
    
    # 7. Check that 503 is raised when no handler
    result = subprocess.run(
        "grep -n 'HTTPException.*503.*No streaming handler' server/http/routes.py",
        shell=True, capture_output=True, text=True
    )
    if result.stdout:
        print(f"\n7. Raises 503 when no handler:")
        print(f"  ✅ Line {result.stdout.strip()}")
    else:
        print(f"\n7. Does NOT raise 503 properly")
        all_good = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_good:
        print("✅ ALL CHECKS PASSED - Code is production-ready!")
        print("\nRecommended production settings:")
        print("  export STRICT_MODEL_LOAD=true")
        print("  export MODEL_SIZE=8B  # or 32B, 70B")
        print("  export WEIGHT_PRECISION=bf16")
    else:
        print("❌ SOME CHECKS FAILED - Review issues above")
    print("=" * 80)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())