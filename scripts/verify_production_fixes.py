#!/usr/bin/env python3
"""
Verify that all production fixes are actually in place.
"""

import os
import re
from pathlib import Path

def check_file_for_pattern(filepath, pattern, should_exist=False):
    """Check if pattern exists in file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            matches = re.findall(pattern, content)
            found = len(matches) > 0
            return found == should_exist, matches
    except Exception as e:
        return False, [str(e)]

def main():
    print("=" * 80)
    print("Verifying Production Fixes")
    print("=" * 80)
    
    issues = []
    
    # 1. Check manager.py for dynamic blockchain readiness
    print("\n1. Checking backend/model/manager.py...")
    
    # Should NOT have old hardcoded variable
    ok, matches = check_file_for_pattern(
        "backend/model/manager.py",
        r"_min_blockchain_layers\s*=\s*int\(os\.getenv",
        should_exist=False
    )
    if ok:
        print("   ✅ No hardcoded _min_blockchain_layers")
    else:
        print(f"   ❌ Found hardcoded _min_blockchain_layers: {matches}")
        issues.append("manager.py still has hardcoded _min_blockchain_layers")
    
    # Should have dynamic calculation
    ok, matches = check_file_for_pattern(
        "backend/model/manager.py",
        r"LAYERS\[\"num_hidden_layers\"\]\s*\+\s*3",
        should_exist=True
    )
    if ok:
        print("   ✅ Has dynamic calculation (num_hidden_layers + 3)")
    else:
        print("   ❌ Missing dynamic calculation")
        issues.append("manager.py missing dynamic layer calculation")
    
    # 2. Check chunked_blockchain_loader.py for mock transforms
    print("\n2. Checking backend/model/chunked_blockchain_loader.py...")
    
    # Should NOT have mock transforms
    ok, matches = check_file_for_pattern(
        "backend/model/chunked_blockchain_loader.py",
        r"0\.01\s*\*\s*torch\.randn",
        should_exist=False
    )
    if ok:
        print("   ✅ No mock random transforms")
    else:
        print(f"   ❌ Found mock transforms: {matches}")
        issues.append("chunked_blockchain_loader.py still has mock transforms")
    
    # Should NOT have random logits
    ok, matches = check_file_for_pattern(
        "backend/model/chunked_blockchain_loader.py",
        r"torch\.randn\([^)]*vocab_size",
        should_exist=False
    )
    if ok:
        print("   ✅ No random logits generation")
    else:
        print(f"   ❌ Found random logits: {matches}")
        issues.append("chunked_blockchain_loader.py still generates random logits")
    
    # 3. Check pipeline_coordinator.py for local sticky fix
    print("\n3. Checking backend/dense/pipeline_coordinator.py...")
    
    # Should iterate through stages
    ok, matches = check_file_for_pattern(
        "backend/dense/pipeline_coordinator.py",
        r"for stage in self\.partition_plan\.stages:",
        should_exist=True
    )
    if ok:
        print("   ✅ Local path iterates through all stages")
    else:
        print("   ❌ Local path does not iterate stages")
        issues.append("pipeline_coordinator.py local path doesn't chain stages")
    
    # 4. Check routes.py for mock responses
    print("\n4. Checking server/http/routes.py...")
    
    # Should NOT have mock response text
    ok, matches = check_file_for_pattern(
        "server/http/routes.py",
        r"\"Hello world! This is a non-streaming response\.\"",
        should_exist=False
    )
    if ok:
        print("   ✅ No mock response text")
    else:
        print(f"   ❌ Found mock response: {matches}")
        issues.append("routes.py still has mock responses")
    
    # Should always raise 503 when no handler
    ok, matches = check_file_for_pattern(
        "server/http/routes.py",
        r"raise HTTPException\(status_code=503.*No streaming handler",
        should_exist=True
    )
    if ok:
        print("   ✅ Raises 503 when no handler")
    else:
        print("   ❌ Does not properly raise 503")
        issues.append("routes.py doesn't always return 503 when no handler")
    
    # 5. Summary
    print("\n" + "=" * 80)
    if not issues:
        print("✅ All production fixes are in place!")
    else:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   - {issue}")
    print("=" * 80)
    
    return 0 if not issues else 1

if __name__ == "__main__":
    exit(main())