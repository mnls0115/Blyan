#!/usr/bin/env python3
"""
Baseline benchmark harness for blockchain storage performance.
Measures boot time, iteration speed, and single-block fetch latency.
"""

import json
import time
import sys
import gc
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.chain import Chain
from backend.core.storage import BlockStorage


class BaselineBenchmark:
    """Measure baseline performance of current blockchain storage."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(data_dir),
            "metrics": {}
        }
        
    def measure_cold_boot(self) -> float:
        """Measure cold boot time (chain initialization to ready)."""
        print("Measuring cold boot time...")
        
        # Clear any caches
        gc.collect()
        gc.disable()
        
        start_ns = time.perf_counter_ns()
        
        # Initialize chain (this builds indexes)
        chain = Chain(self.data_dir, "B", skip_pol=True)
        blocks = chain.get_all_blocks()
        block_count = len(blocks)
        
        end_ns = time.perf_counter_ns()
        gc.enable()
        
        boot_time_ms = (end_ns - start_ns) / 1_000_000
        
        self.results["metrics"]["cold_boot_ms"] = boot_time_ms
        self.results["metrics"]["total_blocks"] = block_count
        
        print(f"  Cold boot: {boot_time_ms:.1f} ms for {block_count} blocks")
        return boot_time_ms
    
    def measure_iteration(self, n_blocks: int = 1000) -> float:
        """Measure time to iterate N blocks."""
        print(f"Measuring iteration of {n_blocks} blocks...")
        
        storage = BlockStorage(self.data_dir / "B")
        
        # Count available blocks
        all_paths = list(storage.dir_path.glob("*.json"))
        numeric_paths = []
        for p in all_paths:
            try:
                int(p.stem)
                numeric_paths.append(p)
            except ValueError:
                continue
        
        available = len(numeric_paths)
        actual_n = min(n_blocks, available)
        
        if actual_n == 0:
            print("  No blocks available to iterate")
            self.results["metrics"]["iteration_ms"] = 0
            self.results["metrics"]["iteration_blocks"] = 0
            return 0
        
        gc.collect()
        gc.disable()
        
        start_ns = time.perf_counter_ns()
        
        # Iterate blocks
        count = 0
        for block in storage.iter_blocks():
            count += 1
            if count >= actual_n:
                break
        
        end_ns = time.perf_counter_ns()
        gc.enable()
        
        iteration_ms = (end_ns - start_ns) / 1_000_000
        blocks_per_sec = (actual_n / iteration_ms) * 1000 if iteration_ms > 0 else 0
        
        self.results["metrics"]["iteration_ms"] = iteration_ms
        self.results["metrics"]["iteration_blocks"] = actual_n
        self.results["metrics"]["blocks_per_sec"] = blocks_per_sec
        
        print(f"  Iteration: {iteration_ms:.1f} ms for {actual_n} blocks ({blocks_per_sec:.0f} blocks/sec)")
        return iteration_ms
    
    def measure_single_fetch(self, trials: int = 100) -> Dict[str, float]:
        """Measure single-block fetch latency (p50/p95)."""
        print(f"Measuring single-block fetch latency ({trials} trials)...")
        
        storage = BlockStorage(self.data_dir / "B")
        
        # Get valid block indices
        indices = []
        for p in storage.dir_path.glob("*.json"):
            try:
                idx = int(p.stem)
                indices.append(idx)
            except ValueError:
                continue
        
        if not indices:
            print("  No blocks available for fetch test")
            self.results["metrics"]["fetch_p50_ms"] = 0
            self.results["metrics"]["fetch_p95_ms"] = 0
            return {"p50": 0, "p95": 0}
        
        # Run trials
        latencies_ns = []
        actual_trials = min(trials, len(indices))
        
        gc.collect()
        gc.disable()
        
        for i in range(actual_trials):
            idx = indices[i % len(indices)]
            
            start_ns = time.perf_counter_ns()
            block = storage.load_block(idx)
            end_ns = time.perf_counter_ns()
            
            if block:
                latencies_ns.append(end_ns - start_ns)
        
        gc.enable()
        
        if not latencies_ns:
            print("  No successful fetches")
            self.results["metrics"]["fetch_p50_ms"] = 0
            self.results["metrics"]["fetch_p95_ms"] = 0
            return {"p50": 0, "p95": 0}
        
        # Calculate percentiles
        latencies_ms = [ns / 1_000_000 for ns in latencies_ns]
        latencies_ms.sort()
        
        p50 = statistics.median(latencies_ms)
        p95_idx = int(len(latencies_ms) * 0.95)
        p95 = latencies_ms[p95_idx] if p95_idx < len(latencies_ms) else latencies_ms[-1]
        
        self.results["metrics"]["fetch_p50_ms"] = p50
        self.results["metrics"]["fetch_p95_ms"] = p95
        self.results["metrics"]["fetch_trials"] = len(latencies_ms)
        
        print(f"  Fetch latency: p50={p50:.2f} ms, p95={p95:.2f} ms")
        return {"p50": p50, "p95": p95}
    
    def save_results(self, output_dir: Path):
        """Save benchmark results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"baseline_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        return output_file


def main():
    """Run baseline benchmarks."""
    
    # Parse arguments
    if len(sys.argv) > 1:
        chain_dir = Path(sys.argv[1])
    else:
        chain_dir = Path("./data")
    
    print(f"Running baseline benchmark on: {chain_dir}")
    print("=" * 60)
    
    # Run benchmarks
    bench = BaselineBenchmark(chain_dir)
    
    # Test on chain B (parameter chain with most blocks)
    if (chain_dir / "B").exists():
        bench.data_dir = chain_dir
        bench.measure_cold_boot()
        bench.measure_iteration(1000)
        bench.measure_single_fetch(100)
    else:
        print(f"Warning: Chain B not found at {chain_dir / 'B'}")
        print("Creating empty results...")
    
    # Save results
    output_dir = Path("benchmarks")
    bench.save_results(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    metrics = bench.results["metrics"]
    print(f"Total blocks:      {metrics.get('total_blocks', 0)}")
    print(f"Cold boot time:    {metrics.get('cold_boot_ms', 0):.1f} ms")
    print(f"Iteration speed:   {metrics.get('blocks_per_sec', 0):.0f} blocks/sec")
    print(f"Fetch p50:         {metrics.get('fetch_p50_ms', 0):.2f} ms")
    print(f"Fetch p95:         {metrics.get('fetch_p95_ms', 0):.2f} ms")


if __name__ == "__main__":
    main()