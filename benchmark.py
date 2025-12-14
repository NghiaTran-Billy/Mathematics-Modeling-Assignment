import sys
import time
import tracemalloc
from typing import Callable, Any
from typing import Set, Tuple


from task1 import PetriNet
from task2 import bfs_reachable, dfs_reachable
from task3 import bdd_reachable
from task4 import deadlock_bdd2, deadlock_iterative_ilp_bdd
from task5 import max_reachable_marking
import numpy as np


# -------------------------------------------------------------
# Helper function to measure time + memory
# -------------------------------------------------------------
def profile(func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
    tracemalloc.start()
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms = (end_time - start_time) * 1000
    peak_kb = peak / 1024

    return result, elapsed_ms, peak_kb


# -------------------------------------------------------------
# Benchmark Runner
# -------------------------------------------------------------
def run_benchmark(filename: str):
    print("========================================")
    print("======= BENCHMARK: REACHABILITY ========")
    print("========================================")
    print(f"Loading PNML: {filename}")

    pn = PetriNet.from_pnml(filename)
    print("Loaded Petri Net:")
    print(f"- Places: {len(pn.place_ids)}")
    print(f"- Transitions: {len(pn.trans_ids)}")
    print(f"- Initial marking: {pn.M0.tolist()}")
    print("----------------------------------------")

    # ---------------- BFS ----------------
    print("\n[1] BFS Reachability")
    (bfs_res, bfs_time, bfs_mem) = profile(bfs_reachable, pn)
    print(f"  Reachable markings: {len(bfs_res)}")
    print(f"  Time: {bfs_time:.2f} ms")
    print(f"  Peak Memory: {bfs_mem:.2f} KB")

    # ---------------- DFS ----------------
    print("\n[2] DFS Reachability")
    (dfs_res, dfs_time, dfs_mem) = profile(dfs_reachable, pn)
    print(f"  Reachable markings: {len(dfs_res)}")
    print(f"  Time: {dfs_time:.2f} ms")
    print(f"  Peak Memory: {dfs_mem:.2f} KB")

    # ---------------- BDD ----------------
    print("\n[3] BDD Reachability")
    ((bdd_node, bdd_count), bdd_time, bdd_mem) = profile(bdd_reachable, pn)
    print(f"  Reachable markings: {bdd_count}")
    print(f"  Time: {bdd_time:.2f} ms")
    print(f"  Peak Memory: {bdd_mem:.2f} KB")

    # # ---------------- Deadlock (ILP + BDD) ----------------
    print("\n[4] Deadlock Search (Iterative ILP + BDD)")
    (dead_mark, dead_msg), dead_time, dead_mem = profile(deadlock_iterative_ilp_bdd, pn,bdd_node,bdd_count)
    print(f"  Result: {dead_mark}")
    print(f"  Message: {dead_msg}")
    print(f"  Time: {dead_time:.2f} ms")
    print(f"  Peak Memory: {dead_mem:.2f} KB")

    # # ---------------- Deadlock (BDD2) ----------------
    print("\n[5] Deadlock Search (BDD-Only & ILP Filtering)")
    (dead2_mark, dead2_msg), dead2_time, dead2_mem = profile(deadlock_bdd2, pn, bdd_node,bdd_count)
    print(f"  Result: {dead2_mark}")
    print(f"  Message: {dead2_msg}")
    print(f"  Time: {dead2_time:.2f} ms")
    print(f"  Peak Memory: {dead2_mem:.2f} KB")

    # ---------------- Optimization (Branch-and-Bound) ----------------
    print("\n[6] Optimization (Max cᵀM with BDD)")
    c = np.ones(len(pn.place_ids))  # Example cost vector: all weights = 1
    (opt_mark, opt_val), opt_time, opt_mem = profile(max_reachable_marking, pn.place_ids, bdd_node, c)
    print(f"  Best marking: {opt_mark}")
    print(f"  Best value: {opt_val}")
    print(f"  Time: {opt_time:.2f} ms")
    print(f"  Peak Memory: {opt_mem:.2f} KB")

    
# -------------------------------------------------------------
# Command line interface
# -------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 benchmark.py <pnml-file>")
        sys.exit(1)

    filename = sys.argv[1]
    run_benchmark(filename)
