import time
import tracemalloc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET
import math

from task1 import PetriNet
from task2 import bfs_reachable
from task3 import bdd_reachable

def generate_parallel_pnml(target_states, filename):
    a = int(target_states ** (1/3))
    if a < 2: a = 2
    remaining = target_states / a
    b = int(remaining ** 0.5)
    if b < 2: b = 2
    c = math.ceil(target_states / (a * b))
    
    dims = [a, b, c]
    real_states = a * b * c
    
    net = ET.Element('pnml')
    net_el = ET.SubElement(net, 'net', id="ParallelChainNet", type="http://www.pnml.org/version-2009/grammar/pnmlcoremodel")
    page = ET.SubElement(net_el, 'page', id="page0")

    for chain_idx, length in enumerate(dims):
        prefix = f"c{chain_idx}"
        for i in range(length):
            pid = f"{prefix}_p{i}"
            place = ET.SubElement(page, 'place', id=pid)
            ET.SubElement(ET.SubElement(place, 'name'), 'text').text = pid
            if i == 0:
                initial = ET.SubElement(place, 'initialMarking')
                ET.SubElement(initial, 'text').text = "1"

        for i in range(length - 1):
            tid = f"{prefix}_t{i}"
            trans = ET.SubElement(page, 'transition', id=tid)
            ET.SubElement(ET.SubElement(trans, 'name'), 'text').text = tid
            ET.SubElement(page, 'arc', id=f"{prefix}_a1_{i}", source=f"{prefix}_p{i}", target=tid)
            ET.SubElement(page, 'arc', id=f"{prefix}_a2_{i}", source=tid, target=f"{prefix}_p{i+1}")

    tree = ET.ElementTree(net)
    with open(filename, "wb") as f:
        tree.write(f)
    return real_states

def measure_performance(algorithm_func, pn, algo_type="Explicit"):
    tracemalloc.start()
    start_time = time.time()
    
    result = algorithm_func(pn)
    
    if algo_type == "Symbolic":
        count = result[1] if isinstance(result, tuple) else result
    else:
        count = len(result)
        
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return count, (end_time - start_time) * 1000, peak / (1024 * 1024)

def main_benchmark_parallel():
    targets = np.linspace(100, 50000, 20, dtype=int)
    targets = np.concatenate([[10, 50], targets])
    targets = np.unique(targets)

    results = {"States": [], "BFS_Time": [], "BFS_Mem": [], "BDD_Time": [], "BDD_Mem": []}

    print("States     | BFS (ms)   | BDD (ms)   | BFS (MB)   | BDD (MB)")

    for t in targets:
        filename = f"temp_para_{t}.pnml"
        real_states = generate_parallel_pnml(t, filename)
        
        try:
            pn = PetriNet.from_pnml(filename)
            cnt_bfs, t_bfs, m_bfs = measure_performance(bfs_reachable, pn, "Explicit")
            cnt_bdd, t_bdd, m_bdd = measure_performance(bdd_reachable, pn, "Symbolic")
            
            print(f"{real_states:<10} | {t_bfs:8.2f}   | {t_bdd:8.2f}   | {m_bfs:8.2f}   | {m_bdd:8.2f}")
            
            results["States"].append(real_states)
            results["BFS_Time"].append(t_bfs)
            results["BFS_Mem"].append(m_bfs)
            results["BDD_Time"].append(t_bdd)
            results["BDD_Mem"].append(m_bdd)
        except:
            pass
            
        if os.path.exists(filename): os.remove(filename)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_parallel.csv", index=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(df["States"], df["BFS_Time"], 'r-o', label='BFS')
    ax1.plot(df["States"], df["BDD_Time"], 'b-s', label='BDD')
    ax1.set_title('Time Complexity')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df["States"], df["BFS_Mem"], 'r--o', label='BFS Mem')
    ax2.plot(df["States"], df["BDD_Mem"], 'b--s', label='BDD Mem')
    ax2.set_title('Memory Usage')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("benchmark_parallel.png")
    plt.show()

def run_single_file_check():
    filename = input("Nhập tên file (Enter chọn 'input.pnml'): ").strip() or "input.pnml"

    pn = PetriNet.from_pnml(filename)
    print(f"Loaded! Places: {len(pn.place_ids)}, Transitions: {len(pn.trans_ids)}")

    print("-" * 60)
    print(f"{'Algorithm':<15} | {'States':<10} | {'Time (ms)':<10} | {'Mem (MB)':<10}")
    print("-" * 60)

    cnt_bfs, t_bfs, m_bfs = measure_performance(bfs_reachable, pn, "Explicit")
    print(f"{'BFS':<15} | {cnt_bfs:<10} | {t_bfs:10.2f} | {m_bfs:10.2f}")

    cnt_bdd, t_bdd, m_bdd = measure_performance(bdd_reachable, pn, "Symbolic")
    print(f"{'BDD':<15} | {cnt_bdd:<10} | {t_bdd:10.2f} | {m_bdd:10.2f}")
    print("-" * 60)

if __name__ == "__main__":
    print("1. Benchmark Parallel")
    print("2. Check Single File")
    choice = input("Chose (1/2): ").strip()
    
    if choice == '1':
        main_benchmark_parallel()
    elif choice == '2':
        run_single_file_check()