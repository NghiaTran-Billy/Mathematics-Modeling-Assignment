# Symbolic and Algebraic Reasoning in Petri Nets
**Course:** Mathematical Modeling (CO2011) - Semester 1 (2025-2026)  
**University:** HCMC University of Technology (HCMUT)  
**Faculty:** Computer Science and Engineering  

## 1. Group Information
 
**Members | Group ID:**
1. **Hoàng Minh Hải**, 2410898
2. **Trần Trung Nghĩa**, 2412278
3. **Phạm Minh Trung**, 2413712
4. **Phạm Minh Hiếu**, 2411008
5. **Lê Bảo Nghiêm**, 2412252

---

## 2. Project Description
This project implements a software tool for analyzing **1-safe Petri Nets** using a combination of explicit traversal algorithms, Symbolic Model Checking (using Binary Decision Diagrams - BDD), and Integer Linear Programming (ILP).

The software is designed to fulfill the 5 key tasks of the assignment :
1.  **Parsing:** Reading Petri Nets from PNML files.
2.  **Explicit Reachability:** Computing reachable markings using BFS/DFS.
3.  **Symbolic Reachability:** Computing the reachability set efficiently using BDDs.
4.  **Deadlock Detection:** Detecting deadlocks using a hybrid ILP and BDD approach.
5.  **Optimization:** Finding a reachable marking that maximizes a cost function.

---

## 3. Installation & Requirements

### System Requirements
* **Python 3.8+**

### Dependencies
The project relies on the following Python packages:
* `numpy`: For matrix operations (Incidence matrices).
* `dd`: For BDD (Binary Decision Diagram) manipulation (pure Python implementation).
* `pulp`: For Integer Linear Programming (ILP) solving.

### Installation
Run the following command to install the necessary dependencies:

```bash
pip install numpy dd pulp
```
## 4. Project Structure

The source code is organized into modular files corresponding to the assignment tasks:

| File | Role | Description |
| :--- | :--- | :--- |
| `benchmark.py` | **Main Entry Point** | Runs all tasks sequentially (BFS, DFS, BDD, Deadlock, Opt) and reports time/memory usage. |
| `PetriNetReading.py` | **Task 1** | Parses `.pnml` files and builds the Petri Net structure ($P, T, I, O, M_0$). |
| `ExplicitComputation.py` | **Task 2** | Implements Explicit Reachability (BFS and DFS algorithms). |
| `SymbolicComputation.py` | **Task 3** | Implements Symbolic Reachability using `dd`. Uses **Transition Chaining** for efficiency. |
| `DeadlockDetecting.py` | **Task 4** | Implements Deadlock Detection (Iterative ILP & BDD Filtering). |
| `Optimization.py` | **Task 5** | Implements **Branch-and-Bound** optimization over BDD nodes. |
## 5. Usage

The project is designed to be run via the `benchmark.py` script, which takes a PNML file path as an argument. It executes all tasks sequentially and prints the results/metrics to the console.

### Syntax
```bash
python benchmark.py <path_to_pnml_file>
```
### Output Explanation
The script generates a detailed report in the console:

1.  **Basic Info:** Displays the number of places, transitions, and the initial marking vector.
2.  **BFS/DFS Results:** Reports the total number of reachable markings found explicitly, along with execution time (ms) and peak memory usage (KB).
3.  **BDD Reachability:** Shows the size of the symbolic set (number of markings), execution time, and memory. This is typically faster for large nets.
4.  **Deadlock Search:**
    * *Iterative ILP+BDD:* Attempts to find a deadlock using an iterative refinement loop (ILP candidate generation -> BDD reachability check -> Canonical cut).
    * *BDD-Only Filtering:* Uses an ILP pre-check followed by BDD operations to filter out all "live" states. This method is generally recommended for performance.
5.  **Optimization:** Identifies the reachable marking that maximizes the objective function $\sum c_i M_i$ (using a default cost vector $c = \vec{1}$).

---

## 6. Implementation Details

### Task 1: PNML Parsing (`PetriNetReading.py`)
* Uses `xml.etree.ElementTree` to parse the hierarchical structure of standard PNML files.
* Constructs the Input ($I$) and Output ($O$) Incidence Matrices as `numpy` arrays for efficient algebraic manipulation.
* Extracts the initial marking $M_0$ and ensures 1-safe consistency.

### Task 2: Explicit Reachability (`ExplicitComputation.py`)
* **BFS:** Implemented using `collections.deque` as a FIFO queue to explore the state space layer by layer.
* **DFS:** Implemented using `collections.deque` as a LIFO stack to explore deep paths first.
* **State Storage:** Visited markings are stored as Python `tuples` within a `set` data structure, ensuring $O(1)$ average time complexity for lookup and insertion.

### Task 3: Symbolic Reachability (`SymbolicComputation.py`)
* **Library:** Utilizes the `dd` library for pure Python Binary Decision Diagram manipulation.
* **Encoding:** Places are encoded as boolean variables. A set of markings is represented as a boolean function.
* **Transition Chaining:** Instead of computing a monolithic transition relation, the algorithm applies transitions sequentially and updates the reachable set immediately within the loop ("chaining"). This technique significantly reduces the number of iterations required to reach a fixed point.

### Task 4: Deadlock Detection (`DeadlockDetecting.py`)
Two distinct strategies are implemented to handle deadlock detection:
1.  **Iterative ILP (Hybrid):** Solves an Integer Linear Program to find a *potential* dead marking. If the BDD check reveals this marking is unreachable, a "canonical cut" constraint is added to the ILP to exclude it, and the process repeats.
2.  **BDD Filtering (Optimized):** * First, performs a quick ILP check to see if *any* dead marking exists in the mathematical vector space.
    * If yes, it takes the BDD of all reachable markings and logically filters out any state that enables at least one transition.
    * Any remaining states in the BDD are reachable deadlocks.

### Task 5: Optimization (`Optimization.py`)
* **Objective:** Maximize $c^T M$ subject to $M \in Reachable(M_0)$.
* **Algorithm:** Implements a **Branch-and-Bound** search directly over the BDD structure (nodes).
* **Pruning:** At each BDD node, the algorithm calculates the upper bound of the potential value for that subtree. If this upper bound is not greater than the current best value found, the entire branch is pruned to save computation time.

---
## 7. About the images
* To generate these images comparing time and peaked memory used for bfs (explicit) and bdd reachable, make sure to place all files in a folder (including large_input pnml file)
*  Make sure to pip install matplotlib, pandas and numpy
  ```bash
pip install matplotlib pandas numpy
```
* Then simply run 
```bash
python main_compare.py
```
*Then select option 1
---
## 8. Notes
* **1-Safe Assumption:** The input PNML models are assumed to be 1-safe Petri Nets as per the assignment specification.
* **Deadlock Definition:** A deadlock is strictly defined as a reachable marking where **no** transitions are enabled.
* **Performance:** For networks with large state spaces (state explosion), the function **deadlock_iterative_ilp_bdd(pn, ReachSet_BDD, num_reach, max_iter=1000)** in DeadlockDetecting.py may explode running time, especially if tunning higher figure for max_iter this make sure to comment the test for this function out if planning to do use big dataset (containing many places, transitions, reachable states).
* **About the pnml files** : we have already provided 2 pnml file for convenience to run (this file also used to be disussed in the report), but the user can paste any 1 safe petrinet pnml file and run that file if desired.