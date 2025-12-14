from collections import deque
import numpy as np
from PetriNetReading import PetriNet
from typing import Set, Tuple

def bfs_reachable(pn: PetriNet) -> Set[Tuple[int, ...]]:
    I = pn.I
    O = pn.O
    C = O - I              
    
    M0 = pn.M0.flatten().astype(int)
    initial = tuple(M0.tolist())

    reachable= set()
    queue = deque([initial])
    reachable.add(initial)

    while queue:
        current_tuple = queue.popleft()
        current = np.array(current_tuple, dtype=int) 
        enabled = np.all(current >= I, axis=1)        # (T,)

        for t_idx in np.where(enabled)[0]:
            # C[t_idx, :] là vector (P,)
            new_marking = current + C[t_idx, :]
            new_tuple = tuple(new_marking.tolist())

            if new_tuple not in reachable:
                reachable.add(new_tuple)
                queue.append(new_tuple)

    return reachable

def dfs_reachable(pn: PetriNet) -> Set[Tuple[int, ...]]:
    I = pn.I
    O = pn.O
    C = O - I              # shape (T, P)

    M0 = pn.M0.flatten().astype(int)
    initial = tuple(M0.tolist())

    reachable: Set[Tuple[int, ...]] = set()
    stack = deque([initial])
    reachable.add(initial)

    while stack:
        current_tuple = stack.pop()
        current = np.array(current_tuple, dtype=int)  # (P,)

        enabled = np.all(current >= I, axis=1)        # (T,)

        for t_idx in np.where(enabled)[0]:
            new_marking = current + C[t_idx, :]
            new_tuple = tuple(new_marking.tolist())

            if new_tuple not in reachable:
                reachable.add(new_tuple)
                stack.append(new_tuple)

    return reachable

#print(len(dfs_reachable(PetriNet.from_pnml(r"D:\py_1stbtlmhh\SimpleLoadBal-pnml\SimpleLoadBal\PT\simple_lbs-5.pnml"))))
#print(len(bfs_reachable(PetriNet.from_pnml(r"D:\py_1stbtlmhh\SimpleLoadBal-pnml\SimpleLoadBal\PT\simple_lbs-5.pnml"))))