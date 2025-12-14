from typing import List, Tuple, Optional, Union
import numpy as np
from dd.autoref import Function as DDNode


def _normalize_c(c: Union[List[int], np.ndarray]) -> List[int]:
    """Convert c to a plain Python list of ints."""
    if isinstance(c, np.ndarray):
        return [int(x) for x in c.tolist()]
    return [int(x) for x in c]


def max_reachable_marking(
    place_ids: List[str],
    node: DDNode,                      # dd.autoref.Function (reachable markings BDD)
    c: Union[List[int], np.ndarray],
) -> Tuple[Optional[List[int]], Optional[int]]:
    """
    Branch-and-bound optimization over a dd.autoref BDD.

    Args:
        place_ids: list of place IDs (same order as in c).
        node:      dd.autoref.Function encoding the set of reachable markings.
        c:         cost/weight vector.

    Returns:
        (best_marking, best_value), where:
            best_marking is a list of 0/1 for each place,
            best_value  is the maximum c^T M,
        or (None, None) if there is no satisfying marking.
    """
    c_list = _normalize_c(c)
    n = len(place_ids)

    # Manager associated with this node
    mgr = node.bdd

    # 1. Trivial cases
    if node == mgr.false:
        # No reachable marking
        return None, None

    if node == mgr.true:
        # All {0,1}^n are allowed -> just choose 1 where c_i > 0
        best_mark = [1 if ci > 0 else 0 for ci in c_list]
        best_val = sum(ci * mi for ci, mi in zip(c_list, best_mark))
        return best_mark, best_val

    # 2. Support: variables that actually appear in the BDD
    support_names = mgr.support(node)  # set of variable names (strings)

    ordered_names: List[str] = []
    ordered_indices: List[int] = []
    for idx, name in enumerate(place_ids):
        if name in support_names:
            ordered_names.append(name)
            ordered_indices.append(idx)

    num_active = len(ordered_names)

    # Places not in support are free (unconstrained by the BDD)
    support_name_set = set(support_names)
    free_indices: List[int] = [
        i for i, name in enumerate(place_ids)
        if name not in support_name_set
    ]

    # 3. Precompute positive contributions for bounds
    free_pos_total = sum(
        ci for i, ci in enumerate(c_list)
        if i in free_indices and ci > 0
    )

    support_pos_weights = [max(c_list[idx], 0) for idx in ordered_indices]
    total_support_pos = sum(support_pos_weights)

    # Degenerate: no active vars (should be mgr.true/mgr.false cases), but keep safe
    if num_active == 0:
        best_mark = [1 if ci > 0 else 0 for ci in c_list]
        best_val = sum(ci * mi for ci, mi in zip(c_list, best_mark))
        return best_mark, best_val

    # 4. Branch & Bound search
    best_value: Optional[int] = None
    best_marking: Optional[List[int]] = None

    # current_marking holds decisions for all places; active ones will be set in DFS
    current_marking = [0] * n

    def dfs(level: int, sub_node: DDNode,
            curr_val: int, remaining_support_pos: int) -> None:
        nonlocal best_value, best_marking, current_marking

        # Infeasible: no model under this partial assignment
        if sub_node == mgr.false:
            return

        # Upper bound: current value + remaining positive support + all free positives
        ub = curr_val + remaining_support_pos + free_pos_total
        if best_value is not None and ub <= best_value:
            return

        # All active vars assigned -> build full marking and evaluate
        if level == num_active:
            candidate = [0] * n

            # Active vars
            for idx in ordered_indices:
                candidate[idx] = current_marking[idx]

            # Free vars: greedily 1 if ci > 0
            for idx in free_indices:
                candidate[idx] = 1 if c_list[idx] > 0 else 0

            total = sum(ci * mi for ci, mi in zip(c_list, candidate))
            if best_value is None or total > best_value:
                best_value = total
                best_marking = candidate
            return

        # Next active variable
        name = ordered_names[level]
        idx = ordered_indices[level]
        ci = c_list[idx]

        # Positive part of this var's weight for remaining_support_pos
        pos_part = max(ci, 0)
        new_remaining = remaining_support_pos - pos_part

        # Heuristic: branch on the "good" value first
        if ci >= 0:
            branch_vals = (1, 0)
        else:
            branch_vals = (0, 1)

        for val in branch_vals:
            current_marking[idx] = val

            assignment = {name: bool(val)}
            next_node = mgr.let(assignment, sub_node)

            dfs(level + 1, next_node, curr_val + val * ci, new_remaining)

        current_marking[idx] = 0  # optional cleanup

    # Kick off DFS from the root
    dfs(0, node, curr_val=0, remaining_support_pos=total_support_pos)

    if best_marking is None:
        return None, None

    return best_marking, best_value
