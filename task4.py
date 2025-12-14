from task1 import PetriNet
from task3 import bdd_reachable 
import numpy as np
import pulp

def build_deadmark_ilp_model(pn: PetriNet):
    """
    Tạo ILP model cho bài toán tìm dead marking (1-safe).
    """
    P = len(pn.place_ids)
    T = len(pn.trans_ids)

    model = pulp.LpProblem("DeadMarkingIterative", pulp.LpMinimize)

    # Biến M_p ∈ {0,1}
    M = pulp.LpVariable.dicts("M", range(P), lowBound=0, upBound=1, cat="Binary")

    # Deadness constraints: sum_{p in preset(t)} M_p <= |preset(t)| - 1
    for t in range(T):
        inputs = np.where(pn.I[t, :] > 0)[0].tolist()

        if not inputs:
            # Transition không có input → luôn enable → không có dead marking.
            # Tạo bộ ràng buộc mâu thuẫn để ILP chắc chắn vô nghiệm:
            total_M = pulp.lpSum([M[p] for p in range(P)])
            model += total_M <= 0
            model += total_M >= 1
            continue

        model += pulp.lpSum([M[p] for p in inputs]) <= len(inputs) - 1

    # Không quan tâm objective, đặt 0 cho gọn
    model += 0

    return model, M


def marking_to_bdd(marking, pn, bdd):
    """
    Convert một marking (tuple 0/1, độ dài = số place)
    thành node BDD tương ứng, dùng manager `bdd` và tên biến = pn.place_ids.
    """
    node = bdd.true
    for bit, place_id in zip(marking, pn.place_ids):
        var = bdd.var(place_id)
        if bit == 1:
            node &= var
        else:
            node &= ~var
    return node

def deadlock_iterative_ilp_bdd(pn, ReachSet_BDD, num_reach, max_iter=1000):
    """
    Version dùng dd.autoref.BDD:
      - Input:
          pn            : PetriNet
          ReachSet_BDD  : node BDD trả về từ bdd_algo.bdd_reachable(pn)
          num_reach     : số trạng thái reachable (chỉ để report)
      - Output: (deadlock_marking or None, message)
    """
    # Lấy BDD manager từ node ReachSet_BDD
    bdd = ReachSet_BDD.bdd
    One = bdd.true
    Zero = bdd.false

    # Build ILP model (deadness constraints)
    model, Mvars = build_deadmark_ilp_model(pn)
    P = len(pn.place_ids)

    for it in range(max_iter):
        # 1) Giải ILP
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            # Không còn dead marking nào trong toàn space 0/1^P
            if it == 0:
                msg = "No dead marking at all → NO DEADLOCK."
            else:
                msg = "All dead markings found so far are unreachable → NO DEADLOCK."
            return None, msg

        # 2) Lấy nghiệm Mcand
        Mcand = tuple(int(pulp.value(Mvars[p])) for p in range(P))

        # 3) Kiểm tra reachable bằng BDD
        cand_bdd = marking_to_bdd(Mcand, pn, bdd)
        Check = ReachSet_BDD & cand_bdd

        if Check != Zero:
            # Mcand thuộc tập reachable → DEADLOCK
            return Mcand, f"Deadlock found at iteration {it + 1}"

        # 4) Nếu unreachable → thêm cắt ILP để loại Mcand đi
        P1 = [p for p in range(P) if Mcand[p] == 1]
        P0 = [p for p in range(P) if Mcand[p] == 0]

        cut_expr = (
            pulp.lpSum([Mvars[p] for p in P1]) -
            pulp.lpSum([Mvars[p] for p in P0])
        )
        model += cut_expr <= len(P1) - 1

    return None, f"Stopped after {max_iter} iterations without finding a reachable deadlock."

def deadlock_bdd2(pn, ReachSet_BDD, num_reach):
    """
    Version 2:
      1) ILP pre-check xem có dead marking nào trong 0/1^P không.
      2) Nếu có → dùng BDD để lọc ra deadlock reachable.

    Input:
        pn           : PetriNet
        ReachSet_BDD : node BDD từ bdd_algo.bdd_reachable(pn)
        num_reach    : số trạng thái reachable
    """
    # Lấy BDD manager từ node
    bdd = ReachSet_BDD.bdd
    Zero = bdd.false

    # --------- 1. ILP PRE-CHECK ---------
    model, Mvars = build_deadmark_ilp_model(pn)
    P = len(pn.place_ids)

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))

    # CASE A: Không có dead marking nào
    if pulp.LpStatus[status] != "Optimal":
        return None, "No dead marking at all (ILP) -> NO DEADLOCK."

    # CASE B: Có ít nhất 1 dead marking (trong toàn space, chưa chắc reachable)
    Mcand_ilp = tuple(int(pulp.value(Mvars[p])) for p in range(P))

    cand_bdd = marking_to_bdd(Mcand_ilp, pn, bdd)
    Check = ReachSet_BDD & cand_bdd
    
    if Check != Zero:
        # Deadmark ILP đã reachable → deadlock luôn
        return Mcand_ilp, "Deadlock found directly from ILP pre-check (reachable deadmark)."
    else:
        print("there are deadmark(s) somewhere... ")
    # --------- 2. BDD THUẦN: tìm reachable deadlock ---------
    num_trans = len(pn.trans_ids)
    DeadlockBDD = ReachSet_BDD  # ban đầu: mọi reachable đều là candidate

    for trans_idx in range(num_trans):
        if DeadlockBDD == Zero:
            # Không còn candidate nào
            return None, "Dead markings exist (ILP), but none reachable (BDD) -> NO DEADLOCK."

        if trans_idx >= pn.I.shape[0] or trans_idx >= pn.O.shape[0]:
            continue

        input_places = np.flatnonzero(pn.I[trans_idx, :] > 0).tolist()
        output_places = np.flatnonzero(pn.O[trans_idx, :] > 0).tolist()

        # Xây cond: những marking enable được transition này
        cond = bdd.true
        for idx in input_places:
            cond &= bdd.var(pn.place_ids[idx])
        for idx in output_places:
            if idx not in input_places:
                cond &= ~bdd.var(pn.place_ids[idx])

        # Loại bỏ những reachable state vẫn enable được transition này
        DeadlockBDD &= ~cond

    if DeadlockBDD == Zero:
        return None, "Dead markings exist (ILP), but none reachable (BDD) -> NO DEADLOCK."

    # Lấy một reachable deadlock cụ thể
    assignment = bdd.pick(DeadlockBDD)  # dict {var_name: 0/1}
    deadlock_marking = tuple(int(assignment.get(place_id, 0)) for place_id in pn.place_ids)

    return deadlock_marking, "Deadlock found by BDD filtering (no big AnyEnabled)."

def deadlock_iterative_ilp_bdd_auto(pn, max_iter=1000):
    ReachSet_BDD, num_reach = bdd_reachable(pn)
    return deadlock_iterative_ilp_bdd(pn, ReachSet_BDD, num_reach, max_iter)

def deadlock_bdd2_auto(pn):
    ReachSet_BDD, num_reach = bdd_reachable(pn)
    return deadlock_bdd2(pn, ReachSet_BDD, num_reach)