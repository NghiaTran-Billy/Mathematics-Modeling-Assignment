import collections
from typing import Tuple, List, Optional
# from pyeda.inter import *
from task1 import PetriNet
from collections import deque
import numpy as np
from dd.autoref import BDD  


def bdd_reachable(pn):
    # ---------------------------------------------------------
    # 1. KHỞI TẠO QUẢN LÝ BDD
    # ---------------------------------------------------------
    bdd = BDD()
    
    # Khai báo biến: DD quản lý biến theo tên (string)
    # Ta dùng chính place_ids trong PNML làm tên biến
    bdd.declare(*pn.place_ids)
    
    # ---------------------------------------------------------
    # 2. TẠO TRẠNG THÁI KHỞI TẠO (M0)
    # ---------------------------------------------------------
    # Trong DD: bdd.var('name') trả về node biến
    # ~var là phủ định (NOT), & là AND, | là OR
    
    M0_expr = bdd.true # Bắt đầu là True (1)
    
    for i, p_id in enumerate(pn.place_ids):
        var_node = bdd.var(p_id)
        if pn.M0[i] == 1:
            M0_expr &= var_node  # Có token
        else:
            M0_expr &= ~var_node # Không có token
            
    Reached = M0_expr

    # ---------------------------------------------------------
    # 3. XÂY DỰNG LOGIC TRANSITION
    # ---------------------------------------------------------
    transitions_logic = []
    
    for t_idx in range(len(pn.trans_ids)):
        if t_idx >= pn.I.shape[0] or t_idx >= pn.O.shape[0]: continue

        input_indices = np.flatnonzero(pn.I[t_idx, :]).tolist()
        output_indices = np.flatnonzero(pn.O[t_idx, :]).tolist()
        
        # A. Condition (Guard)
        condition = bdd.true
        for idx in input_indices:
            p_name = pn.place_ids[idx]
            condition &= bdd.var(p_name)
            
        # B. Change Vars (Lưu danh sách TÊN BIẾN cần xóa)
        # DD hàm quantify cần set các string tên biến
        change_vars_set = set()
        for idx in set(input_indices + output_indices):
            change_vars_set.add(pn.place_ids[idx])
            
        # C. Update Mask (Giá trị mới)
        update_mask = bdd.true
        for idx in input_indices:
            if idx not in output_indices:
                # Mất token -> AND NOT
                update_mask &= ~bdd.var(pn.place_ids[idx])
        for idx in output_indices:
            # Có token -> AND VAR
            update_mask &= bdd.var(pn.place_ids[idx])
            
        # D. Sort Key (Để tối ưu thứ tự duyệt)
        min_input_idx = min(input_indices) if input_indices else 999999
        
        transitions_logic.append({
            'name': pn.trans_ids[t_idx],
            'condition': condition,
            'change_vars': change_vars_set, # Set các chuỗi tên biến
            'update': update_mask,
            'sort_key': min_input_idx
        })

    # Sắp xếp theo dòng chảy dữ liệu (Tối ưu Domino)
    transitions_logic.sort(key=lambda x: x['sort_key'])

    # ---------------------------------------------------------
    # 4. VÒNG LẶP CHAINING (Domino Effect)
    # ---------------------------------------------------------
    print("   [DD] Starting Reachability Analysis...")
    
    while True:
        previous_reached = Reached
        
        for t in transitions_logic:
            # 1. Tìm tập trạng thái thỏa mãn (AND)
            potential = t['condition'] & Reached
            
            # Kiểm tra rỗng (trong DD so sánh với bdd.false)
            if potential == bdd.false:
                continue
                
            # 2. Tính Next State
            # a. Existential Quantification (Xóa biến cũ)
            # Thay vì smoothing, DD dùng hàm quantify(expr, vars, forall=False)
            abstracted = bdd.quantify(potential, t['change_vars'], forall=False)
            
            # b. Apply Update (Gán biến mới)
            next_states = abstracted & t['update']
            
            # 3. Update ngay lập tức (Chaining)
            Reached = Reached | next_states
            
        # Điều kiện dừng
        if Reached == previous_reached:
            break
            
    # Đếm số lượng trạng thái
    # nvars=len(pn.place_ids) để đảm bảo đếm đúng không gian biến
    num_reachable = bdd.count(Reached, nvars=len(pn.place_ids))
    
    return Reached, num_reachable
    
