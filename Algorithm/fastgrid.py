import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

from Algorithm.unique_rows import unique_rows
from Algorithm.single_peel import single_peel
from Algorithm.link_core import link_core
from Algorithm.merge_core_nd import merge_core_nd


def fastgrid(
    data: np.ndarray,
    no_grid: int = 26,
    percentile: float = 0.1,
    max_iters: int = 9,
) -> np.ndarray:
    """
    Cài đặt thuật toán GCBD theo bài báo:
    Du & Wu - Grid-Based Clustering Using Boundary Detection (Entropy 2022)

    Pipeline của thuật toán:

    Step 1  : Chuẩn hóa dữ liệu và ánh xạ vào lưới (Eq.3)
    Step 2  : Tính mật độ node (Eq.4–5)
    Step 3  : Boundary detection bằng peeling (Section 4.3)
    Step 4  : Merge core nodes thành cluster (Definition 6–8)
    Step 5  : Gán border node vào core cluster (Section 4.4.2)
    Step 6  : Gán điểm dữ liệu vào cluster (Section 4.4.3)
    """
    verbose = False
    nd_change_thres = 1

    no_objs, no_dims = data.shape

     # ============================================================
    # STEP 1 — STANDARD GRID STRUCTURE (Section 4.1)
    # ============================================================
    #
    # Eq.(3) trong paper:
    #
    # Φ(xij) = l * (xij − xmin) / (xmax − xmin) + 1
    #
    # Ý nghĩa:
    # - Chuẩn hóa dữ liệu về hệ tọa độ lưới
    # - Sau phép biến đổi:
    #       Φ(xij) ∈ [1, l+1]
    #
    # Node grid có tọa độ nguyên:
    #       vj ∈ {1,2,...,l+1}
    #
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_scaled = (no_grid-1) * (data - data_min) / (data_max - data_min) + 1

    data_temp = data_scaled.copy()
    data_temp[data_temp == no_grid] = no_grid  - 1e-10

   # ============================================================
    # STEP 2 — NODE DENSITY ESTIMATION (Section 4.2)
    # ============================================================
    #
    # Mỗi điểm dữ liệu nằm trong một cell của grid.
    #
    # Trong không gian m chiều:
    # mỗi cell có 2^m corner nodes.
    #
    # GCBD phân phối mật độ của điểm tới các corner nodes
    # thông qua hàm membership fj (Eq.4).
    #
    n_corners = 2 ** no_dims

        # Tạo vector nhị phân biểu diễn 2^m corner
    # Ví dụ m=2:
    # 00
    # 01
    # 10
    # 11
    #
    # => offset node
    bin_sequence = np.array(
        [[int(b) for b in format(k, f'0{no_dims}b')] for k in range(n_corners)],
        dtype=float,
    )

    # floor() xác định góc trái dưới của cell chứa điểm
    data_floor = np.floor(data_temp) 
    # ============================================================
    # Tìm tất cả node lân cận của mỗi điểm
    #
    # nodes_exp shape = (n_points, 2^m, m)
    # ============================================================
    nodes_exp = data_floor[:, np.newaxis, :] + bin_sequence[np.newaxis, :, :]

    # flatten để dễ tính toán sau này
    data_nodes_mat = nodes_exp.reshape(no_objs * n_corners, no_dims)

   # ============================================================
    # Eq.(4) — Membership function
    #
    # fj(vj) = max(1 − |Φ(xij) − vj| , 0)
    #
    # Ý nghĩa:
    # khoảng cách càng gần node → mật độ đóng góp càng lớn
    # ============================================================
    data_exp = data_temp[:, np.newaxis, :]          # (n, 1, m)
    fd = np.maximum(1.0 - np.abs(data_exp - nodes_exp), 0.0)  # (n, 2^m, m)

    # ============================================================
    # Eq.(5) — Node density contribution
    #
    # ρv = Π fj(vj)
    #
    # Nhân các membership theo từng chiều
    # ============================================================

    single_gain_mat = np.prod(fd, axis=2)           # (n, 2^m)
    single_gain_onedim = single_gain_mat.reshape(no_objs * n_corners)  # (n*2^m,)

    # Chỉ giữ các node thực sự tồn tại (sparse nodes)
    node_sparse, _, IC = unique_rows(data_nodes_mat)  # IC: 1-based index per row
    no_node_sparse = node_sparse.shape[0]

    # ============================================================
    # Tính mật độ node
    #
    # ρv = Σ contributions của các điểm lân cận
    # ============================================================
    rows = np.arange(no_objs * n_corners)
    cols = IC - 1  # convert to 0-based
    membership_remat = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(no_objs * n_corners, no_node_sparse),
    )

    density = np.array(membership_remat.T.dot(single_gain_onedim)).ravel()

    # ── node → data-point mapping ──────────────────────────────────────────
    # membership_pt[i, j] = 1 if data point i is adjacent to node j
    # (collapse the 2^m corner rows per point with max)
    membership_dense = membership_remat.toarray()                      # (n*2^m, N)
    membership_pt = membership_dense.reshape(
        no_objs, n_corners, no_node_sparse
    ).max(axis=1)                                                       # (n, N)

    # node_ind_point_cell[j] = 0-based indices of data points adjacent to node j
    node_ind_point_cell = [
        np.where(membership_pt[:, j] > 0)[0] for j in range(no_node_sparse)
    ]

   # ============================================================
    # STEP 3 — BOUNDARY DETECTION (Section 4.3)
    # ============================================================
    #
    # Ý tưởng chính của GCBD:
    #
    # Lặp nhiều vòng:
    #   - loại bỏ node có mật độ thấp (boundary nodes)
    #   - giữ lại node mật độ cao (core nodes)
    #
    dist_node = squareform(pdist(node_sparse, metric='chebyshev'))

    activate_node_sparse = np.ones(no_node_sparse)
    # activate_data_remat[k, i] = 1 iff data point i is still active
    # shape (2^m, n) mirrors MATLAB's repmat layout
    activate_data_remat = np.ones((n_corners, no_objs))

    old_unactivate_nodes = np.array([], dtype=int)
    iters = 1
    prev_iter_nd_len = 0

    border_nodes_per_iter = []
    core_nodes_per_iter = []

    while max_iters >= iters:
        # single_peel:
        # chọn các node nằm trong percentile thấp nhất
        activate_node_sparse, old_unactivate_nodes, new_add_unactivate_nodes = single_peel(
            iters, percentile, node_sparse, density,
            activate_node_sparse, old_unactivate_nodes,
        )

        # nếu không còn node mới bị loại → dừng
        if len(new_add_unactivate_nodes) == 0:
            if verbose:
                print("No new inactive nodes — stopping.")
            break

        curr_iter_nd_len = len(old_unactivate_nodes)
        if abs(curr_iter_nd_len - prev_iter_nd_len) < nd_change_thres and iters != 1:
            if verbose:
                print("Node-count change below threshold — stopping.")
            break
        prev_iter_nd_len = curr_iter_nd_len

        # LINK CORE (Section 4.4.2)
        #
        # mỗi boundary node sẽ được liên kết
        # với core node gần nhất và có mật độ cao nhất
        link_core_nodes_indices, unactivate_points = link_core(
            dist_node, node_ind_point_cell,
            activate_node_sparse, new_add_unactivate_nodes, density,
        )

        border_nodes_per_iter.append(new_add_unactivate_nodes.copy())
        core_nodes_per_iter.append(link_core_nodes_indices.copy())

       # Eq.(9) trong paper
        #
        # X(t+1) = X(t) − X_U(t)
        #
        # loại bỏ các điểm bị peel khỏi tính density ở vòng tiếp theo
        activate_data_remat[:, unactivate_points] = 0
        oneiter_gain = single_gain_mat * activate_data_remat.T  # (n, 2^m)
        oneiter_gain_1d = oneiter_gain.reshape(no_objs * n_corners)
        density = np.array(membership_remat.T.dot(oneiter_gain_1d)).ravel()

        iters += 1

    # ============================================================
    # STEP 4 — MERGE CORE NODES (Section 4.4.1)
    # ============================================================
    #
    # Các core node được gộp thành cluster
    # dựa trên connectivity
    #
    core_nd = np.setdiff1d(np.arange(no_node_sparse), old_unactivate_nodes)

    # Build distance matrix restricted to active nodes:
    # inactive nodes get distance 0 → set to Inf so they are never "nearest"
    # Broadcast: multiply each row i by activate_node_sparse[i]
    dist_last = (dist_node * activate_node_sparse[:, np.newaxis]).T
    dist_last[dist_last == 0] = np.inf

    cluster_lists = merge_core_nd(dist_last, core_nd)

    nd_clusters = np.zeros(no_node_sparse, dtype=int)
    cluster_index = 1
    for cl in cluster_lists:
        nd_clusters[cl] = cluster_index
        cluster_index += 1

    # ── Step 5: Assign boundary nodes back to clusters (Section 4.4.2) ────
    # Reverse iteration order: innermost boundary layer first
    for i in range(len(border_nodes_per_iter) - 1, -1, -1):
        nd_clusters[border_nodes_per_iter[i]] = nd_clusters[core_nodes_per_iter[i]]

    # ── Step 6: Map data points to clusters (Section 4.4.3) ───────────────
    # "round" finds the nearest node for each point (paper Section 4.4.3)
    # mỗi điểm dữ liệu được gán vào cluster của node gần nhất
    nearest_node = np.round(data_scaled).astype(int)  # use scaled (not clamped) data

    # Vectorised lookup: find row index in node_sparse matching each nearest_node
    node_sparse_int = node_sparse.astype(int)
    # Build a dict for O(n) lookup instead of O(n * N) loop
    node_coord_to_idx = {
        tuple(node_sparse_int[j]): j for j in range(no_node_sparse)
    }
    locb = np.array(
        [node_coord_to_idx.get(tuple(nearest_node[i]), 0) for i in range(no_objs)],
        dtype=int,
    )

    pt_clusters = nd_clusters[locb]
    return pt_clusters