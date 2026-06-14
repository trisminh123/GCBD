import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

from unique_rows import unique_rows
from single_peel import single_peel
from link_core import link_core
from merge_core_nd import merge_core_nd


def fastgrid_gaussian(
    data      : np.ndarray,
    no_grid   : int   = 26,
    percentile: float = 0.1,
    max_iters : int   = 9,
    sigma     : float = 1.0,
) -> np.ndarray:
    """
    Phiên bản cải tiến của GCBD với Gaussian kernel density estimation.

    Thay đổi duy nhất so với fastgrid gốc: STEP 2
    ─────────────────────────────────────────────────────────────────
    Gốc  (Eq.4-5): ρ_v = Π max(1 - |Φ(x_ij) - v_j|, 0)
                   → Linear membership, chỉ xét 2^m corner nodes của cell

    Mới  (Gaussian kernel):
                   ρ_v = Σ_{x_i ∈ neighbors} exp(-||Φ(x_i) - v||² / 2σ²)
                   → Smooth density surface, decay tự nhiên theo khoảng cách
    ─────────────────────────────────────────────────────────────────

    Lý do cải tiến:
    - Linear membership tạo density surface gãy khúc tại biên cell
    - Gaussian tạo density surface mượt → contrast core/boundary rõ hơn
    - Boundary detection (Step 3) chính xác hơn nhờ gradient mượt

    Về tham số sigma:
    - σ = 1.0 tương ứng 1 cell width trong không gian đã chuẩn hóa
    - Cùng vùng ảnh hưởng với linear gốc nhưng smooth hơn
    - Không cần người dùng chỉnh (hardcode theo cell width)

    Args:
        data       : (n, m) dữ liệu đầu vào
        no_grid    : số interval mỗi chiều (tham số l trong paper)
        percentile : tỉ lệ node bị peel mỗi iter (giữ nguyên như gốc)
        max_iters  : số vòng lặp tối đa (giữ nguyên như gốc)
        sigma      : bandwidth của Gaussian kernel (mặc định 1.0)
    """
    nd_change_thres = 1

    no_objs, no_dims = data.shape

    # ============================================================
    # STEP 1 — STANDARD GRID STRUCTURE (không đổi)
    # ============================================================
    data_min    = data.min(axis=0)
    data_max    = data.max(axis=0)
    data_scaled = (no_grid - 1) * (data - data_min) / (data_max - data_min) + 1

    data_temp = data_scaled.copy()
    data_temp[data_temp == no_grid] = no_grid - 1e-10

    # ============================================================
    # STEP 2 — GAUSSIAN KERNEL DENSITY ESTIMATION  ← THAY ĐỔI CHÍNH
    # ============================================================
    #
    # Với mỗi node v trong sparse grid:
    #
    #   ρ_v = Σ_{x_i ∈ neighbors(v)} exp( -||Φ(x_i) - v||² / 2σ² )
    #
    # "neighbors(v)" = các điểm nằm trong cell có v là corner node
    # (giữ nguyên cấu trúc sparse của paper, chỉ thay hàm đóng góp)
    #
    # So sánh với linear gốc:
    #   Linear  : contribution = Π max(1 - |Φ(x_ij) - v_j|, 0)
    #             → tích các membership 1 chiều, gãy khúc tại biên
    #   Gaussian: contribution = exp(-||Φ(x_i) - v||² / 2σ²)
    #             → khoảng cách Euclidean, smooth và isotropic
    #
    n_corners    = 2 ** no_dims
    bin_sequence = np.array(
        [[int(b) for b in format(k, f'0{no_dims}b')] for k in range(n_corners)],
        dtype=float,
    )

    data_floor = np.floor(data_temp)
    nodes_exp  = data_floor[:, np.newaxis, :] + bin_sequence[np.newaxis, :, :]
    # nodes_exp shape: (n_objs, 2^m, m)

    data_nodes_mat = nodes_exp.reshape(no_objs * n_corners, no_dims)

    # ── Tính Gaussian contribution ──────────────────────────────────────────
    #
    # data_exp : (n, 1, m)  — tọa độ đã scale của từng điểm
    # nodes_exp: (n, 2^m, m) — tọa độ 2^m corner node của cell chứa điểm
    #
    # sq_dist[i, k] = ||Φ(x_i) - v_k||²   (Euclidean, sum over dims)
    #
    data_exp = data_temp[:, np.newaxis, :]                          # (n, 1, m)
    diff     = data_exp - nodes_exp                                  # (n, 2^m, m)
    sq_dist  = np.sum(diff ** 2, axis=2)                            # (n, 2^m)

    # Gaussian weight — thay thế single_gain_mat của bản gốc
    single_gain_mat    = np.exp(-sq_dist / (2.0 * sigma ** 2))      # (n, 2^m)
    single_gain_onedim = single_gain_mat.reshape(no_objs * n_corners)

    # ── Sparse node lookup (giữ nguyên) ─────────────────────────────────────
    node_sparse, _, IC = unique_rows(data_nodes_mat)
    no_node_sparse      = node_sparse.shape[0]

    rows = np.arange(no_objs * n_corners)
    cols = IC - 1
    membership_remat = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(no_objs * n_corners, no_node_sparse),
    )

    # ρ_v = Σ Gaussian contributions từ các điểm lân cận
    density = np.array(membership_remat.T.dot(single_gain_onedim)).ravel()

    # ── Node → data-point mapping (giữ nguyên) ──────────────────────────────
    membership_dense    = membership_remat.toarray()
    membership_pt       = membership_dense.reshape(
        no_objs, n_corners, no_node_sparse
    ).max(axis=1)
    node_ind_point_cell = [
        np.where(membership_pt[:, j] > 0)[0] for j in range(no_node_sparse)
    ]

    # ============================================================
    # STEP 3–6 — Giữ nguyên hoàn toàn so với bản gốc
    # ============================================================
    dist_node = squareform(pdist(node_sparse, metric='chebyshev'))

    activate_node_sparse = np.ones(no_node_sparse)
    activate_data_remat  = np.ones((n_corners, no_objs))
    old_unactivate_nodes = np.array([], dtype=int)
    iters                = 1
    prev_iter_nd_len     = 0
    border_nodes_per_iter = []
    core_nodes_per_iter   = []

    while max_iters >= iters:
        activate_node_sparse, old_unactivate_nodes, new_add_unactivate_nodes = single_peel(
            iters, percentile, node_sparse, density,
            activate_node_sparse, old_unactivate_nodes,
        )

        if len(new_add_unactivate_nodes) == 0:
            break

        curr_iter_nd_len = len(old_unactivate_nodes)
        if abs(curr_iter_nd_len - prev_iter_nd_len) < nd_change_thres and iters != 1:
            break
        prev_iter_nd_len = curr_iter_nd_len

        link_core_nodes_indices, unactivate_points = link_core(
            dist_node, node_ind_point_cell,
            activate_node_sparse, new_add_unactivate_nodes, density,
        )

        border_nodes_per_iter.append(new_add_unactivate_nodes.copy())
        core_nodes_per_iter.append(link_core_nodes_indices.copy())

        activate_data_remat[:, unactivate_points] = 0
        oneiter_gain    = single_gain_mat * activate_data_remat.T
        oneiter_gain_1d = oneiter_gain.reshape(no_objs * n_corners)
        density         = np.array(membership_remat.T.dot(oneiter_gain_1d)).ravel()

        iters += 1

    # ── Step 4: Merge core nodes ─────────────────────────────────────────────
    core_nd   = np.setdiff1d(np.arange(no_node_sparse), old_unactivate_nodes)
    dist_last = (dist_node * activate_node_sparse[:, np.newaxis]).T
    dist_last[dist_last == 0] = np.inf

    cluster_lists = merge_core_nd(dist_last, core_nd)
    nd_clusters   = np.zeros(no_node_sparse, dtype=int)
    for cluster_index, cl in enumerate(cluster_lists, start=1):
        nd_clusters[cl] = cluster_index

    # ── Step 5: Assign boundary nodes ────────────────────────────────────────
    for i in range(len(border_nodes_per_iter) - 1, -1, -1):
        nd_clusters[border_nodes_per_iter[i]] = nd_clusters[core_nodes_per_iter[i]]

    # ── Step 6: Map data points to clusters ──────────────────────────────────
    nearest_node      = np.round(data_scaled).astype(int)
    node_sparse_int   = node_sparse.astype(int)
    node_coord_to_idx = {tuple(node_sparse_int[j]): j for j in range(no_node_sparse)}
    locb = np.array(
        [node_coord_to_idx.get(tuple(nearest_node[i]), 0) for i in range(no_objs)],
        dtype=int,
    )

    return nd_clusters[locb]