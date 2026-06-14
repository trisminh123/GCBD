import numpy as np


def single_peel(
    iters: int,
    percentile: float,
    node_sparse: np.ndarray,
    density: np.ndarray,
    activate_node_sparse: np.ndarray,
    old_unactivate_nodes: np.ndarray,
):
    """
    Loại bỏ (peel) các node có mật độ thấp.

    Args:
        iters               : số vòng lặp hiện tại (bắt đầu từ 1)
        percentile          : tỉ lệ phần trăm ngưỡng để loại node
        node_sparse         : (no_node_sparse, no_dims) tọa độ các node
        density             : (no_node_sparse,) mật độ của từng node
        activate_node_sparse: (no_node_sparse,) cờ kích hoạt node (1 = còn hoạt động, 0 = đã bị loại)
        old_unactivate_nodes: index (0-based) của các node đã bị loại ở các vòng trước

    Returns:
        activate_node_sparse : mảng trạng thái node sau khi cập nhật
        old_unactivate_nodes : tập index tất cả các node đã bị loại
        new_unactivate_nodes : các node mới bị loại trong vòng lặp hiện tại
    """
    no_node_sparse = node_sparse.shape[0]
    # Sắp xếp index các node theo mật độ tăng dần
    order_by_density = np.argsort(density)
    # Lấy danh sách mật độ theo thứ tự đã sắp xếp
    nodes_sorted_by_dens = density[order_by_density]

    # Tính index của ngưỡng percentile tích lũy
    index_percentile = int(np.ceil(no_node_sparse * (1 - (1 - percentile) ** iters))) - 1
    index_percentile = max(0, min(index_percentile, no_node_sparse - 1))
    # Giá trị mật độ tại ngưỡng
    threshold_value = nodes_sorted_by_dens[index_percentile]
    # Các node có mật độ nhỏ hơn hoặc bằng ngưỡng sẽ bị loại
    unactivate_nodes = np.where(density <= threshold_value)[0]
    # Các node mới bị loại trong vòng lặp này
    new_unactivate_nodes = np.setdiff1d(unactivate_nodes, old_unactivate_nodes)
    # Cập nhật tập node đã bị loại
    old_unactivate_nodes = unactivate_nodes.copy()
    # Đánh dấu các node mới bị loại là inactive
    activate_node_sparse[new_unactivate_nodes] = 0

    return activate_node_sparse, old_unactivate_nodes, new_unactivate_nodes
