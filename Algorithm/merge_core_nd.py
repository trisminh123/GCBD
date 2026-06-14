import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def merge_core_nd(dist_last: np.ndarray, core_nd: np.ndarray) -> list:
    """
    Phân cụm các node lõi (core nodes) dựa trên quan hệ khoảng cách giữa chúng.
    Hai node lõi được coi là liên kết nếu khoảng cách Chebyshev giữa chúng = 1
    (Definition 6 trong bài báo).

    Args:
        dist_last : ma trận khoảng cách (no_node_sparse x no_node_sparse)
        core_nd   : index (0-based) của các node lõi

    Returns:
        Danh sách các cụm; mỗi cụm là một mảng chứa index (0-based) của các node.
    """
    #  Lấy ma trận khoảng cách chỉ giữa các core node
    dist_core = dist_last[np.ix_(core_nd, core_nd)]
    # Tạo ma trận liên kết:
    # nếu khoảng cách Chebyshev giữa 2 node = 1 → chúng được nối với nhau
    link = csr_matrix((dist_core == 1).astype(float))

    # Tìm các thành phần liên thông trong đồ thị
    # mỗi thành phần liên thông tương ứng với một cluster
    n_components, labels = connected_components(link, directed=False)

    # Gom các node theo nhãn component
    clusters = [core_nd[labels == i] for i in range(n_components)]
    return clusters