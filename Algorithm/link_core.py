"""
gcbd_viz.py — Visualize từng bước lặp của thuật toán GCBD
Scroll down để xem từng iteration:
  - Điểm dữ liệu (đã chuẩn hóa): xanh dương
  - Node còn active: đỏ, kích thước theo density, có số thứ tự
  - Node vừa bị loại ở bước này: đen
  - Node đã bị loại ở các bước trước: xám

Cách dùng:
    python gcbd_viz.py
    python gcbd_viz.py --arff dataset/arff_files/jain.arff
    python gcbd_viz.py --arff dataset/arff_files/jain.arff --no_grid 26 --max_iters 9
"""

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.io.arff import loadarff
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pathlib import Path

def link_core(dist_node, node_ind_point_cell,
              activate_node_sparse, new_unactivate_nodes, density):
    unactivate_points = np.array([], dtype=int)
    dist = dist_node[:, new_unactivate_nodes] * activate_node_sparse[:, np.newaxis]
    dist[dist == 0] = np.inf
    min_dist = np.min(dist, axis=0)
    link_core_nodes_indices = np.zeros(len(new_unactivate_nodes), dtype=int)
    for i, node_idx in enumerate(new_unactivate_nodes):
        unactivate_points = np.union1d(unactivate_points,
                                       node_ind_point_cell[node_idx])
        min_dist_indices = np.where(dist[:, i] == min_dist[i])[0]
        sec_ind = np.argmax(density[min_dist_indices])
        link_core_nodes_indices[i] = min_dist_indices[sec_ind]
    return link_core_nodes_indices, unactivate_points


