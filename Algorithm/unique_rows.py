import numpy as np


def unique_rows(a: np.ndarray):
    """
    Find unique rows of matrix a.
    Returns:
        c    : unique rows (sorted)
        ind_a: indices into a of the unique rows
        ind_c: for each row of a, its index in c (1-indexed to match MATLAB)
    """
    # Use numpy's built-in; lexsort mimics sortrows
    ind_sort = np.lexsort(a.T[::-1])
    sort_a = a[ind_sort]

    # Groups: rows that differ from the previous
    diff = np.any(sort_a[:-1] != sort_a[1:], axis=1)
    mask = np.concatenate([[True], diff])

    c = sort_a[mask]
    ind_a = ind_sort[mask]

    # ind_c: each original row → index (1-based) in c
    ind_c_sorted = np.cumsum(mask)          # 1-based indices in sorted order
    ind_c = np.empty(a.shape[0], dtype=int)
    ind_c[ind_sort] = ind_c_sorted

    return c, ind_a, ind_c
