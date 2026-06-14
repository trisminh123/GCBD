"""
benchmark_gaussian.py — So sánh GCBD gốc vs GCBD Gaussian kernel vs CLIQUE
trên toàn bộ file .arff trong folder dataset/synthetic/

Kết quả xuất ra:
    results/benchmark_gaussian_results.csv   ← bảng đầy đủ mọi tổ hợp
    results/benchmark_gaussian_summary.csv   ← best AMI mỗi thuật toán / dataset
    results/clustering_<dataset>.png         ← hình phân cụm 4 panel

Cách dùng:
    python benchmark_gaussian.py
"""

import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.preprocessing import MinMaxScaler

from Algorithm.fastgrid import fastgrid as GCBD
from Algorithm.fastgrid_gaussian import fastgrid_gaussian as GKGCBD

try:
    from pyclustering.cluster.clique import clique
    CLIQUE_AVAILABLE = True
except ImportError:
    CLIQUE_AVAILABLE = False
    print("CẢNH BÁO: pyclustering chưa được cài. Chạy: pip install pyclustering")


# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
DATASET_DIR = "Datasets/Synthetic_dataset"
RESULT_DIR  = "results/Synthetic"

# Grid search — cùng không gian tham số cho cả 2
GRID_PARAMS = {
    "no_grid"   : list(range(6, 52)),
    "percentile": [0.1],
    "max_iters" : list(range(2, 13)),
}

# Gaussian thêm sigma
SIGMA_VALUES = [0.25, 0.5, 0.75, 1.0]

# CLIQUE grid search
CLIQUE_PARAMS = {
    "intervals"  : list(range(5, 51)),      # l = [5, 50]
    "threshold": list(range(0, 6))  # c = [0, 1, 2, 3, 4, 5]
}

# Màu sắc cho các cluster (tương tự ảnh mẫu)
CLUSTER_COLORS = [
    "#4CAF50",  # green
    "#F44336",  # red
    "#FF9800",  # orange
    "#2196F3",  # blue
    "#9C27B0",  # purple
    "#795548",  # brown
    "#009688",  # teal
    "#FF5722",  # deep orange
    "#607D8B",  # blue grey
    "#E91E63",  # pink
    "#CDDC39",  # lime
    "#00BCD4",  # cyan
]
NOISE_COLOR = "#BDBDBD"
# ─────────────────────────────────────────────


def load_arff(path: str):
    from scipy.io.arff import loadarff
    raw, meta = loadarff(path)
    feature_cols, label_col = [], None
    for name in meta.names():
        typ = meta[name][0]
        col = np.array([row[name] for row in raw])
        if name.lower() in ("class", "label", "target", "cluster"):
            label_col = col
        elif typ in ("numeric", "real", "integer"):
            feature_cols.append(col.astype(float))
    data = np.column_stack(feature_cols) if feature_cols else None
    if label_col is not None:
        try:
            labels = label_col.astype(int)
        except (ValueError, TypeError):
            unique_vals = sorted(set(label_col))
            mapping     = {v: i for i, v in enumerate(unique_vals)}
            labels      = np.array([mapping[v] for v in label_col], dtype=int)
    else:
        labels = np.full(len(data), -1, dtype=int)
    return data, labels


def compute_metrics(true_labels, pred_labels):
    mask = true_labels >= 0
    if mask.sum() == 0:
        return None, None
    ami = adjusted_mutual_info_score(true_labels[mask], pred_labels[mask])
    fmi = fowlkes_mallows_score(true_labels[mask], pred_labels[mask])
    return round(ami, 4), round(fmi, 4)


def clique_predict(data, intervals: int, threshold: float):
    """
    Chạy CLIQUE và trả về mảng nhãn cluster (0-indexed, -1 = noise).
    """
    if not CLIQUE_AVAILABLE:
        return None

    # CLIQUE yêu cầu dữ liệu trong [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data).tolist()

    clique_instance = clique(data_scaled, intervals, threshold)
    clique_instance.process()
    clusters = clique_instance.get_clusters()

    labels = np.full(len(data), -1, dtype=int)
    for cluster_id, indices in enumerate(clusters):
        for idx in indices:
            labels[idx] = cluster_id
    return labels


def grid_search(data, true_labels, algorithm: str, sigma: float = 1.0):
    """
    Chạy grid search cho GCBD gốc hoặc GCBD Gaussian.
    """
    keys   = list(GRID_PARAMS.keys())
    values = list(GRID_PARAMS.values())
    results = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if algorithm == "original":
                    pred = GCBD(
                        data,
                        no_grid    = params["no_grid"],
                        percentile = params["percentile"],
                        max_iters  = params["max_iters"],
                    )
                else:
                    pred = GKGCBD(
                        data,
                        no_grid    = params["no_grid"],
                        percentile = params["percentile"],
                        max_iters  = params["max_iters"],
                        sigma      = sigma,
                    )
        except Exception:
            continue

        ami, fmi = compute_metrics(true_labels, pred)
        if ami is None:
            continue

        label = "GCBD_original" if algorithm == "original" else f"GCBD_gaussian_s{sigma}"
        results.append({
            "algorithm" : label,
            "no_grid"   : params["no_grid"],
            "percentile": params["percentile"],
            "max_iters" : params["max_iters"],
            "sigma"     : "-" if algorithm == "original" else sigma,
            "n_clusters": int(np.unique(pred[pred > 0]).shape[0]),
            "AMI"       : ami,
            "FMI"       : fmi,
            "_pred"     : pred,   # lưu để vẽ hình
        })

    results.sort(key=lambda x: x["AMI"], reverse=True)
    return results


def grid_search_clique(data, true_labels):
    """
    Chạy grid search cho CLIQUE.
    """
    if not CLIQUE_AVAILABLE:
        return []

    results = []
    intervals_list = CLIQUE_PARAMS["intervals"]
    threshold_list = CLIQUE_PARAMS["threshold"]

    total = len(intervals_list) * len(threshold_list)
    done  = 0

    for intervals in intervals_list:
        for threshold in threshold_list:
            done += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = clique_predict(data, intervals, threshold)
                if pred is None:
                    continue
            except Exception:
                continue

            ami, fmi = compute_metrics(true_labels, pred)
            if ami is None:
                continue

            n_clusters = int(np.unique(pred[pred >= 0]).shape[0])
            results.append({
                "algorithm" : "CLIQUE",
                "intervals" : intervals,
                "threshold" : threshold,
                "no_grid"   : "-",
                "percentile": "-",
                "max_iters" : "-",
                "sigma"     : "-",
                "n_clusters": n_clusters,
                "AMI"       : ami,
                "FMI"       : fmi,
                "_pred"     : pred,
            })

    results.sort(key=lambda x: x["AMI"], reverse=True)
    return results


def labels_to_colors(labels):
    """Chuyển mảng nhãn thành danh sách màu."""
    colors = []
    for lbl in labels:
        if lbl < 0:
            colors.append(NOISE_COLOR)
        else:
            colors.append(CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)])
    return colors


def plot_clustering_result(data, true_labels,
                           pred_orig, pred_gauss, pred_clique,
                           save_path: str):
    """
    Vẽ 4 panel: Ground Truth | GCBD | GCBD Gaussian | CLIQUE
    Không có text chú thích nào.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                             facecolor="white",
                             gridspec_kw={"wspace": 0.04})

    panels = [
        (true_labels, true_labels),
        (pred_orig,   true_labels),
        (pred_gauss,  true_labels),
        (pred_clique, true_labels),
    ]

    for ax, (pred, _) in zip(axes, panels):
        colors = labels_to_colors(pred)
        ax.scatter(data[:, 0], data[:, 1],
                   c=colors, s=10, linewidths=0, alpha=0.85)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#cccccc")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)


def run_benchmark():
    os.makedirs(RESULT_DIR, exist_ok=True)

    arff_files = sorted(Path(DATASET_DIR).glob("*.arff"))
    if not arff_files:
        print(f"Không tìm thấy file .arff trong: {DATASET_DIR}")
        return

    n_combos_orig     = len(GRID_PARAMS["no_grid"]) * len(GRID_PARAMS["percentile"]) * len(GRID_PARAMS["max_iters"])
    n_combos_gaussian = n_combos_orig * len(SIGMA_VALUES)
    n_combos_clique   = len(CLIQUE_PARAMS["intervals"]) * len(CLIQUE_PARAMS["threshold"])

    print(f"Tìm thấy {len(arff_files)} file .arff")
    print(f"GCBD gốc     : {n_combos_orig} tổ hợp / dataset")
    print(f"GCBD Gaussian: {n_combos_gaussian} tổ hợp / dataset ({len(SIGMA_VALUES)} sigma × {n_combos_orig})")
    print(f"CLIQUE       : {n_combos_clique} tổ hợp / dataset (l=[5,50], c=[0,5])")
    print(f"Tổng lần chạy: {len(arff_files) * (n_combos_orig + n_combos_gaussian + n_combos_clique)}")
    print("─" * 65)

    all_rows     = []
    summary_rows = []

    for arff_path in arff_files:
        name = arff_path.stem
        print(f"\n[{name}]")

        try:
            data, true_labels = load_arff(str(arff_path))
        except Exception as e:
            print(f"  LỖI load: {e}")
            continue

        if data is None or data.shape[1] < 2:
            print("  bỏ qua (không đủ feature)")
            continue

        best_orig_pred   = None
        best_gauss_pred  = None
        best_clique_pred = None

        # ── GCBD gốc ─────────────────────────────────────────────────────────
        print(f"  GCBD gốc ({n_combos_orig} combos)...", end=" ", flush=True)
        orig_results = grid_search(data, true_labels, "original")
        if orig_results:
            best = orig_results[0]
            best_orig_pred = best.pop("_pred")
            print(f"best AMI={best['AMI']:.4f} "
                  f"(no_grid={best['no_grid']}, max_iters={best['max_iters']})")
            for r in orig_results:
                r.pop("_pred", None)
                all_rows.append({"dataset": name, **r})
            summary_rows.append({"dataset": name, **best})
        else:
            print("không có kết quả")

        # ── GCBD Gaussian — từng sigma ────────────────────────────────────────
        best_gauss_ami  = -999
        for sigma in SIGMA_VALUES:
            print(f"  GCBD Gaussian σ={sigma} ({n_combos_orig} combos)...", end=" ", flush=True)
            gauss_results = grid_search(data, true_labels, "gaussian", sigma)
            if gauss_results:
                best = gauss_results[0]
                _pred = best.pop("_pred")
                print(f"best AMI={best['AMI']:.4f} "
                      f"(no_grid={best['no_grid']}, max_iters={best['max_iters']})")
                for r in gauss_results:
                    r.pop("_pred", None)
                    all_rows.append({"dataset": name, **r})
                summary_rows.append({"dataset": name, **best})
                if best["AMI"] > best_gauss_ami:
                    best_gauss_ami  = best["AMI"]
                    best_gauss_pred = _pred
            else:
                print("không có kết quả")

        # ── CLIQUE ────────────────────────────────────────────────────────────
        if CLIQUE_AVAILABLE:
            print(f"  CLIQUE ({n_combos_clique} combos)...", end=" ", flush=True)
            clique_results = grid_search_clique(data, true_labels)
            if clique_results:
                best = clique_results[0]
                best_clique_pred = best.pop("_pred")
                print(f"best AMI={best['AMI']:.4f} "
                      f"(intervals={best['intervals']}, threshold={best['threshold']})")
                for r in clique_results:
                    r.pop("_pred", None)
                    all_rows.append({"dataset": name, **r})
                summary_rows.append({"dataset": name, **best})
            else:
                print("không có kết quả")
        else:
            print("  CLIQUE: bỏ qua (pyclustering chưa cài)")

        # ── Vẽ hình ───────────────────────────────────────────────────────────
        if (best_orig_pred is not None and
                best_gauss_pred is not None and
                best_clique_pred is not None):

            fig_path = os.path.join(RESULT_DIR, f"clustering_{name}.png")
            plot_clustering_result(
                data[:, :2],        # chỉ lấy 2 chiều đầu để vẽ
                true_labels,
                best_orig_pred,
                best_gauss_pred,
                best_clique_pred,
                fig_path,
            )
            print(f"  → Đã lưu hình: {fig_path}")

    # ── Xuất CSV ──────────────────────────────────────────────────────────────
    # Chuẩn hoá cột trước khi lưu (CLIQUE có thêm cột intervals, threshold)
    df_all     = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows)

    all_path     = os.path.join(RESULT_DIR, "benchmark_gaussian_results.csv")
    summary_path = os.path.join(RESULT_DIR, "benchmark_gaussian_summary.csv")
    df_all.to_csv(all_path,      index=False)
    df_summary.to_csv(summary_path, index=False)

    # ── Bảng tóm tắt ─────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("TỔNG KẾT SO SÁNH (best AMI mỗi dataset)")
    print("═" * 80)

    datasets   = df_summary["dataset"].unique()
    algo_names = [
        "GCBD_original",
        *[f"GCBD_gaussian_s{s}" for s in SIGMA_VALUES],
        "CLIQUE",
    ]

    header = f"{'Dataset':<22}" + "".join(f"{a:>22}" for a in algo_names)
    print(header)
    print("─" * (22 + 22 * len(algo_names)))

    wins = {a: 0 for a in algo_names}
    ties = 0

    for ds in sorted(datasets):
        sub  = df_summary[df_summary["dataset"] == ds]
        amis = {}
        for a in algo_names:
            row = sub[sub["algorithm"] == a]["AMI"].values
            amis[a] = row[0] if len(row) > 0 else None

        if any(v is None for v in amis.values()):
            continue

        best_ami = max(amis.values())
        winners  = [a for a, v in amis.items() if v >= best_ami - 0.005]

        if len(winners) > 1:
            ties += 1
        else:
            wins[winners[0]] += 1

        row_str = f"{ds:<22}"
        for a in algo_names:
            marker = " ✓" if a in winners and len(winners) == 1 else "  "
            row_str += f"{amis[a]:>20.4f}{marker}"
        print(row_str)

    print("─" * (22 + 22 * len(algo_names)))
    win_str = f"{'Wins':<22}" + "".join(f"{wins[a]:>22}" for a in algo_names)
    print(win_str)
    print(f"Ties: {ties}")
    print(f"\nFile đã lưu:\n  {all_path}\n  {summary_path}")
    print(f"  Hình phân cụm: {RESULT_DIR}/clustering_<dataset>.png")


if __name__ == "__main__":
    run_benchmark()