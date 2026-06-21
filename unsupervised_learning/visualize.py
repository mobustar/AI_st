"""
==============================================================
  visualize.py | ASCII 可視化ユーティリティ
==============================================================
ターミナルだけで動く散布図・損失曲線・ヒートマップ。
matplotlib 不使用。NumPy のみ。

提供する関数:
  scatter_2d         - 2D 散布図 (ラベルごとに文字を変える)
  scatter_compare    - 真ラベルと予測ラベルを横並び比較
  loss_curve         - 損失/スコアの推移を折れ線グラフで表示
  cluster_summary    - クラスタリング結果のサマリを表示
  anomaly_summary    - 異常検知結果のサマリを表示
"""

import numpy as np
from typing import Optional, List


# ASCII 文字セット: ラベル 0,1,2,...,-1(ノイズ) に対応
_MARKERS = list("O+X#*@$&%") + ["?"]   # ラベル 0-8, 9以上は ?
_NOISE_MARKER = "."                      # DBSCANノイズ (-1)
_COLORS = ""   # ターミナルカラーなし (ポータブル)


def _normalize_to_grid(X: np.ndarray, width: int, height: int):
    """データ座標 → グリッドインデックスに変換"""
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    # パディング
    margin_x, margin_y = x_range * 0.05, y_range * 0.05
    x_min -= margin_x; x_max += margin_x
    y_min -= margin_y; y_max += margin_y
    col = ((X[:, 0] - x_min) / (x_max - x_min) * (width  - 1)).astype(int)
    row = ((X[:, 1] - y_min) / (y_max - y_min) * (height - 1)).astype(int)
    # 反転: 上がY大
    row = height - 1 - row
    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)
    return row, col


def scatter_2d(X: np.ndarray, labels: np.ndarray,
               title: str = "", width: int = 60, height: int = 20,
               indent: str = "  ") -> None:
    """
    2D 散布図を ASCII で表示。

    Args:
        X      : shape (n, 2) の座標
        labels : shape (n,)  の整数ラベル (-1 = ノイズ)
        title  : タイトル文字列
        width  : グリッド幅 (文字数)
        height : グリッド高 (行数)
        indent : 各行の先頭インデント
    """
    grid = [["." for _ in range(width)] for _ in range(height)]
    rows, cols = _normalize_to_grid(X, width, height)

    unique = sorted(set(labels))
    for i, (r, c, lbl) in enumerate(zip(rows, cols, labels)):
        if lbl < 0:
            grid[r][c] = _NOISE_MARKER
        else:
            idx = unique.index(lbl)
            grid[r][c] = _MARKERS[idx % len(_MARKERS)]

    if title:
        print(f"{indent}{'─' * width}")
        print(f"{indent} {title}")
        print(f"{indent}{'─' * width}")
    else:
        print(f"{indent}{'─' * width}")

    for row in grid:
        print(indent + "".join(row))

    print(f"{indent}{'─' * width}")

    # 凡例
    legend_parts = []
    for lbl in unique:
        if lbl < 0:
            legend_parts.append(f". =ノイズ")
        else:
            idx = unique.index(lbl)
            m = _MARKERS[idx % len(_MARKERS)]
            legend_parts.append(f"{m}={lbl}")
    print(f"{indent}凡例: " + "  ".join(legend_parts))


def scatter_compare(X: np.ndarray,
                    labels_true: np.ndarray, labels_pred: np.ndarray,
                    title_true: str = "真ラベル", title_pred: str = "予測ラベル",
                    width: int = 30, height: int = 15,
                    indent: str = "  ") -> None:
    """真ラベルと予測ラベルを左右に並べて表示"""

    def make_grid(labels):
        g = [["." for _ in range(width)] for _ in range(height)]
        rows, cols = _normalize_to_grid(X, width, height)
        unique = sorted(set(labels))
        for r, c, lbl in zip(rows, cols, labels):
            if lbl < 0:
                g[r][c] = _NOISE_MARKER
            else:
                idx = unique.index(lbl)
                g[r][c] = _MARKERS[idx % len(_MARKERS)]
        return g

    g_true = make_grid(labels_true)
    g_pred = make_grid(labels_pred)

    sep = "  │  "
    header = indent + f"{'─'*width}{sep[2:]}{'─'*width}"
    print(header)
    t_padded = title_true.center(width)
    p_padded = title_pred.center(width)
    print(f"{indent}{t_padded}{sep}{p_padded}")
    print(header)
    for r_t, r_p in zip(g_true, g_pred):
        print(f"{indent}{''.join(r_t)}{sep}{''.join(r_p)}")
    print(header)


def loss_curve(values: List[float], title: str = "Loss",
               width: int = 60, height: int = 12,
               indent: str = "  ") -> None:
    """
    1D 数値列を折れ線グラフで表示。

    例: VAE の ELBO 損失推移など。
    """
    if not values:
        return
    arr = np.array(values, dtype=float)
    v_min, v_max = arr.min(), arr.max()
    v_range = v_max - v_min if v_max > v_min else 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]
    n = len(arr)

    prev_row = None
    for i, v in enumerate(arr):
        col = int(i / (n - 1) * (width - 1)) if n > 1 else 0
        row = int((1 - (v - v_min) / v_range) * (height - 1))
        row = min(max(row, 0), height - 1)
        grid[row][col] = "●"
        if prev_row is not None and abs(row - prev_row) > 1:
            # 縦方向を繋ぐ
            for r in range(min(row, prev_row) + 1, max(row, prev_row)):
                grid[r][col] = "│"
        prev_row = row

    # 枠
    print(f"{indent}{'─' * (width + 4)}")
    print(f"{indent} {title} ({len(arr)} iters,  min={v_min:.4f}, final={arr[-1]:.4f})")
    print(f"{indent}{'─' * (width + 4)}")
    labels_y = [f"{v_max:.2f}", "", f"{(v_max+v_min)/2:.2f}", "", f"{v_min:.2f}"]
    label_step = height // (len(labels_y) - 1)
    for ri, row in enumerate(grid):
        lbl_idx = ri // label_step if label_step > 0 else 0
        lbl = labels_y[lbl_idx] if lbl_idx < len(labels_y) else ""
        print(f"{indent}{lbl:>6} │{''.join(row)}│")
    print(f"{indent}{'─' * (width + 4)}")


def cluster_summary(name: str, labels: np.ndarray,
                    silhouette: float, davies_bouldin: float,
                    ari: Optional[float] = None,
                    indent: str = "  ") -> None:
    """クラスタリング結果の数値サマリを整形表示"""
    unique = np.unique(labels)
    n_clusters = int((unique >= 0).sum())
    n_noise    = int((labels == -1).sum())

    print(f"{indent}┌─ {name}")
    print(f"{indent}│  クラスタ数  : {n_clusters}"
          + (f"  (ノイズ点: {n_noise})" if n_noise > 0 else ""))
    print(f"{indent}│  シルエット  : {silhouette:+.4f}  "
          "(+1=完璧 / 0=境界 / -1=誤割当)")
    print(f"{indent}│  Davies-Bouldin: {davies_bouldin:.4f}  (小さいほど良い)")
    if ari is not None:
        print(f"{indent}│  ARI (真ラベルとの一致): {ari:+.4f}")
    print(f"{indent}└─")


def anomaly_summary(name: str, metrics: dict, n_total: int,
                    n_outliers_pred: int, n_outliers_true: int,
                    indent: str = "  ") -> None:
    """異常検知結果のサマリを整形表示"""
    print(f"{indent}┌─ {name}")
    print(f"{indent}│  実際の外れ値  : {n_outliers_true} / {n_total}")
    print(f"{indent}│  予測した外れ値: {n_outliers_pred} / {n_total}")
    print(f"{indent}│  Precision     : {metrics['precision']:.4f}  "
          "(予測した異常のうち本物の異常の割合)")
    print(f"{indent}│  Recall        : {metrics['recall']:.4f}  "
          "(本物の異常のうち検出できた割合)")
    print(f"{indent}│  F1            : {metrics['f1']:.4f}")
    print(f"{indent}│  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"{indent}└─")
