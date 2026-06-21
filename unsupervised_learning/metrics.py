"""
==============================================================
  metrics.py | 教師なし学習の評価指標
==============================================================
外部ライブラリ不使用。NumPy のみ。

クラスタリング指標:
  silhouette_score    - 凝集度と分離度のバランス (-1 ~ +1)
  davies_bouldin_score - クラスタ間/内距離比の平均 (小さいほど良い)
  adjusted_rand_index  - 真ラベルとの一致度 (-1 ~ +1)

異常検知指標:
  anomaly_precision   - 予測した異常のうち本当の異常の割合
  anomaly_recall      - 真の異常のうち正しく予測できた割合
  anomaly_f1          - precision と recall の調和平均

次元削減指標:
  reconstruction_error - 元データとの再構成誤差 (MSE)
  trustworthiness      - 低次元でも近傍関係が保たれているかの指標 (0 ~ 1)
"""

import numpy as np
from typing import Optional


# ─── クラスタリング指標 ──────────────────────────────────────

def silhouette_score(X: np.ndarray, labels: np.ndarray,
                     sample_size: Optional[int] = 200,
                     seed: int = 0) -> float:
    """
    シルエットスコア: クラスタの品質を -1 ~ +1 で評価する。

    各点 i について:
        a(i) = 同クラスタ内の他点との平均距離   (凝集度: 小さいほど良い)
        b(i) = 最近傍の異クラスタ点との平均距離 (分離度: 大きいほど良い)
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    全点のシルエット係数の平均を返す。
        s → +1 : 完璧なクラスタリング
        s →  0 : クラスタ境界上
        s → -1 : 誤ったクラスタに割り当てられている

    O(n²) のため sample_size でサブサンプリングを推奨。

    Note:
        クラスタ数が 1 または全点が同クラスタの場合は 0.0 を返す。
    """
    unique = np.unique(labels)
    unique = unique[unique >= 0]      # ノイズラベル (-1) を除外
    if len(unique) < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    n = len(X)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        X, labels = X[idx], labels[idx]

    n = len(X)
    scores = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            scores[i] = 0.0
            continue
        a = np.mean(np.linalg.norm(X[same] - X[i], axis=1))

        b_vals = []
        for c in unique:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.sum() == 0:
                continue
            b_vals.append(np.mean(np.linalg.norm(X[mask] - X[i], axis=1)))
        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)
        scores[i] = (b - a) / denom if denom > 0 else 0.0

    return float(np.mean(scores))


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies-Bouldin 指数: 値が小さいほどクラスタリングが良い。

    各クラスタ i について:
        s_i = クラスタ内の点から重心までの平均距離 (内部散布度)
        d_ij = クラスタ i と j の重心間距離

    DB = (1/K) Σ_i max_{j≠i} (s_i + s_j) / d_ij

    Note:
        クラスタ数が 1 の場合は 0.0 を返す。
    """
    unique = np.unique(labels)
    unique = unique[unique >= 0]
    K = len(unique)
    if K < 2:
        return 0.0

    centroids = np.array([X[labels == c].mean(axis=0) for c in unique])
    s = np.array([
        np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
        for i, c in enumerate(unique)
    ])

    db_values = []
    for i in range(K):
        ratios = []
        for j in range(K):
            if i == j:
                continue
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d > 0:
                ratios.append((s[i] + s[j]) / d)
        db_values.append(max(ratios) if ratios else 0.0)

    return float(np.mean(db_values))


def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    調整ランド指数 (ARI): 真ラベルと予測ラベルの一致度。

    ARI = (RI - E[RI]) / (max(RI) - E[RI])

    RI (ランド指数) は「同じクラスタに属すか否か」の
    全サンプルペアに対する一致率。ARI は偶然の一致を補正する。

        ARI = +1: 完全一致
        ARI =  0: ランダムと同等
        ARI < 0 : ランダムより悪い

    Note:
        ノイズラベル (-1) は y_true/y_pred どちらも除外して計算。
    """
    mask = (y_true >= 0) & (y_pred >= 0)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n == 0:
        return 0.0

    classes_true  = np.unique(y_true)
    classes_pred  = np.unique(y_pred)
    contingency   = np.zeros((len(classes_true), len(classes_pred)), dtype=int)
    for i, ct in enumerate(classes_true):
        for j, cp in enumerate(classes_pred):
            contingency[i, j] = np.sum((y_true == ct) & (y_pred == cp))

    def comb2(n): return n * (n - 1) // 2

    sum_comb_c = sum(comb2(contingency[i, j])
                     for i in range(len(classes_true))
                     for j in range(len(classes_pred)))
    sum_comb_row = sum(comb2(contingency[i].sum()) for i in range(len(classes_true)))
    sum_comb_col = sum(comb2(contingency[:, j].sum()) for j in range(len(classes_pred)))
    total_comb = comb2(n)

    expected = sum_comb_row * sum_comb_col / (total_comb + 1e-12)
    maximum  = (sum_comb_row + sum_comb_col) / 2

    if maximum - expected < 1e-12:
        return 1.0 if sum_comb_c == maximum else 0.0

    return float((sum_comb_c - expected) / (maximum - expected))


# ─── 異常検知指標 ────────────────────────────────────────────

def anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    異常検知の評価指標。

    Args:
        y_true: 真のラベル  (+1=正常, -1=異常)
        y_pred: 予測ラベル  (+1=正常, -1=異常)

    Returns:
        dict with keys: precision, recall, f1, accuracy
    """
    tp = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true ==  1) & (y_pred == -1)))
    fn = int(np.sum((y_true == -1) & (y_pred ==  1)))
    tn = int(np.sum((y_true ==  1) & (y_pred ==  1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# ─── 次元削減指標 ────────────────────────────────────────────

def reconstruction_error(X_orig: np.ndarray, X_reconstructed: np.ndarray) -> float:
    """
    再構成誤差 (MSE): 次元削減後に再構成したデータと元データの誤差。

    MSE = (1/n) Σ ||x_i - x̂_i||²

    値が小さいほど情報損失が少ない。
    PCA の場合、主成分数を増やすと MSE は単調減少する。
    """
    return float(np.mean(np.sum((X_orig - X_reconstructed) ** 2, axis=1)))


def trustworthiness(X_high: np.ndarray, X_low: np.ndarray,
                    n_neighbors: int = 5) -> float:
    """
    信頼度 (Trustworthiness): 低次元でも高次元の近傍関係が保たれているか。

    直観:
        低次元空間で近傍になった点が、高次元でも近傍だったか?
        違う点が近くに来た場合はペナルティを与える。

    値域: [0, 1]、1 に近いほど近傍構造を保存できている。

    T = 1 - (2 / (n * k * (2n - 3k - 1))) * Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)

    ここで r(i,j) は j が高次元での i の近傍ランク。
    U_k(i) は低次元で i の k 近傍だが高次元では k 近傍でない点の集合。

    O(n²) のため大規模データでは遅い。
    """
    n = len(X_high)
    k = n_neighbors

    def rank_matrix(X):
        dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        return np.argsort(np.argsort(dists, axis=1), axis=1)

    rank_high = rank_matrix(X_high)
    rank_low  = rank_matrix(X_low)

    total = 0.0
    for i in range(n):
        # 低次元での k 近傍 (自分自身 rank=0 は除く)
        nn_low = set(np.where(rank_low[i] <= k)[0]) - {i}
        for j in nn_low:
            r = rank_high[i, j]
            if r > k:
                total += r - k

    denom = n * k * (2 * n - 3 * k - 1) / 2
    return float(1 - total / denom) if denom > 0 else 1.0
