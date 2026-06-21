"""
==============================================================
  cluster.py | クラスタリングアルゴリズム (戦略パターン)
==============================================================
NumPy のみ。外部 ML ライブラリ不使用。

実装アルゴリズム:
  1. KMeans              - K-means++ 初期化 + 収束まで反復
  2. DBSCAN              - 密度ベースクラスタリング (ノイズ点 = -1)
  3. AgglomerativeClustering - 凝集型階層クラスタリング (Ward 連結)

すべてのクラスは fit(X) → self を返し、
  labels_  : shape (n,) のクラスタラベル (-1 = ノイズ)
  n_clusters_: 発見されたクラスタ数
を属性として持つ。

アルゴリズム選択ガイド:
  ┌──────────────────┬────────────────────────────────────────┐
  │ アルゴリズム      │ 向いている形状 / 特徴                   │
  ├──────────────────┼────────────────────────────────────────┤
  │ K-means          │ 球状・等分散クラスタ、K を事前指定       │
  │ DBSCAN           │ 任意形状、K 不要、ノイズを自動除外       │
  │ Agglomerative    │ 階層構造、Ward はコンパクトなクラスタ    │
  └──────────────────┴────────────────────────────────────────┘
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseCluster(ABC):
    labels_: np.ndarray
    n_clusters_: int

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseCluster": ...


# ─── 1. K-means (K-means++ 初期化) ───────────────────────────
class KMeans(BaseCluster):
    """
    K-means クラスタリング。

    アルゴリズム:
        1. K-means++ で初期重心を選択 (収束を安定化)
        2. 割り当て: 各点を最近傍重心に割り当てる
        3. 更新: 割り当て結果から重心を再計算
        4. 収束条件 (重心移動 < tol) まで 2-3 を繰り返す
        5. n_init 回試行して最良の慣性 (inertia) を選択

    K-means++ 初期化:
        最初の重心をランダムに選び、以後は既存重心からの
        距離²に比例した確率で次の重心を選ぶ。
        この確率的な「遠い点優先」で局所解のリスクを低減。

    慣性 (Inertia):
        各点と割り当て重心との距離²の総和。
        最小化が目標。

    Attributes:
        k        : クラスタ数
        max_iter : 最大反復回数
        tol      : 重心移動の収束閾値
        n_init   : 試行回数 (最良のものを選択)
        seed     : 乱数シード
        centroids_: 最終重心 shape (k, d)
        inertia_  : 最小慣性
    """

    def __init__(self, k: int = 3, max_iter: int = 300,
                 tol: float = 1e-4, n_init: int = 10, seed: int = 0):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.seed = seed

    def _init_centroids_pp(self, X: np.ndarray, rng) -> np.ndarray:
        """K-means++ 初期重心選択"""
        n = len(X)
        idx = rng.integers(n)
        centroids = [X[idx]]
        for _ in range(self.k - 1):
            # 各点から既存重心への最短距離²
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids.append(X[idx])
        return np.array(centroids)

    def _run_once(self, X: np.ndarray, rng) -> tuple:
        """1 回の K-means 実行。(labels, centroids, inertia) を返す"""
        centroids = self._init_centroids_pp(X, rng)
        labels = np.zeros(len(X), dtype=int)

        for _ in range(self.max_iter):
            # 割り当てステップ
            dists = np.array([
                np.sum((X - c) ** 2, axis=1) for c in centroids
            ])                                        # (k, n)
            new_labels = dists.argmin(axis=0)

            # 更新ステップ
            new_centroids = np.array([
                X[new_labels == j].mean(axis=0)
                if (new_labels == j).any()
                else centroids[j]
                for j in range(self.k)
            ])

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            labels, centroids = new_labels, new_centroids
            if shift < self.tol:
                break

        inertia = float(sum(
            np.sum((X[labels == j] - centroids[j]) ** 2)
            for j in range(self.k)
            if (labels == j).any()
        ))
        return labels, centroids, inertia

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.seed)
        best_labels, best_centroids, best_inertia = None, None, float("inf")
        for _ in range(self.n_init):
            labels, centroids, inertia = self._run_once(X, rng)
            if inertia < best_inertia:
                best_labels, best_centroids, best_inertia = labels, centroids, inertia
        self.labels_    = best_labels
        self.centroids_ = best_centroids
        self.inertia_   = best_inertia
        self.n_clusters_ = self.k
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """学習済み重心を使って新データを分類"""
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids_])
        return dists.argmin(axis=0)


# ─── 2. DBSCAN ───────────────────────────────────────────────
class DBSCAN(BaseCluster):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)。

    アルゴリズム:
        1. 各点の ε-近傍を列挙
        2. 近傍点数 ≥ min_samples の点を「コア点」と定義
        3. コア点から幅優先探索 (BFS) でクラスタを拡張
        4. いずれのクラスタにも属さない点を「ノイズ」(ラベル = -1) とする

    パラメータ選択のヒント:
        ε   : k-NN 距離のエルボーをプロットして選ぶ
        min_samples: 通常 2*d (d=次元数) が出発点

    K-means との違い:
        - K を指定不要 (密度から自動決定)
        - 任意形状クラスタを検出できる
        - ノイズ点を明示的に除外する
        - 密度が均一でないデータには eps 選択が難しい

    Attributes:
        eps        : 近傍半径
        min_samples: コア点判定の最小近傍点数 (自分自身を含む)
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: np.ndarray) -> "DBSCAN":
        n = len(X)
        labels = np.full(n, -2, dtype=int)  # -2 = 未処理
        cluster_id = 0

        # ε-近傍を事前計算 (O(n²) / 小規模データ向け)
        dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
        neighbors = [np.where(dists[i] <= self.eps)[0].tolist() for i in range(n)]

        def expand_cluster(seed_idx: int, cid: int):
            """BFS でクラスタを拡張"""
            queue = list(neighbors[seed_idx])
            labels[seed_idx] = cid
            qi = 0
            while qi < len(queue):
                q = queue[qi]; qi += 1
                if labels[q] == -2:       # 未処理
                    labels[q] = cid
                    if len(neighbors[q]) >= self.min_samples:  # コア点なら拡張
                        for nb in neighbors[q]:
                            if labels[nb] == -2:
                                queue.append(nb)
                elif labels[q] == -1:     # ノイズ → 境界点に変更
                    labels[q] = cid

        for i in range(n):
            if labels[i] != -2:
                continue
            if len(neighbors[i]) < self.min_samples:
                labels[i] = -1     # ノイズ点
            else:
                expand_cluster(i, cluster_id)
                cluster_id += 1

        self.labels_     = labels
        self.n_clusters_ = cluster_id
        return self


# ─── 3. AgglomerativeClustering (Ward 連結) ──────────────────
class AgglomerativeClustering(BaseCluster):
    """
    凝集型階層クラスタリング (Ward 連結法)。

    アルゴリズム:
        1. 各点を独立したクラスタとして開始
        2. 最も「マージコスト」が小さいクラスタペアを統合
        3. k 個になるまで繰り返す

    Ward 連結:
        2 クラスタをマージしたときの SSE (残差平方和) の増分を
        コストとして使う。
        SSE増分 = (n_A * n_B) / (n_A + n_B) * ||μ_A - μ_B||²

        Ward 法はコンパクトで等径なクラスタを好む。
        K-means と同様に凸形状に強く、複雑な形状には弱い。

    計算量: O(n³) — 大規模データには不向き (n ≲ 1000 推奨)

    Attributes:
        n_clusters: 最終クラスタ数
    """

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        n = len(X)
        # 各クラスタを {インデックスセット} で管理
        clusters = [{i} for i in range(n)]
        # ラベル配列
        labels = np.arange(n, dtype=int)

        def centroid(c_set):
            return X[list(c_set)].mean(axis=0)

        def ward_cost(A: set, B: set) -> float:
            nA, nB = len(A), len(B)
            cA, cB = centroid(A), centroid(B)
            return nA * nB / (nA + nB) * np.sum((cA - cB) ** 2)

        while len(clusters) > self.n_clusters:
            m = len(clusters)
            best_cost = float("inf")
            best_i, best_j = 0, 1
            for i in range(m):
                for j in range(i + 1, m):
                    cost = ward_cost(clusters[i], clusters[j])
                    if cost < best_cost:
                        best_cost, best_i, best_j = cost, i, j

            # マージ (j を i に統合)
            merged = clusters[best_i] | clusters[best_j]
            clusters = [c for k, c in enumerate(clusters)
                        if k != best_i and k != best_j]
            clusters.append(merged)

        # ラベル付け
        for cid, c_set in enumerate(clusters):
            for idx in c_set:
                labels[idx] = cid

        self.labels_     = labels
        self.n_clusters_ = self.n_clusters
        return self


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_cluster(name: str, **kwargs) -> BaseCluster:
    """
    Args:
        name: "kmeans" | "dbscan" | "agglomerative"
        **kwargs: 各クラスのコンストラクタ引数
    """
    table = {
        "kmeans":        KMeans,
        "dbscan":        DBSCAN,
        "agglomerative": AgglomerativeClustering,
    }
    if name not in table:
        raise ValueError(f"unknown cluster: {name}. choices: {list(table)}")
    return table[name](**kwargs)
