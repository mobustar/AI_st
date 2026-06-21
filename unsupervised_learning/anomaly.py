"""
==============================================================
  anomaly.py | 異常検知アルゴリズム (戦略パターン)
==============================================================
NumPy のみ。外部 ML ライブラリ不使用。

実装アルゴリズム:
  1. IsolationForest - ランダム分割による隔離に基づく外れ値検出
  2. LOF             - 局所外れ値因子 (密度比較による外れ値検出)

API:
    fit(X)                        → self  (正常データで学習)
    predict(X)                    → np.ndarray (+1=正常, -1=異常)
    score_samples(X)              → np.ndarray (異常スコア: 小さいほど異常)

Isolation Forest と LOF の違い:
  ┌────────────────┬───────────────────────────────────────┐
  │                │ Isolation Forest    │ LOF              │
  ├────────────────┼─────────────────────┼──────────────────┤
  │ 原理           │ 少ない分割で孤立   │ 近傍密度が低い  │
  │ グローバル/局所 │ グローバル外れ値  │ 局所外れ値      │
  │ パラメータ     │ 木の数・高さ       │ k (近傍数)      │
  │ 密度クラスタ混在│ 苦手               │ 得意            │
  └────────────────┴─────────────────────┴──────────────────┘
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAnomaly(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomaly": ...

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        スコアの下位 contamination 割合を異常 (-1) と判定。

        Args:
            contamination: 異常と見なす割合 (0 < contamination < 0.5)
        """
        scores = self.score_samples(X)
        threshold = np.percentile(scores, contamination * 100)
        return np.where(scores <= threshold, -1, 1).astype(int)


# ─── 1. Isolation Forest ─────────────────────────────────────
class IsolationForest(BaseAnomaly):
    """
    Isolation Forest (孤立フォレスト)。

    直観:
        外れ値は「珍しいデータ」なので、ランダム分割を繰り返すと
        他の点より早く孤立する (パスが短くなる)。
        正常データは多くの分割が必要。

    アルゴリズム:
        1. n_estimators 本の孤立木を構築
           - ランダムに特徴量を選び、その値の範囲でランダムに分割
           - max_samples 個のサブサンプルを使用
           - 深さ max_depth まで再帰的に分割
        2. 各点の平均パス長 h(x) を計算
        3. 正規化スコア: s(x) = 2^{-h(x)/c(n)}
           c(n) = 2 * H(n-1) - 2(n-1)/n  (平均パス長の期待値)
           H(k) = ln(k) + 0.5772...  (調和数)

        s → 1: 短いパス → 外れ値
        s → 0: 長いパス → 正常
        s ≈ 0.5: 判断困難

    Note:
        score_samples は -s(x) を返す (小さいほど外れ値 → 統一的 API)。

    Attributes:
        n_estimators : 孤立木の本数
        max_samples  : 各木が使うサブサンプル数 (None なら min(256, n))
        max_depth    : 木の最大深さ (None なら ceil(log2(max_samples)))
        seed         : 乱数シード
    """

    def __init__(self, n_estimators: int = 100, max_samples=None,
                 max_depth=None, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_samples  = max_samples
        self.max_depth    = max_depth
        self.seed         = seed

    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int) -> dict:
        """再帰的に孤立木を構築"""
        n, d = X.shape
        if n <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": n}

        feat = int(self._rng.integers(d))
        lo, hi = X[:, feat].min(), X[:, feat].max()
        if lo >= hi:
            return {"type": "leaf", "size": n}

        split = self._rng.uniform(lo, hi)
        left_mask  = X[:, feat] < split
        right_mask = ~left_mask
        return {
            "type":  "node",
            "feat":  feat,
            "split": split,
            "left":  self._build_tree(X[left_mask],  depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth),
        }

    def _path_length(self, x: np.ndarray, node: dict, depth: int) -> float:
        """点 x の孤立木での平均パス長を計算"""
        if node["type"] == "leaf":
            return depth + self._c(node["size"])
        if x[node["feat"]] < node["split"]:
            return self._path_length(x, node["left"],  depth + 1)
        else:
            return self._path_length(x, node["right"], depth + 1)

    @staticmethod
    def _c(n: int) -> float:
        """n 点の BST の平均パス長"""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def fit(self, X: np.ndarray) -> "IsolationForest":
        self._rng = np.random.default_rng(self.seed)
        n = len(X)
        max_samples = self.max_samples or min(256, n)
        max_depth   = self.max_depth   or int(np.ceil(np.log2(max_samples)))
        self._c_n   = self._c(max_samples)

        self._trees = []
        for _ in range(self.n_estimators):
            idx = self._rng.choice(n, size=max_samples, replace=False)
            tree = self._build_tree(X[idx], 0, max_depth)
            self._trees.append(tree)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """異常スコア (-s(x)): 小さいほど外れ値"""
        scores = np.zeros(len(X))
        for i, x in enumerate(X):
            avg_depth = np.mean([self._path_length(x, t, 0) for t in self._trees])
            s = 2 ** (-avg_depth / self._c_n)
            scores[i] = -s     # 符号反転: 小さいほど外れ値
        return scores


# ─── 2. LOF (Local Outlier Factor) ───────────────────────────
class LOF(BaseAnomaly):
    """
    LOF (Local Outlier Factor: 局所外れ値因子)。

    直観:
        自分の周囲の密度に比べて、近傍の密度が著しく高い点は外れ値。
        "局所"性がポイント: 密度の異なる複数クラスタが混在しても検出できる。

    アルゴリズム:
        1. k-最近傍距離 reach-dist の計算:
               reach-dist_k(x, o) = max(k-dist(o), dist(x, o))
               (o の k 近傍距離よりも短い距離は k 近傍距離で打ち切る)

        2. 局所到達可能密度 lrd:
               lrd_k(x) = 1 / (mean_{o ∈ N_k(x)} reach-dist_k(x, o))
               (小さいほど点 x の周囲は疎 = 外れ値候補)

        3. LOF スコア:
               LOF_k(x) = mean_{o ∈ N_k(x)} (lrd_k(o) / lrd_k(x))
               LOF > 1  : 周囲より密度が低い → 外れ値
               LOF ≈ 1  : 周囲と密度が同程度 → 正常

    Attributes:
        n_neighbors : k の値 (近傍数)
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray) -> "LOF":
        self._X_train = X.copy()
        n = len(X)
        k = min(self.n_neighbors, n - 1)

        # ペア距離行列
        D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(D, np.inf)

        # k-近傍と k-距離
        knn_idx = np.argsort(D, axis=1)[:, :k]      # (n, k)
        k_dists = D[np.arange(n), knn_idx[:, k - 1]] # (n,)  k 番目の距離

        # reach-dist_k(x, o) = max(k-dist(o), dist(x, o))
        reach = np.maximum(k_dists[knn_idx], D[np.arange(n)[:, None], knn_idx])

        # lrd: 局所到達可能密度
        lrd = 1.0 / (reach.mean(axis=1) + 1e-12)

        # LOF スコア = 近傍の lrd / 自分の lrd の平均
        lof = np.array([
            lrd[knn_idx[i]].mean() / (lrd[i] + 1e-12)
            for i in range(n)
        ])

        self._k       = k
        self._knn_idx = knn_idx
        self._k_dists = k_dists
        self._lrd     = lrd
        self._lof_train = lof
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        新データの LOF スコアを計算して返す。
        -(LOF score) を返す (小さいほど外れ値)。

        訓練データを参照して近傍・lrd を計算する。
        """
        X_tr = self._X_train
        k    = self._k

        D = np.sqrt(np.sum((X[:, None, :] - X_tr[None, :, :]) ** 2, axis=2))
        # (n_test, k)
        knn_idx_test = np.argsort(D, axis=1)[:, :k]
        k_dists_test = D[np.arange(len(X)), knn_idx_test[:, k - 1]]

        reach_test = np.maximum(
            self._k_dists[knn_idx_test],
            D[np.arange(len(X))[:, None], knn_idx_test]
        )
        lrd_test = 1.0 / (reach_test.mean(axis=1) + 1e-12)

        lof_test = np.array([
            self._lrd[knn_idx_test[i]].mean() / (lrd_test[i] + 1e-12)
            for i in range(len(X))
        ])
        return -lof_test   # 符号反転: 小さいほど外れ値


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_anomaly(name: str, **kwargs) -> BaseAnomaly:
    """
    Args:
        name: "isolation_forest" | "lof"
        **kwargs: 各クラスのコンストラクタ引数
    """
    table = {
        "isolation_forest": IsolationForest,
        "lof":              LOF,
    }
    if name not in table:
        raise ValueError(f"unknown anomaly detector: {name}. choices: {list(table)}")
    return table[name](**kwargs)
