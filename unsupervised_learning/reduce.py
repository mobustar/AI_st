"""
==============================================================
  reduce.py | 次元削減アルゴリズム (戦略パターン)
==============================================================
NumPy のみ。外部 ML ライブラリ不使用。

実装アルゴリズム:
  1. PCA   - 主成分分析 (線形、SVD ベース)
  2. TSNE  - t-SNE (非線形、O(n²) 厳密実装)

どちらも fit_transform(X) → X_low (shape: n × n_components) を返す。

選択ガイド:
  PCA  : 線形構造の保存・再構成・前処理向き
         再構成可能 (inverse_transform あり)
         大規模データに対応 (SVD は O(n*d²))

  t-SNE: 非線形構造 (マニフォールド) の可視化向き
         2D/3D への視覚的埋め込みに特化
         再構成不可 (out-of-sample extension なし)
         O(n²) のため n ≲ 1000 推奨
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseReduce(ABC):
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


# ─── 1. PCA (主成分分析) ─────────────────────────────────────
class PCA(BaseReduce):
    """
    主成分分析 (Principal Component Analysis)。

    数学的背景:
        データ行列 X (n×d) を中心化 (平均を引く) した後、
        共分散行列 C = X^T X / n を固有値分解する。
        固有値が大きい方向 (主成分) が分散の大きい方向。

        実装では SVD (特異値分解) を使う:
            X_centered = U Σ V^T
        V の列が主成分方向 (右特異ベクトル)、
        対応する特異値 σ_i² / n が分散。

    射影:
        X_low = X_centered @ V[:, :n_components]

    再構成:
        X_reconstructed = X_low @ V[:, :n_components].T + mean

    分散説明率 (explained_variance_ratio_):
        各主成分が全分散のうち何割を説明するか。
        e.g., [0.85, 0.12] → 第 1・2 主成分で 97% を説明

    Attributes:
        n_components : 削減後の次元数
        components_  : 主成分ベクトル shape (n_components, d)
        explained_variance_ratio_ : 各主成分の分散説明率
        mean_        : 学習データの平均
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # 特異値分解: X_c = U Σ V^T
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vt[:self.n_components]           # (n_comp, d)

        # 分散説明率
        var = (s ** 2) / (len(X) - 1)
        self.explained_variance_ratio_ = var[:self.n_components] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_low: np.ndarray) -> np.ndarray:
        """低次元データを元の次元数に再構成"""
        return X_low @ self.components_ + self.mean_


# ─── 2. t-SNE (t-distributed Stochastic Neighbor Embedding) ─
class TSNE(BaseReduce):
    """
    t-SNE: 高次元データの 2D/3D 可視化アルゴリズム。

    直観:
        高次元での「近さ」と低次元での「近さ」を揃えるように
        低次元座標を最適化する。

    数学:
        ① 高次元での類似度 p_ij (対称化ガウス確率):
            p_j|i = exp(-||x_i - x_j||² / 2σ_i²) / Σ_{k≠i}(...)
            p_ij  = (p_j|i + p_i|j) / (2n)

        ② 低次元での類似度 q_ij (自由度 1 の t 分布):
            q_ij = (1 + ||y_i - y_j||²)^{-1} / Σ_{k≠l}(...)

        ③ KL ダイバージェンスを勾配降下で最小化:
            C = Σ_{ij} p_ij log(p_ij / q_ij)

    ガウスカーネルの幅 σ_i:
        各点の σ_i をバイナリサーチで自動調整。
        perplexity ≈ 2^{エントロピー} になるように σ_i を決定。
        perplexity は「実効的な近傍数」(通常 5〜50)。

    t 分布を使う理由:
        ガウス分布は裾が薄いため低次元では離れた点を
        引き寄せてしまう ("crowding problem")。
        t 分布は裾が厚いため、クラスタ間のギャップを自然に広げる。

    Attributes:
        n_components : 削減後の次元数 (通常 2)
        perplexity   : 実効的な近傍数 (5〜50 推奨)
        n_iter       : 勾配降下の反復回数
        lr           : 学習率
        early_exaggeration : 序盤の p_ij スケーリング係数
        seed         : 乱数シード

    Note:
        O(n²) の実装。n ≲ 500 推奨。
        大規模データには Barnes-Hut 近似 (O(n log n)) を使う。
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 n_iter: int = 1000, lr: float = 200.0,
                 early_exaggeration: float = 12.0,
                 momentum: float = 0.8, seed: int = 0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.lr = lr
        self.early_exaggeration = early_exaggeration
        self.momentum = momentum
        self.seed = seed

    def _compute_pairwise_dists(self, X: np.ndarray) -> np.ndarray:
        """||x_i - x_j||² を一括計算"""
        sum_sq = np.sum(X ** 2, axis=1)
        D2 = sum_sq[:, None] + sum_sq[None, :] - 2 * X @ X.T
        np.clip(D2, 0, None, out=D2)
        np.fill_diagonal(D2, 0)
        return D2

    def _compute_sigma(self, D2: np.ndarray) -> np.ndarray:
        """各点の σ を perplexity に合うようバイナリサーチ"""
        n = len(D2)
        sigmas = np.ones(n)
        target_entropy = np.log2(self.perplexity)

        for i in range(n):
            lo, hi = 1e-5, 1e5
            for _ in range(50):
                sigma = (lo + hi) / 2
                # perplexity 計算
                d = D2[i].copy()
                d[i] = np.inf
                p = np.exp(-d / (2 * sigma ** 2))
                p_sum = p.sum()
                if p_sum < 1e-12:
                    break
                p /= p_sum
                # エントロピー H = -Σ p log2 p
                p_nz = p[p > 1e-12]
                H = -np.sum(p_nz * np.log2(p_nz))
                if H < target_entropy:
                    lo = sigma
                else:
                    hi = sigma
            sigmas[i] = (lo + hi) / 2

        return sigmas

    def _compute_p(self, D2: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """対称化条件付き確率行列 P を計算"""
        n = len(D2)
        P = np.zeros((n, n))
        for i in range(n):
            d = D2[i].copy()
            d[i] = np.inf
            p_cond = np.exp(-d / (2 * sigmas[i] ** 2))
            p_cond /= p_cond.sum() + 1e-12
            P[i] = p_cond

        # 対称化: P_ij = (P_j|i + P_i|j) / 2n
        P = (P + P.T) / (2 * n)
        np.clip(P, 1e-12, None, out=P)
        return P

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = len(X)

        # 初期 PCA で低次元初期解 (収束を安定化)
        pca = PCA(n_components=self.n_components)
        Y = pca.fit_transform(X) * 1e-4

        D2_high = self._compute_pairwise_dists(X)
        sigmas  = self._compute_sigma(D2_high)
        P       = self._compute_p(D2_high, sigmas)

        # 勾配降下 (モーメンタム付き)
        Y_old = Y.copy()
        exagg_end = 250   # early exaggeration を終わらせる反復数

        for it in range(self.n_iter):
            # 低次元での q_ij (t 分布)
            D2_low = self._compute_pairwise_dists(Y)
            Q_num = 1.0 / (1.0 + D2_low)     # 非正規化
            np.fill_diagonal(Q_num, 0)
            Q = Q_num / (Q_num.sum() + 1e-12)
            np.clip(Q, 1e-12, None, out=Q)

            # Early exaggeration
            p_eff = P * self.early_exaggeration if it < exagg_end else P

            # KL 勾配: dC/dY_i = 4 Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i-y_j||²)^{-1}
            PQ = (p_eff - Q) * Q_num          # (n, n)
            grad = 4 * (PQ[:, :, None] * (Y[:, None, :] - Y[None, :, :])).sum(axis=1)

            # モーメンタム更新
            mom = 0.5 if it < 250 else self.momentum
            Y_new = Y - self.lr * grad + mom * (Y - Y_old)
            # 中心化 (ドリフト防止)
            Y_new -= Y_new.mean(axis=0)
            Y_old, Y = Y, Y_new

        self.embedding_ = Y
        return Y


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_reducer(name: str, **kwargs) -> BaseReduce:
    """
    Args:
        name: "pca" | "tsne"
        **kwargs: 各クラスのコンストラクタ引数
    """
    table = {"pca": PCA, "tsne": TSNE}
    if name not in table:
        raise ValueError(f"unknown reducer: {name}. choices: {list(table)}")
    return table[name](**kwargs)
