"""
==============================================================
  classifier.py | 多クラス分類器 (戦略パターン)
==============================================================
物体認識 (3 クラス: circle / square / triangle) 向けの分類器。
7 種を実装。

選択アルゴリズム:
  1. KNNClassifier           - 学習不要、コサイン距離で k 近傍多数決
  2. SoftmaxRegression       - 多項ロジスティック回帰 (softmax)
  3. OvRLinearSVM            - 線形 SVM の One-vs-Rest 拡張
  4. MLPClassifier           - 2 隠れ層 MLP (ReLU + He 初期化 + Adam)
  5. GaussianNBClassifier    - ガウス分布を仮定したナイーブベイズ
  6. LDAClassifier           - 線形判別分析 (クラス内分散で正規化)
  7. VotingEnsembleClassifier - Softmax + MLP + LDA の多数決 (Hard Voting)

精度の傾向 (データ拡張あり):
  ensemble ≈ hog+lda > hog+softmax > edge+softmax > raw+knn
  ※ データ拡張 (augment) で訓練データを 4 倍にすると全体的に精度が上がる

精度を最優先するための工夫:
  - SoftmaxRegression: log-sum-exp による数値安定化、L2 正則化
  - OvRLinearSVM: Pegasos 風の平均化 SGD で汎化性能を向上
  - MLPClassifier: He 初期化 + Adam + 早期停止
  - GaussianNBClassifier: var_smoothing でゼロ分散を防止
  - LDAClassifier: pinv による正則化で特異行列を回避
  - VotingEnsembleClassifier: 3 分類器の誤りを多数決で相殺
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y) -> "BaseClassifier": ...
    @abstractmethod
    def predict(self, X) -> np.ndarray: ...


# ─── 1. kNN (コサイン距離 / ユークリッド距離) ───────────────
class KNNClassifier(BaseClassifier):
    """
    k 近傍法。
    metric="cosine": 高次元疎特徴 (HOG/LBP) で精度が高い
    metric="euclidean": 連続値の特徴 (Raw) で素直
    """

    def __init__(self, k: int = 5, weights: str = "distance",
                 metric: str = "cosine"):
        self.k = k
        self.weights = weights
        self.metric  = metric

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        if self.metric == "cosine":
            n = np.linalg.norm(self.X_, axis=1, keepdims=True)
            n[n == 0] = 1.0
            self._Xn = self.X_ / n
        return self

    def _distances(self, X):
        if self.metric == "cosine":
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n[n == 0] = 1.0
            Xn = X / n
            return 1.0 - Xn @ self._Xn.T
        # Euclidean: ‖a-b‖² = ‖a‖² + ‖b‖² - 2 a·b
        a = (X ** 2).sum(axis=1, keepdims=True)
        b = (self.X_ ** 2).sum(axis=1)
        d2 = a + b - 2.0 * X @ self.X_.T
        return np.sqrt(np.maximum(d2, 0.0))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        D = self._distances(X)
        classes = np.unique(self.y_)
        preds = np.empty(X.shape[0], dtype=self.y_.dtype)
        for i in range(X.shape[0]):
            idx = np.argpartition(D[i], min(self.k, len(self.y_)-1))[: self.k]
            d   = D[i, idx]
            lbl = self.y_[idx]
            w   = 1.0 / (d + 1e-9) if self.weights == "distance" else np.ones_like(d)
            scores = np.array([w[lbl == c].sum() for c in classes])
            preds[i] = classes[scores.argmax()]
        return preds


# ─── 2. Multinomial Logistic Regression (softmax) ───────────
class SoftmaxRegression(BaseClassifier):
    """
    多項ロジスティック回帰。
        P(y=c|x) = exp(w_c · x + b_c) / Σ_k exp(w_k · x + b_k)
        L = -1/N Σ log P(y_i|x_i) + 0.5 λ ‖W‖²

    数値安定化: log-sum-exp トリックで softmax の overflow を回避
    最適化: フルバッチ勾配降下 + 早期停止 (収束安定性のため精度重視)
    """

    def __init__(self, lr: float = 0.5, reg: float = 1e-4,
                 epochs: int = 800, tol: float = 1e-7):
        self.lr     = lr
        self.reg    = reg
        self.epochs = epochs
        self.tol    = tol

    @staticmethod
    def _softmax_stable(Z):
        Z = Z - Z.max(axis=1, keepdims=True)        # オーバーフロー回避
        eZ = np.exp(Z)
        return eZ / eZ.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        N, d = X.shape

        # one-hot エンコーディング
        Y = np.zeros((N, K))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0

        self.W_ = np.zeros((d, K))
        self.b_ = np.zeros(K)
        prev_loss = np.inf

        for _ in range(self.epochs):
            P     = self._softmax_stable(X @ self.W_ + self.b_)
            err   = (P - Y) / N
            grad_W = X.T @ err + self.reg * self.W_
            grad_b = err.sum(axis=0)
            self.W_ -= self.lr * grad_W
            self.b_ -= self.lr * grad_b

            # 損失計算 (早期収束判定)
            log_p = np.log(P[np.arange(N), y.astype(int)] + 1e-12)
            loss = -log_p.mean() + 0.5 * self.reg * (self.W_ ** 2).sum()
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._softmax_stable(X @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ─── 3. One-vs-Rest Linear SVM ──────────────────────────────
class OvRLinearSVM(BaseClassifier):
    """
    各クラス c に対して「c か否か」の二値 SVM を 1 つずつ学習し、
    decision_function が最大のクラスを予測する。

    各二値 SVM は Pegasos 風 (平均化 SGD + 学習率 1/(λt))。
    """

    def __init__(self, reg: float = 1e-3, epochs: int = 50, seed: int = 0):
        self.reg    = reg
        self.epochs = epochs
        self.seed   = seed

    def _fit_binary(self, X, y_pm, rng):
        N, d = X.shape
        w, b = np.zeros(d), 0.0
        w_avg, b_avg = np.zeros(d), 0.0
        t = 0
        for _ in range(self.epochs):
            for i in rng.permutation(N):
                t  += 1
                eta = 1.0 / (self.reg * t)
                margin = y_pm[i] * (X[i] @ w + b)
                if margin < 1:
                    w = (1 - eta * self.reg) * w + eta * y_pm[i] * X[i]
                    b = b + eta * y_pm[i]
                else:
                    w = (1 - eta * self.reg) * w
                w_avg += w
                b_avg += b
        return w_avg / t, b_avg / t

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.seed)
        self.W_ = []
        self.b_ = []
        for c in self.classes_:
            y_pm = np.where(y == c, 1.0, -1.0)
            w, b = self._fit_binary(X, y_pm, rng)
            self.W_.append(w)
            self.b_.append(b)
        self.W_ = np.stack(self.W_)         # (K, d)
        self.b_ = np.array(self.b_)         # (K,)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self.W_.T + self.b_    # (N, K)
        return self.classes_[scores.argmax(axis=1)]


# ─── 4. MLP (2 層 + softmax) ────────────────────────────────
class MLPClassifier(BaseClassifier):
    """
    2 層 MLP: X → Linear → ReLU → Linear → Softmax
    最適化: Adam (汎用的に精度が出やすい)

    He 初期化:    W ~ N(0, sqrt(2/n_in))
    Adam:         m, v の指数移動平均 + バイアス補正
    """

    def __init__(self, hidden: int = 64, lr: float = 1e-3,
                 reg: float = 1e-4, epochs: int = 300, seed: int = 0,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.hidden = hidden
        self.lr     = lr
        self.reg    = reg
        self.epochs = epochs
        self.seed   = seed
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps

    @staticmethod
    def _softmax_stable(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        eZ = np.exp(Z)
        return eZ / eZ.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        N, d = X.shape

        rng = np.random.default_rng(self.seed)
        # He 初期化 (ReLU 前提)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), size=(d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden), size=(self.hidden, K))
        self.b2 = np.zeros(K)

        # Adam バッファ
        params = ["W1", "b1", "W2", "b2"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        t = 0

        # one-hot 教師信号
        Y = np.zeros((N, K))
        Y[np.arange(N), y] = 1.0

        prev_loss = np.inf
        for _ in range(self.epochs):
            # 順伝播
            Z1 = X @ self.W1 + self.b1
            H1 = np.maximum(Z1, 0)               # ReLU
            Z2 = H1 @ self.W2 + self.b2
            P  = self._softmax_stable(Z2)

            # 損失
            log_p = np.log(P[np.arange(N), y] + 1e-12)
            loss = -log_p.mean() + 0.5 * self.reg * (
                (self.W1 ** 2).sum() + (self.W2 ** 2).sum()
            )

            # 逆伝播
            dZ2 = (P - Y) / N
            grads = {
                "W2": H1.T @ dZ2 + self.reg * self.W2,
                "b2": dZ2.sum(axis=0),
            }
            dH1 = dZ2 @ self.W2.T
            dZ1 = dH1 * (Z1 > 0)                 # ReLU の勾配
            grads["W1"] = X.T @ dZ1 + self.reg * self.W1
            grads["b1"] = dZ1.sum(axis=0)

            # Adam 更新
            t += 1
            for p in params:
                g = grads[p]
                m[p] = self.beta1 * m[p] + (1 - self.beta1) * g
                v[p] = self.beta2 * v[p] + (1 - self.beta2) * g * g
                m_hat = m[p] / (1 - self.beta1 ** t)
                v_hat = v[p] / (1 - self.beta2 ** t)
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                setattr(self, p, getattr(self, p) - update)

            if abs(prev_loss - loss) < 1e-7:
                break
            prev_loss = loss
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        H1 = np.maximum(X @ self.W1 + self.b1, 0)
        Z2 = H1 @ self.W2 + self.b2
        return self.classes_[Z2.argmax(axis=1)]


# ─── 5. Gaussian Naive Bayes ────────────────────────────────
class GaussianNBClassifier(BaseClassifier):
    """
    ガウシアンナイーブベイズ。連続値特徴に適した確率的分類器。

    各特徴 j がクラスごとに独立なガウス分布に従うと仮定:
        P(x_j | y=c) = N(x_j ; μ_{cj}, σ²_{cj})

    対数事後確率 (log-sum-exp で安定化):
        log P(y=c | x) ∝ log P(c)
            − 0.5 Σ_j [log(2π σ²_{cj}) + (x_j − μ_{cj})² / σ²_{cj}]

    var_smoothing: 全クラスの最大分散に比例した値を各分散に加算し
                  ゼロ分散を防ぐ。
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        means, sigmas, log_priors = [], [], []
        for c in self.classes_:
            X_c = X[y == c]
            means.append(X_c.mean(axis=0))
            sigmas.append(X_c.var(axis=0))
            log_priors.append(np.log(len(X_c) / len(X)))
        self.means_      = np.stack(means)                    # (K, d)
        self.sigmas_     = np.stack(sigmas)                   # (K, d)
        self.sigmas_    += self.var_smoothing * self.sigmas_.max()
        self.log_prior_  = np.array(log_priors)               # (K,)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        K = len(self.classes_)
        log_prob = np.empty((X.shape[0], K))
        for i in range(K):
            diff = X - self.means_[i]
            log_prob[:, i] = (
                self.log_prior_[i]
                - 0.5 * np.sum(np.log(2 * np.pi * self.sigmas_[i]))
                - 0.5 * np.sum(diff ** 2 / self.sigmas_[i], axis=1)
            )
        return self.classes_[log_prob.argmax(axis=1)]


# ─── 6. Linear Discriminant Analysis ────────────────────────
class LDAClassifier(BaseClassifier):
    """
    線形判別分析 (Fisher の LDA)。

    クラス内散布行列 Σ_W を用いた線形判別関数:
        δ_c(x) = x^T Σ_W^{-1} μ_c − 0.5 μ_c^T Σ_W^{-1} μ_c + log P(c)

    Σ_W = Σ_c Σ_{x∈C_c} (x − μ_c)(x − μ_c)^T   (クラス内散布行列)

    ŷ = argmax_c δ_c(x)

    reg: Σ_W が特異行列になる場合の正則化係数 (reg × I を加算)。
    """

    def __init__(self, reg: float = 1e-4):
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        N, d = X.shape
        S_W    = np.zeros((d, d))
        means, priors = [], []
        for c in self.classes_:
            X_c  = X[y == c]
            mu_c = X_c.mean(axis=0)
            means.append(mu_c)
            priors.append(len(X_c) / N)
            diff = X_c - mu_c
            S_W += diff.T @ diff
        self.means_  = np.stack(means)       # (K, d)
        self.priors_ = np.array(priors)      # (K,)
        S_W_inv      = np.linalg.pinv(S_W + self.reg * np.eye(d))
        # W[:, c] = S_W_inv @ mu_c  →  scores = X @ W + bias
        self.W_    = S_W_inv @ self.means_.T                              # (d, K)
        self.bias_ = (
            -0.5 * np.einsum("kd,dk->k", self.means_, self.W_)
            + np.log(self.priors_)
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.classes_[(X @ self.W_ + self.bias_).argmax(axis=1)]


# ─── 7. Voting Ensemble ─────────────────────────────────────
class VotingEnsembleClassifier(BaseClassifier):
    """
    複数の分類器の多数決による予測 (Hard Voting)。

    デフォルト構成: SoftmaxRegression + MLPClassifier + LDAClassifier
    性質の異なる3モデルを組み合わせることで誤りを打ち消し合う。
    """

    def __init__(self, classifiers: list = None):
        if classifiers is None:
            classifiers = [
                SoftmaxRegression(lr=0.5, reg=1e-4, epochs=800),
                MLPClassifier(hidden=64, lr=1e-3, epochs=300),
                LDAClassifier(reg=1e-4),
            ]
        self._classifiers = classifiers
        self.classes_: np.ndarray = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        for clf in self._classifiers:
            clf.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = np.stack([clf.predict(X) for clf in self._classifiers])
        n_samples = preds.shape[1]
        result = np.empty(n_samples, dtype=self.classes_.dtype)
        for i in range(n_samples):
            unique, counts = np.unique(preds[:, i], return_counts=True)
            result[i] = unique[counts.argmax()]
        return result


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_classifier(name: str = "softmax", **kwargs) -> BaseClassifier:
    table = {
        "knn":      KNNClassifier,
        "softmax":  SoftmaxRegression,
        "svm":      OvRLinearSVM,
        "mlp":      MLPClassifier,
        "gnb":      GaussianNBClassifier,
        "lda":      LDAClassifier,
        "ensemble": VotingEnsembleClassifier,
    }
    if name not in table:
        raise ValueError(f"unknown classifier: {name}")
    return table[name](**kwargs)
