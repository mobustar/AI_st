"""
==============================================================
  classifier.py | 分類器 (戦略パターン)
==============================================================
ベクトル化された文書を分類する。8 種を実装。

選択アルゴリズム:
  1. MultinomialNB          - 多項ナイーブベイズ。高速・小データに強い
  2. LogisticRegression     - 確率的線形分類器。万能なベースライン
  3. LinearSVM              - マージン最大化。高次元疎データに強い
  4. KNNClassifier          - インスタンスベース。学習不要だが推論が重い
  5. AveragedPerceptron     - オンライン誤り訂正 + 平均化で汎化向上
  6. RidgeClassifier        - 閉形式解で瞬時に収束。L2 正則化
  7. PyTorchMLPClassifier   - 2 隠れ層 MLP + Dropout (PyTorch, CUDA 対応)
  8. VotingEnsembleClassifier - NB + Ridge(OvR) + LogReg(OvR) の多数決

多クラス対応:
  二値分類器 (LogReg / SVM / Perceptron / Ridge) は
  OneVsRestClassifier でラップして K クラスに拡張する。
  get_classifier() が自動でラップするため呼び出し側は意識不要。

精度の傾向 (char_ngram + bm25/tfidf の場合):
  ensemble > svm(OvR) ≈ logreg(OvR) > nb > ridge(OvR) > perceptron > knn
  ※ NB は CountVectorizer と組み合わせるとさらに精度が上がる

数値安定性のための工夫:
  - Naive Bayes:     log 確率で計算 (アンダーフロー防止)
  - LogReg:          log-sum-exp で交差エントロピーを安定計算
  - SVM:             ヒンジ損失 + L2 正則化 + 平均化 SGD
  - PyTorchMLP:      CrossEntropyLoss (内部で log-softmax) + Dropout
"""

from abc import ABC, abstractmethod
import copy
import numpy as np


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} は save() をサポートしていません")


# ─── 1. Multinomial Naive Bayes ─────────────────────────────
class MultinomialNB(BaseClassifier):
    """
    多項ナイーブベイズ (テキスト分類の古典的かつ強力なベースライン)。

    モデル:
      P(c|d) ∝ P(c) * Π_i P(t_i|c)^{x_i}
      → log P(c|d) = log P(c) + Σ_i x_i * log P(t_i|c)

    学習:
      P(c)     = N_c / N                    (クラス事前確率)
      P(t_i|c) = (count(t_i,c) + α) /       (Laplace smoothing)
                 (Σ_j count(t_j,c) + α*|V|)

    アンダーフロー対策で全て対数で扱う。
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: np.ndarray = None
        self.class_log_prior_: np.ndarray = None
        self.feature_log_prob_: np.ndarray = None  # (n_classes, n_features)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        # NB は非負カウント前提なので負値はクリップ (Hashing 符号トリックなどへの防御)
        X = np.clip(X, 0.0, None)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            # クラス事前確率 log P(c)
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            # 特徴量の出現回数 + Laplace smoothing (alpha>0 で必ず正値になる)
            count = X_c.sum(axis=0) + self.alpha
            # 正規化して log P(t|c) を得る
            self.feature_log_prob_[i] = np.log(count) - np.log(count.sum())
        return self

    def _joint_log_likelihood(self, X):
        # log P(c|d) ∝ log P(c) + X @ log P(t|c)^T
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        X = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
        return self.classes_[self._joint_log_likelihood(X).argmax(axis=1)]

    def predict_proba(self, X):
        """log-sum-exp で安定的に確率化"""
        X = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
        jll = self._joint_log_likelihood(X)
        log_norm = self._logsumexp(jll, axis=1, keepdims=True)
        return np.exp(jll - log_norm)

    @staticmethod
    def _logsumexp(x, axis=None, keepdims=False):
        m = np.max(x, axis=axis, keepdims=True)
        out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
        return out if keepdims else np.squeeze(out, axis=axis)


# ─── 2. Logistic Regression (二値) ──────────────────────────
class LogisticRegression(BaseClassifier):
    """
    L2 正則化付きロジスティック回帰 (二値分類)。

    モデル: P(y=1|x) = σ(w·x + b),  σ(z) = 1/(1+e^{-z})

    損失: L = -1/N Σ [y log σ(z) + (1-y) log(1-σ(z))] + 0.5 * λ‖w‖²
    勾配: ∂L/∂w = 1/N X^T (σ(z) - y) + λw
          ∂L/∂b = 1/N Σ (σ(z) - y)

    最適化: フルバッチ勾配降下 (収束が安定 → 精度重視)
    """

    def __init__(self,
                 lr: float = 0.5,
                 reg: float = 1e-3,
                 epochs: int = 1000,
                 tol: float = 1e-6):
        self.lr      = lr
        self.reg     = reg
        self.epochs  = epochs
        self.tol     = tol
        self.w_:  np.ndarray = None
        self.b_:  float      = 0.0

    @staticmethod
    def _sigmoid(z):
        # オーバーフロー対策の数値安定実装
        out = np.empty_like(z)
        pos = z >= 0
        out[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
        ez        = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self.w_ = np.zeros(d)
        self.b_ = 0.0
        prev_loss = np.inf

        for _ in range(self.epochs):
            z       = X @ self.w_ + self.b_
            p       = self._sigmoid(z)
            err     = p - y                                  # (n,)
            grad_w  = X.T @ err / n + self.reg * self.w_
            grad_b  = err.mean()
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

            # 早期収束判定
            loss = self._loss(X, y)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def _loss(self, X, y):
        z = X @ self.w_ + self.b_
        # 数値安定な交差エントロピー: log(1+exp(z)) - y*z
        # np.where は両分岐を全要素で評価するため exp のオーバーフローが起きる。
        # clip で各分岐の入力を安全な範囲に絞る。
        log1pexp = np.where(
            z >= 0,
            z + np.log1p(np.exp(-np.clip(z, 0, None))),   # -z <= 0 なので exp は安全
            np.log1p(np.exp(np.clip(z, None, 0))),          # z <= 0 なので exp は安全
        )
        ce = (log1pexp - y * z).mean()
        return ce + 0.5 * self.reg * (self.w_ @ self.w_)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(X @ self.w_ + self.b_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ─── 3. Linear SVM (二値) ───────────────────────────────────
class LinearSVM(BaseClassifier):
    """
    L2 正則化 + ヒンジ損失の線形 SVM。

    モデル: f(x) = w·x + b, 予測 = sign(f(x))
    損失:  L = 1/N Σ max(0, 1 - y_i f(x_i)) + 0.5 * λ‖w‖²
           y ∈ {-1, +1}

    最適化: 平均化 SGD (Pegasos 的) で安定収束。
    精度を重視するため学習率は 1/(λt) で減衰させる。
    """

    def __init__(self,
                 reg: float = 1e-3,
                 epochs: int = 50,
                 seed: int = 0):
        self.reg    = reg
        self.epochs = epochs
        self.seed   = seed
        self.w_: np.ndarray = None
        self.b_: float = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        # ラベルを {-1,+1} に変換
        y01 = np.asarray(y)
        y_pm = np.where(y01 == 1, 1.0, -1.0)

        n, d   = X.shape
        rng    = np.random.default_rng(self.seed)
        self.w_ = np.zeros(d)
        self.b_ = 0.0

        # 平均化用のバッファ
        w_avg = np.zeros(d)
        b_avg = 0.0
        t = 0
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for i in order:
                t += 1
                eta = 1.0 / (self.reg * t)            # 学習率減衰
                margin = y_pm[i] * (X[i] @ self.w_ + self.b_)
                # ヒンジ損失の勾配
                if margin < 1:
                    self.w_ = (1 - eta * self.reg) * self.w_ + eta * y_pm[i] * X[i]
                    self.b_ = self.b_ + eta * y_pm[i]
                else:
                    self.w_ = (1 - eta * self.reg) * self.w_
                w_avg += self.w_
                b_avg += self.b_
        # 平均化 (汎化性能向上)
        self.w_ = w_avg / t
        self.b_ = b_avg / t
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


# ─── 4. k-NN (コサイン距離) ─────────────────────────────────
class KNNClassifier(BaseClassifier):
    """
    学習データ全件を保持し、推論時に k 個の最近傍で多数決する。

    距離関数: コサイン類似度 (テキストベクトルでは L2 距離より精度が高い)
    weights:  "uniform" or "distance" (距離による重み付き投票)

    精度を重視するため、デフォルトは k=5, weights="distance"。
    """

    def __init__(self, k: int = 5, weights: str = "distance"):
        self.k = k
        self.weights = weights
        self.X_: np.ndarray = None
        self.y_: np.ndarray = None
        self._X_norm: np.ndarray = None

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        # 学習データを L2 正規化しておくと、内積=コサイン類似度になる
        norms = np.linalg.norm(self.X_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._X_norm = self.X_ / norms
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms

        # コサイン類似度 → 距離 = 1 - 類似度
        sim  = Xn @ self._X_norm.T              # (n_test, n_train)
        dist = 1.0 - sim

        preds = np.empty(X.shape[0], dtype=self.y_.dtype)
        classes = np.unique(self.y_)

        for i in range(X.shape[0]):
            # 最近傍 k 件のインデックス
            idx = np.argpartition(dist[i], self.k)[: self.k]
            d   = dist[i, idx]
            lbl = self.y_[idx]

            if self.weights == "distance":
                # 距離が近いほど重みを大きく (eps で 0 除算回避)
                w = 1.0 / (d + 1e-9)
            else:
                w = np.ones_like(d)

            # クラスごとに重みを集計して最大のものを選ぶ
            scores = np.array([w[lbl == c].sum() for c in classes])
            preds[i] = classes[scores.argmax()]
        return preds


# ─── 5. Averaged Perceptron ─────────────────────────────────
class AveragedPerceptron(BaseClassifier):
    """
    平均化パーセプトロン (Freund & Schapire 1999)。

    オンライン学習で誤分類のたびに重みを更新し、
    全ステップの平均を最終重みとして使う。

    更新 (y_pm ∈ {−1, +1}):
        ŷ = sign(w · x + b)
        if ŷ ≠ y_pm:
            w ← w + y_pm · x
            b ← b + y_pm

    最終重み (平均化):
        w_final = (1/T) Σ_{t=1}^{T} w_t   （T = 総ステップ数）

    利点:
      - 1 パス学習で収束が速い
      - 平均化により汎化性能が向上し SVM に近い精度を出す
    """

    def __init__(self, epochs: int = 30, seed: int = 0):
        self.epochs = epochs
        self.seed   = seed

    def fit(self, X, y):
        X    = np.asarray(X, dtype=np.float64)
        y_pm = np.where(np.asarray(y) == 1, 1.0, -1.0)
        n, d = X.shape
        rng  = np.random.default_rng(self.seed)
        w, b = np.zeros(d), 0.0
        w_avg, b_avg, t = np.zeros(d), 0.0, 0
        for _ in range(self.epochs):
            for i in rng.permutation(n):
                t += 1
                if y_pm[i] * (X[i] @ w + b) <= 0:
                    w += y_pm[i] * X[i]
                    b += y_pm[i]
                w_avg += w
                b_avg += b
        self.w_ = w_avg / t
        self.b_ = b_avg / t
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


# ─── 6. Ridge Classifier ────────────────────────────────────
class RidgeClassifier(BaseClassifier):
    """
    リッジ回帰による分類器。閉形式解を持つ。

    最小化問題:
        min_{w,b}  ‖y − (Xb w + b)‖²  +  α ‖w‖²
        ※ y ∈ {0, 1}、出力 ≥ 0.5 を正例と判定

    バイアス項を含む双対形式 (n < d のとき O(n²d + n³) で高効率):
        X̃ = [X, 1] ∈ ℝ^{n×(d+1)}
        K = X̃ X̃^T + α I   (n×n)
        θ = X̃^T K^{-1} y  (d+1,)

    利点:
      - 反復なしの閉形式解 → 訓練が瞬時
      - L2 正則化で滑らかな決定境界を学習
    欠点:
      - 確率値を出力しない
      - 特徴次元が非常に大きいと K 計算がメモリ集約的になる
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N = X.shape[0]
        Xb    = np.column_stack([X, np.ones(N)])          # (N, d+1)
        K     = Xb @ Xb.T + self.alpha * np.eye(N)        # (N, N) 双対カーネル
        a     = np.linalg.solve(K, y)                      # (N,)
        theta = Xb.T @ a                                   # (d+1,)
        self.w_ = theta[:-1]
        self.b_ = theta[-1]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)


# ─── 7. One-vs-Rest ラッパー ────────────────────────────────
class OneVsRestClassifier(BaseClassifier):
    """
    二値分類器を One-vs-Rest で多クラスに拡張するラッパー。

    各クラス c に対して「クラス c か否か」の二値分類器を個別に学習し、
    推論時は各分類器のスコアが最も高いクラスを選ぶ。

    これにより LogisticRegression / LinearSVM / AveragedPerceptron /
    RidgeClassifier などの二値分類器をそのまま多クラスに使える。
    """

    def __init__(self, base: BaseClassifier):
        self._base = base
        self.classifiers_: list = []
        self.classes_: np.ndarray = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.classifiers_ = []
        for c in self.classes_:
            clf = copy.deepcopy(self._base)
            clf.fit(X, (y == c).astype(int))
            self.classifiers_.append(clf)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, clf in enumerate(self.classifiers_):
            if hasattr(clf, "predict_proba"):
                scores[:, i] = clf.predict_proba(X)
            elif hasattr(clf, "decision_function"):
                s = clf.decision_function(X)
                # 各二値分類器のスケールを [0,1] に揃えてからargmaxする。
                # 揃えないと大きい値を持つ分類器のクラスが常に選ばれる。
                lo, hi = s.min(), s.max()
                scores[:, i] = (s - lo) / (hi - lo + 1e-9)
            else:
                scores[:, i] = clf.predict(X).astype(float)
        return self.classes_[scores.argmax(axis=1)]


# ─── 8. PyTorch MLP (多クラス) ──────────────────────────────
class PyTorchMLPClassifier(BaseClassifier):
    """
    PyTorch による多層パーセプトロン分類器。
    CUDA が利用可能なら自動的に GPU を使用し、なければ CPU にフォールバックする。

    アーキテクチャ:
        Linear(n_features → hidden_dim) → ReLU → Dropout
        → Linear(hidden_dim → hidden_dim//2) → ReLU
        → Linear(hidden_dim//2 → n_classes)

    save(path) でモデルを .pt ファイルとして保存できる。
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 lr: float = 1e-3,
                 epochs: int = 200,
                 batch_size: int = 32,
                 weight_decay: float = 1e-4):
        self.hidden_dim   = hidden_dim
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.weight_decay = weight_decay
        self.model_:   "torch.nn.Module" = None
        self.classes_: np.ndarray        = None
        self._device = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        label_to_idx = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[int(c)] for c in y])

        self.model_ = nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, n_classes),
        ).to(device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, device=device)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=device)
        n   = X.shape[0]

        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                criterion(self.model_(X_t[idx]), y_t[idx]).backward()
                optimizer.step()
        return self

    def predict(self, X):
        import torch
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self._device)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_t).argmax(dim=1).cpu().numpy()
        return self.classes_[preds]

    def save(self, path: str) -> None:
        import torch
        torch.save({
            "state_dict": self.model_.state_dict(),
            "classes":    self.classes_.tolist(),   # numpy配列→リストでweights_only制限を回避
            "hidden_dim": self.hidden_dim,
            "n_features": next(self.model_.parameters()).shape[1],
        }, path)
        print(f"モデルを保存しました: {path}")


# ─── 9. Voting Ensemble ─────────────────────────────────────
class VotingEnsembleClassifier(BaseClassifier):
    """
    複数の分類器の多数決による予測 (Hard Voting)。

    各分類器が独立に学習・予測し、最も多くの票を得たクラスを返す。
    個々の分類器が異なるパターンで誤るとき、誤りが打ち消されて
    単体より精度が向上する。

    デフォルト構成: MultinomialNB + Ridge(OvR) + LogisticRegression(OvR)
    """

    def __init__(self, classifiers: list = None):
        if classifiers is None:
            classifiers = [
                MultinomialNB(alpha=1.0),
                OneVsRestClassifier(RidgeClassifier(alpha=1.0)),
                OneVsRestClassifier(LogisticRegression(lr=0.5, epochs=1000)),
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
        # 各モデルの予測 → (n_clf, n_samples) の行列
        preds = np.stack([clf.predict(X) for clf in self._classifiers])
        n_samples = preds.shape[1]
        result = np.empty(n_samples, dtype=self.classes_.dtype)
        for i in range(n_samples):
            unique, counts = np.unique(preds[:, i], return_counts=True)
            result[i] = unique[counts.argmax()]
        return result


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_classifier(name: str = "logreg", **kwargs) -> BaseClassifier:
    """
    名前で Classifier を取得する
        name in {"nb", "logreg", "svm", "knn", "perceptron", "ridge", "mlp"}

    二値分類器 (logreg / svm / perceptron / ridge) は自動的に
    OneVsRestClassifier でラップされ、多クラスに対応する。
    """
    # 多クラスをネイティブに扱える分類器
    _multiclass = {
        "nb":       MultinomialNB,
        "knn":      KNNClassifier,
        "mlp":      PyTorchMLPClassifier,
        "ensemble": VotingEnsembleClassifier,
    }
    # 二値分類器 → OvR でラップ
    _binary = {
        "logreg":     LogisticRegression,
        "svm":        LinearSVM,
        "perceptron": AveragedPerceptron,
        "ridge":      RidgeClassifier,
    }
    if name in _multiclass:
        return _multiclass[name](**kwargs)
    if name in _binary:
        return OneVsRestClassifier(_binary[name](**kwargs))
    raise ValueError(f"unknown classifier: {name}")
