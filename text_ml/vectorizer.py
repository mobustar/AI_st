"""
==============================================================
  vectorizer.py | ベクトル化 (Tokens → Numerical Matrix)
==============================================================
トークン列を機械学習モデルに入力できる固定長ベクトルに変換する。

選択アルゴリズム:
  1. CountVectorizer    - 単語の出現回数 (Bag of Words)
  2. TfIdfVectorizer    - 出現頻度を IDF で重み付け (最も精度が出やすい)
  3. HashingVectorizer  - ハッシュで固定次元化 (大規模・オンライン向け)

精度の傾向:
  TF-IDF > Count > Hashing (固定次元)

数式:
  TF(t,d)    = 文書 d における単語 t の出現回数
  IDF(t)     = log( (1 + N) / (1 + DF(t)) ) + 1
                N: 全文書数 / DF(t): 単語 t を含む文書数
  TF-IDF     = TF(t,d) * IDF(t)
  最後に L2 正規化することで文書長の違いを吸収する
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List
import hashlib
import numpy as np


class BaseVectorizer(ABC):
    """全ベクトル化器の共通インタフェース"""

    @abstractmethod
    def fit(self, token_lists: List[List[str]]) -> "BaseVectorizer":
        ...

    @abstractmethod
    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        ...

    def fit_transform(self, token_lists: List[List[str]]) -> np.ndarray:
        return self.fit(token_lists).transform(token_lists)

    @property
    def n_features(self) -> int:
        ...


# ─── 1. Bag of Words (出現回数) ─────────────────────────────
class CountVectorizer(BaseVectorizer):
    """
    各単語の出現回数を数えてベクトル化する素朴な手法。

    パラメータ:
        min_df: 最小文書頻度 (これ未満しか出現しない語は無視)
        max_df: 最大文書頻度 (これより多く出現する語は無視 / 0~1 の比率)
    """

    def __init__(self, min_df: int = 1, max_df: float = 1.0):
        self.min_df = min_df
        self.max_df = max_df
        self.vocab_: Dict[str, int] = {}

    def fit(self, token_lists: List[List[str]]):
        n_docs = len(token_lists)
        # 文書頻度 DF を計算 (各語が出現した文書数)
        df = Counter()
        for tokens in token_lists:
            for t in set(tokens):
                df[t] += 1

        # min_df / max_df で語彙をフィルタ
        max_count = self.max_df * n_docs if isinstance(self.max_df, float) else self.max_df
        kept = sorted(
            t for t, c in df.items()
            if c >= self.min_df and c <= max_count
        )
        self.vocab_ = {t: i for i, t in enumerate(kept)}
        return self

    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        n_docs   = len(token_lists)
        n_feats  = len(self.vocab_)
        X = np.zeros((n_docs, n_feats), dtype=np.float64)
        for i, tokens in enumerate(token_lists):
            for t in tokens:
                j = self.vocab_.get(t)
                if j is not None:
                    X[i, j] += 1.0
        return X

    @property
    def n_features(self) -> int:
        return len(self.vocab_)


# ─── 2. TF-IDF ──────────────────────────────────────────────
class TfIdfVectorizer(BaseVectorizer):
    """
    TF-IDF: 頻出語の影響を抑え、文書を識別しやすい語を強調する。

    sublinear_tf=True の場合 TF を log(1+TF) に置き換えて
    出現回数の大きさによる過大評価を抑える (経験的に精度向上)。
    """

    def __init__(self,
                 min_df: int = 1,
                 max_df: float = 1.0,
                 sublinear_tf: bool = True,
                 norm: str = "l2"):
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray = None  # 学習時に計算

    def fit(self, token_lists: List[List[str]]):
        n_docs = len(token_lists)
        df = Counter()
        for tokens in token_lists:
            for t in set(tokens):
                df[t] += 1

        # 語彙フィルタ
        max_count = self.max_df * n_docs if isinstance(self.max_df, float) else self.max_df
        kept = sorted(
            t for t, c in df.items()
            if c >= self.min_df and c <= max_count
        )
        self.vocab_ = {t: i for i, t in enumerate(kept)}

        # IDF 計算: log((1+N)/(1+df)) + 1  (smooth IDF)
        idf = np.zeros(len(self.vocab_), dtype=np.float64)
        for t, j in self.vocab_.items():
            idf[j] = np.log((1 + n_docs) / (1 + df[t])) + 1.0
        self.idf_ = idf
        return self

    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        n_docs   = len(token_lists)
        n_feats  = len(self.vocab_)
        X = np.zeros((n_docs, n_feats), dtype=np.float64)
        for i, tokens in enumerate(token_lists):
            for t in tokens:
                j = self.vocab_.get(t)
                if j is not None:
                    X[i, j] += 1.0

        # TF を log スケールに (sublinear_tf=True のとき)
        if self.sublinear_tf:
            mask = X > 0
            X[mask] = 1.0 + np.log(X[mask])

        # IDF で重み付け
        X *= self.idf_

        # L2 正規化 (文書長の違いを吸収 → コサイン類似度的な距離になる)
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X /= norms
        return X

    @property
    def n_features(self) -> int:
        return len(self.vocab_)


# ─── 3. Hashing (固定次元) ─────────────────────────────────
class HashingVectorizer(BaseVectorizer):
    """
    各単語をハッシュ関数で固定次元 n に写像する。
    語彙を保持しないためメモリ効率が良く、未知語も自動で扱える。
    衝突 (異なる語が同じ次元に落ちる) は避けられないが、
    n を十分大きく取ると精度劣化は限定的。

    符号トリック: ハッシュ値の偶奇で +1/-1 を割り振り、
    衝突時の打ち消しを期待する (Weinberger et al. 2009)。
    """

    def __init__(self, n_features: int = 2 ** 14, signed: bool = True):
        self._n_features = n_features
        self.signed = signed

    def fit(self, token_lists):
        # 学習不要 (語彙を保持しない)
        return self

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little")

    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        n_docs = len(token_lists)
        X = np.zeros((n_docs, self._n_features), dtype=np.float64)
        for i, tokens in enumerate(token_lists):
            for t in tokens:
                h    = self._hash(t)
                idx  = h % self._n_features
                sign = 1.0 if (not self.signed or (h >> 1) % 2 == 0) else -1.0
                X[i, idx] += sign
        return X

    @property
    def n_features(self) -> int:
        return self._n_features


# ─── 4. BM25 ────────────────────────────────────────────────
class BM25Vectorizer(BaseVectorizer):
    """
    Okapi BM25 によるベクトル化。TF-IDF の改良版。

    TF の飽和処理と文書長正規化を同時に行う:

        IDF(t)       = log( (N − DF(t) + 0.5) / (DF(t) + 0.5)  +  1 )
                       ※ Robertson IDF: 常に正値

        TF_BM25(t,d) = TF(t,d) × (k1 + 1)
                       ────────────────────────────────────────────
                       TF(t,d) + k1 × (1 − b + b × |d| / avgdl)

        BM25(t,d)    = IDF(t) × TF_BM25(t,d)

    TF-IDF との違い:
      ・TF が増えても BM25 スコアは k1+1 に漸近する (飽和効果)
      ・avgdl との比率で文書長を正規化 (b=0: 正規化なし、b=1: 完全正規化)
    """

    def __init__(self, min_df: int = 1, k1: float = 1.5, b: float = 0.75):
        self.min_df = min_df
        self.k1     = k1
        self.b      = b
        self.vocab_:  Dict[str, int] = {}
        self.idf_:    np.ndarray     = None
        self.avgdl_:  float          = 1.0

    def fit(self, token_lists: List[List[str]]):
        n_docs = len(token_lists)
        df     = Counter()
        lengths = []
        for tokens in token_lists:
            lengths.append(len(tokens))
            for t in set(tokens):
                df[t] += 1
        self.avgdl_ = float(np.mean(lengths)) if lengths else 1.0
        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab_ = {t: i for i, t in enumerate(kept)}
        idf = np.zeros(len(self.vocab_))
        for t, j in self.vocab_.items():
            idf[j] = np.log((n_docs - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
        self.idf_ = idf
        return self

    def transform(self, token_lists: List[List[str]]) -> np.ndarray:
        n_docs  = len(token_lists)
        n_feats = len(self.vocab_)
        X = np.zeros((n_docs, n_feats), dtype=np.float64)
        for i, tokens in enumerate(token_lists):
            dl = max(len(tokens), 1)
            tf_map = Counter(tokens)
            for t, tf in tf_map.items():
                j = self.vocab_.get(t)
                if j is None:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl_, 1.0))
                X[i, j] = self.idf_[j] * tf * (self.k1 + 1.0) / denom
        return X

    @property
    def n_features(self) -> int:
        return len(self.vocab_)


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_vectorizer(name: str = "tfidf", **kwargs) -> BaseVectorizer:
    """
    名前で Vectorizer を取得する
        name in {"count", "tfidf", "hashing", "bm25"}
    """
    table = {
        "count":   CountVectorizer,
        "tfidf":   TfIdfVectorizer,
        "hashing": HashingVectorizer,
        "bm25":    BM25Vectorizer,
    }
    if name not in table:
        raise ValueError(f"unknown vectorizer: {name}")
    return table[name](**kwargs)
