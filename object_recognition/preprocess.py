"""
==============================================================
  preprocess.py | 画像前処理 (戦略パターン)
==============================================================
特徴抽出の前に行う画素値正規化。後段モデルの収束性と
精度に直接影響する。

選択アルゴリズム:
  1. NoOp                 - 何もしない
  2. MinMaxScaler         - 0-1 正規化
  3. StandardScaler       - 平均 0 / 分散 1 (チャネル統計)
  4. HistogramEqualizer   - ヒストグラム均等化 (照明変化に頑健)
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self,       images: np.ndarray) -> "BasePreprocessor": ...
    @abstractmethod
    def transform(self, images: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, images):
        return self.fit(images).transform(images)


# ─── 1. NoOp ────────────────────────────────────────────────
class NoOp(BasePreprocessor):
    """変換しない (既に [0,1] のときに使う)"""
    def fit(self, images):       return self
    def transform(self, images): return images.astype(np.float32)


# ─── 2. MinMax ──────────────────────────────────────────────
class MinMaxScaler(BasePreprocessor):
    """全画素値を [0,1] に線形変換"""
    def fit(self, images):
        self.min_ = float(images.min())
        self.max_ = float(images.max())
        return self
    def transform(self, images):
        denom = max(self.max_ - self.min_, 1e-12)
        return ((images - self.min_) / denom).astype(np.float32)


# ─── 3. Standard ────────────────────────────────────────────
class StandardScaler(BasePreprocessor):
    """
    平均 0 / 分散 1 に標準化。
    画像全体の統計を使う (画像個別ではない) ためクラス間で公平。
    """
    def fit(self, images):
        self.mean_ = float(images.mean())
        self.std_  = float(images.std() + 1e-12)
        return self
    def transform(self, images):
        return ((images - self.mean_) / self.std_).astype(np.float32)


# ─── 4. ヒストグラム均等化 ──────────────────────────────────
class HistogramEqualizer(BasePreprocessor):
    """
    各画像ごとに累積分布関数 (CDF) を均等化する。
    照明変化や暗部つぶれに対して頑健になる。
    """
    def __init__(self, bins: int = 256):
        self.bins = bins

    def fit(self, images):
        return self

    def _equalize_one(self, img: np.ndarray) -> np.ndarray:
        # [0,1] を 0..bins-1 の整数に量子化
        flat = (img.flatten() * (self.bins - 1)).astype(np.int64)
        flat = np.clip(flat, 0, self.bins - 1)
        hist = np.bincount(flat, minlength=self.bins)
        cdf  = hist.cumsum().astype(np.float32)
        cdf /= max(cdf[-1], 1)                # 正規化 (0..1)
        eq   = cdf[flat].reshape(img.shape)
        return eq.astype(np.float32)

    def transform(self, images):
        # まず [0,1] に押し込む (堅牢化)
        lo, hi = images.min(), images.max()
        if hi > lo:
            scaled = (images - lo) / (hi - lo)
        else:
            scaled = images
        return np.stack([self._equalize_one(im) for im in scaled])


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_preprocessor(name: str = "standard", **kwargs) -> BasePreprocessor:
    table = {
        "noop":     NoOp,
        "minmax":   MinMaxScaler,
        "standard": StandardScaler,
        "hist":     HistogramEqualizer,
    }
    if name not in table:
        raise ValueError(f"unknown preprocessor: {name}")
    return table[name](**kwargs)
