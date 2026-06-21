"""
==============================================================
  data.py | 教師なし学習用の合成データセット
==============================================================
sklearn に依存せず NumPy だけで生成する。

提供するデータセット:
  blobs        - 球状クラスタ (K-means 向き)
  moons        - 三日月形 2 クラスタ (DBSCAN 向き)
  circles      - 同心円 2 クラスタ (DBSCAN 向き)
  anisotropic  - 楕円形クラスタ (共分散行列で伸長)
  anomaly      - 正常データ + 外れ値 (異常検知向き)
  swiss_roll   - スイスロール 3D 多様体 (次元削減向き)

各関数は DataSet (namedtuple) を返す:
  X      : shape (n, d) の特徴行列
  y      : shape (n,)   の真ラベル (-1 = 外れ値 or 未ラベル)
  name   : データセット名 (文字列)
"""

from collections import namedtuple
from typing import Optional
import numpy as np

DataSet = namedtuple("DataSet", ["X", "y", "name"])


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─── 1. Blobs (球状クラスタ) ─────────────────────────────────
def make_blobs(n_samples: int = 300, n_clusters: int = 3,
               cluster_std: float = 0.8, seed: int = 0) -> DataSet:
    """
    各クラスタが球状ガウス分布に従うデータ。
    K-means が最も得意とする「凸・等分散」なクラスタ構造。

    Args:
        n_samples  : 総サンプル数
        n_clusters : クラスタ数
        cluster_std: クラスタ内の標準偏差 (小さいほど密集)
        seed       : 乱数シード

    Returns:
        X: shape (n_samples, 2)  / y: 真のクラスタラベル (0..K-1)
    """
    rng = _rng(seed)
    # クラスタ中心をランダム配置 (範囲 [-5, 5])
    centers = rng.uniform(-5, 5, size=(n_clusters, 2))
    n_each = n_samples // n_clusters
    X_list, y_list = [], []
    for k, center in enumerate(centers):
        pts = rng.normal(loc=center, scale=cluster_std, size=(n_each, 2))
        X_list.append(pts)
        y_list.append(np.full(n_each, k, dtype=int))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # シャッフル
    idx = rng.permutation(len(X))
    return DataSet(X[idx], y[idx], "blobs")


# ─── 2. Moons (三日月形) ─────────────────────────────────────
def make_moons(n_samples: int = 300, noise: float = 0.1,
               seed: int = 0) -> DataSet:
    """
    2 つの三日月形クラスタ。
    線形分離不可能なため K-means には難しく、DBSCAN が得意。

    数式:
        上の月: x = cos(θ), y = sin(θ)         θ ∈ [0, π]
        下の月: x = 1-cos(θ), y = 1-sin(θ)-0.5  θ ∈ [0, π]
    """
    rng = _rng(seed)
    n_half = n_samples // 2
    theta = np.linspace(0, np.pi, n_half)
    X0 = np.column_stack([np.cos(theta), np.sin(theta)])
    X1 = np.column_stack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5])
    X = np.vstack([X0, X1]) + rng.normal(0, noise, size=(n_samples, 2))
    y = np.array([0] * n_half + [1] * n_half)
    idx = rng.permutation(n_samples)
    return DataSet(X[idx], y[idx], "moons")


# ─── 3. Circles (同心円) ─────────────────────────────────────
def make_circles(n_samples: int = 300, noise: float = 0.05,
                 factor: float = 0.5, seed: int = 0) -> DataSet:
    """
    内側と外側の同心円。
    DBSCAN の密度ベース特性が活きる典型例。

    Args:
        factor: 内円半径の比率 (0 < factor < 1)
    """
    rng = _rng(seed)
    n_half = n_samples // 2
    theta = np.linspace(0, 2 * np.pi, n_half)
    X_outer = np.column_stack([np.cos(theta), np.sin(theta)])
    X_inner = factor * np.column_stack([np.cos(theta), np.sin(theta)])
    X = np.vstack([X_outer, X_inner]) + rng.normal(0, noise, size=(n_samples, 2))
    y = np.array([0] * n_half + [1] * n_half)
    idx = rng.permutation(n_samples)
    return DataSet(X[idx], y[idx], "circles")


# ─── 4. Anisotropic (楕円形クラスタ) ─────────────────────────
def make_anisotropic(n_samples: int = 300, seed: int = 0) -> DataSet:
    """
    共分散行列で伸長した楕円形クラスタ。
    K-means は等方性を仮定するため楕円には苦労する。
    PCA との相性を示す例として利用する。
    """
    rng = _rng(seed)
    # 3 クラスタ、各クラスタを異なる方向に伸長
    transformations = [
        np.array([[2.0, -1.5], [0.5, 0.8]]),
        np.array([[0.5, 0.0], [0.0, 2.5]]),
        np.array([[1.0, 1.5], [-0.5, 1.0]]),
    ]
    centers = [np.array([-3, 0]), np.array([3, 0]), np.array([0, 4])]
    n_each = n_samples // 3
    X_list, y_list = [], []
    for k, (T, c) in enumerate(zip(transformations, centers)):
        pts = rng.normal(0, 0.5, size=(n_each, 2)) @ T.T + c
        X_list.append(pts)
        y_list.append(np.full(n_each, k, dtype=int))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(X))
    return DataSet(X[idx], y[idx], "anisotropic")


# ─── 5. Anomaly (正常 + 外れ値) ─────────────────────────────
def make_anomaly(n_normal: int = 300, n_outliers: int = 30,
                 seed: int = 0) -> DataSet:
    """
    正常データ (ガウス混合 2 クラスタ) + 一様ランダムな外れ値。

    ラベル:
        y = 1  : 正常
        y = -1 : 外れ値 (異常)

    異常検知モデルは y ラベルを使わず X だけで学習し、
    評価時に y と比較する (半教師あり的な使い方)。
    """
    rng = _rng(seed)
    # 正常データ: 2 つの密集クラスタ
    n1, n2 = n_normal // 2, n_normal - n_normal // 2
    X_norm = np.vstack([
        rng.normal([-2, -2], 0.5, size=(n1, 2)),
        rng.normal([2,   2], 0.5, size=(n2, 2)),
    ])
    # 外れ値: 広い範囲に一様分布
    X_out = rng.uniform(-6, 6, size=(n_outliers, 2))
    X = np.vstack([X_norm, X_out])
    y = np.concatenate([np.ones(n_normal, int), -np.ones(n_outliers, int)])
    idx = rng.permutation(len(X))
    return DataSet(X[idx], y[idx], "anomaly")


# ─── 6. Swiss Roll (3D 多様体) ─────────────────────────────
def make_swiss_roll(n_samples: int = 500, noise: float = 0.1,
                    seed: int = 0) -> DataSet:
    """
    巻き貝状の 3 次元データ (本来は 2 次元多様体)。
    t-SNE / PCA の次元削減デモに使う。

    パラメタ t に基づいて生成:
        x = t * cos(t)
        y = height (一様)
        z = t * sin(t)

    y ラベルは t の値 (連続) を 4 段階に量子化した疑似クラスタ。
    """
    rng = _rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n_samples))
    height = rng.uniform(0, 1, n_samples)
    X = np.column_stack([
        t * np.cos(t) + rng.normal(0, noise, n_samples),
        height,
        t * np.sin(t) + rng.normal(0, noise, n_samples),
    ])
    # t を 4 段階に量子化してラベルとする
    y = np.digitize(t, np.percentile(t, [25, 50, 75])).astype(int)
    return DataSet(X, y, "swiss_roll")


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_dataset(name: str, **kwargs) -> DataSet:
    table = {
        "blobs":       make_blobs,
        "moons":       make_moons,
        "circles":     make_circles,
        "anisotropic": make_anisotropic,
        "anomaly":     make_anomaly,
        "swiss_roll":  make_swiss_roll,
    }
    if name not in table:
        raise ValueError(f"unknown dataset: {name}. choices: {list(table)}")
    return table[name](**kwargs)
