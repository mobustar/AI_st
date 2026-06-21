"""
==============================================================
  data.py | 合成図形データセット (Object Recognition)
==============================================================
  3 クラス: 0=circle / 1=square / 2=triangle
  各画像は (H, W) のグレースケール画像 (値域 [0,1])。
  位置・サイズ・ノイズをランダム化することで、
  位置不変性を要する物体認識タスクを擬似的に再現する。
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ImageDataset:
    images: np.ndarray   # (N, H, W) float32
    labels: np.ndarray   # (N,) int
    label_names: list


def _draw_circle(img: np.ndarray, cy: int, cx: int, r: int) -> None:
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[mask] = 1.0


def _draw_square(img: np.ndarray, cy: int, cx: int, r: int) -> None:
    H, W = img.shape
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - r), min(W, cx + r)
    img[y0:y1, x0:x1] = 1.0


def _draw_triangle(img: np.ndarray, cy: int, cx: int, r: int) -> None:
    """頂点を上に向けた二等辺三角形を描画する"""
    H, W = img.shape
    height = 2 * r
    for i in range(height):                  # i: 上から下への走査位置
        half_w = int(r * (i / height))       # 三角形の半幅 (上ほど狭い)
        y = cy - r + i
        if 0 <= y < H:
            x0, x1 = max(0, cx - half_w), min(W, cx + half_w + 1)
            img[y, x0:x1] = 1.0


_DRAWERS = [_draw_circle, _draw_square, _draw_triangle]


def make_shapes(n_per_class: int = 200,
                size: int = 24,
                noise: float = 0.08,
                seed: int = 0) -> ImageDataset:
    """
    各クラス n_per_class 枚の画像を生成する。
    Args:
        size : 画像サイズ (H=W=size)
        noise: ガウシアンノイズ標準偏差
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for cls, drawer in enumerate(_DRAWERS):
        for _ in range(n_per_class):
            img = np.zeros((size, size), dtype=np.float32)
            r   = rng.integers(size // 5, size // 3 + 1)            # 半径
            cy  = rng.integers(r + 1, size - r - 1)                 # 中心座標
            cx  = rng.integers(r + 1, size - r - 1)
            drawer(img, cy, cx, r)
            img += rng.normal(0, noise, img.shape).astype(np.float32)
            np.clip(img, 0.0, 1.0, out=img)
            images.append(img)
            labels.append(cls)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int64)

    # シャッフル
    idx = rng.permutation(len(labels))
    return ImageDataset(
        images=images[idx],
        labels=labels[idx],
        label_names=["circle", "square", "triangle"],
    )


def augment(ds: ImageDataset) -> ImageDataset:
    """
    左右反転・上下反転・180度回転でデータを4倍に拡張する。

    図形は反転後も同じクラスなので、ラベルはそのまま流用できる。
    拡張後のデータで学習することで、位置・向きの変動への汎化性能が上がる。
    """
    imgs: list = []
    labels: list = []
    for img, label in zip(ds.images, ds.labels):
        imgs.extend([
            img,
            img[:, ::-1],       # 左右反転
            img[::-1, :],       # 上下反転
            img[::-1, ::-1],    # 180度回転
        ])
        labels.extend([label] * 4)
    return ImageDataset(
        images=np.stack(imgs).copy(),
        labels=np.array(labels, dtype=ds.labels.dtype),
        label_names=ds.label_names,
    )


def train_test_split(ds: ImageDataset, ratio: float = 0.8, seed: int = 0
                     ) -> Tuple[ImageDataset, ImageDataset]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds.labels))
    n_tr = int(len(idx) * ratio)
    tr, te = idx[:n_tr], idx[n_tr:]
    return (ImageDataset(ds.images[tr], ds.labels[tr], ds.label_names),
            ImageDataset(ds.images[te], ds.labels[te], ds.label_names))
