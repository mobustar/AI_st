# lbph_core.py
# 必要モジュール: numpy, Pillow
import numpy as np
from PIL import Image

SIZE = (100, 100)   # 全画像をこのサイズに統一
GRID = (8, 8)       # 8x8 セルに分割してヒストグラムを取る


def load_gray(path):
    """画像 → グレースケール＆固定サイズの numpy 配列"""
    img = Image.open(path).convert("L").resize(SIZE)
    return np.asarray(img, dtype=np.uint8)


def lbp(img):
    """局所二値パターン: 各画素を周囲8近傍と比較し 0-255 のコードにする"""
    p = np.pad(img, 1, mode="edge").astype(np.int16)
    c = img.astype(np.int16)
    code = np.zeros_like(img, dtype=np.uint8)
    neighbors = [(-1, -1, 1), (-1, 0, 2), (-1, 1, 4), (0, 1, 8),
                 (1, 1, 16), (1, 0, 32), (1, -1, 64), (0, -1, 128)]
    for dy, dx, bit in neighbors:
        s = p[1 + dy:1 + dy + img.shape[0], 1 + dx:1 + dx + img.shape[1]]
        code |= ((s >= c) * bit).astype(np.uint8)
    return code


def feature(img):
    """LBP画像をグリッド分割し、各セルの正規化ヒストグラムを連結"""
    code = lbp(img)
    h, w = code.shape
    gh, gw = h // GRID[0], w // GRID[1]
    hists = []
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            cell = code[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            hist, _ = np.histogram(cell, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            hists.append(hist)
    return np.concatenate(hists)


def feature_from_path(path):
    return feature(load_gray(path))
