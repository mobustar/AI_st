"""
==============================================================
  feature.py | 特徴抽出 (戦略パターン)
==============================================================
画像から固定長ベクトルを取り出す。物体認識の精度を決める
最も重要なコンポーネント。

選択アルゴリズム:
  1. RawPixels  - 画素値をそのまま flatten (ベースライン)
  2. HOG        - 勾配方向ヒストグラム (形状認識に強い)
  3. LBP        - 局所二値パターン (テクスチャに強い)
  4. EdgeHist   - Sobel エッジの方向ヒストグラム (簡易 HOG)

精度の傾向 (図形認識):
  HOG > EdgeHist > RawPixels > LBP

数式:
  HOG: 勾配 gy=dI/dy, gx=dI/dx
       magnitude = √(gx² + gy²)
       orientation = atan2(gy, gx)  (符号なしなら mod π)
       セルごとに方向別ヒストグラム, ブロックで L2 正規化
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseFeature(ABC):
    @abstractmethod
    def extract(self, images: np.ndarray) -> np.ndarray:
        """images: (N,H,W) → features: (N, F)"""


# ─── 1. Raw Pixels ──────────────────────────────────────────
class RawPixels(BaseFeature):
    """画素値をそのままベクトル化"""
    def extract(self, images):
        N = images.shape[0]
        return images.reshape(N, -1).astype(np.float32)


# ─── 2. HOG ─────────────────────────────────────────────────
class HOG(BaseFeature):
    """
    Histogram of Oriented Gradients (Dalal & Triggs 2005)

    手順:
      1. 勾配 gx, gy を計算 (中央差分)
      2. magnitude と orientation を計算
      3. cell_size × cell_size ごとに方向ヒストグラム作成
      4. block_size × block_size のセルブロックを L2-Hys 正規化
      5. 全ブロックを連結

    パラメータ (精度のため少しリッチに):
        n_bins=9, cell_size=4, block_size=2
        signed=False (角度を 0..π にまとめる)
    """

    def __init__(self,
                 n_bins: int = 9,
                 cell_size: int = 4,
                 block_size: int = 2,
                 signed: bool = False,
                 eps: float = 1e-6):
        self.n_bins = n_bins
        self.cell   = cell_size
        self.block  = block_size
        self.signed = signed
        self.eps    = eps

    def _gradients(self, img: np.ndarray):
        """中央差分による勾配計算 (端は 0 で埋める)"""
        gy = np.zeros_like(img)
        gx = np.zeros_like(img)
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        return gy, gx

    def _cell_histograms(self, mag: np.ndarray, ang: np.ndarray):
        """各セルごとに方向ヒストグラムを作る"""
        H, W = mag.shape
        cy, cx = H // self.cell, W // self.cell    # セル数 (整数で切り捨て)
        # 各画素を「どのビン」に属するか計算 (連続値ではなく最近傍ビン)
        bin_edges = np.linspace(0, np.pi if not self.signed else 2*np.pi,
                                self.n_bins + 1)
        bin_idx = np.clip(
            np.searchsorted(bin_edges, ang, side="right") - 1,
            0, self.n_bins - 1,
        )

        hists = np.zeros((cy, cx, self.n_bins), dtype=np.float32)
        for i in range(cy):
            for j in range(cx):
                m_cell = mag[i*self.cell:(i+1)*self.cell,
                             j*self.cell:(j+1)*self.cell]
                b_cell = bin_idx[i*self.cell:(i+1)*self.cell,
                                 j*self.cell:(j+1)*self.cell]
                # 各ビンに magnitude を加算 (重み付きヒストグラム)
                np.add.at(hists[i, j], b_cell.ravel(), m_cell.ravel())
        return hists

    def _block_normalize(self, hists: np.ndarray) -> np.ndarray:
        """重複ありのブロックで L2-Hys 正規化"""
        cy, cx, n_bins = hists.shape
        by, bx = cy - self.block + 1, cx - self.block + 1
        feats = []
        for i in range(by):
            for j in range(bx):
                v = hists[i:i+self.block, j:j+self.block, :].ravel()
                v /= np.sqrt((v ** 2).sum() + self.eps ** 2)   # L2 正規化
                v = np.minimum(v, 0.2)                          # クリッピング
                v /= np.sqrt((v ** 2).sum() + self.eps ** 2)   # 再正規化 (Hys)
                feats.append(v)
        return np.concatenate(feats) if feats else np.zeros(0, dtype=np.float32)

    def extract(self, images):
        feats: List[np.ndarray] = []
        for img in images:
            gy, gx = self._gradients(img)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            ang = np.arctan2(gy, gx)                            # [-π, π]
            if not self.signed:
                ang = ang % np.pi                               # 符号なし: [0, π)
            else:
                ang = (ang + 2 * np.pi) % (2 * np.pi)           # 符号あり: [0, 2π)
            hists = self._cell_histograms(mag, ang)
            feats.append(self._block_normalize(hists))
        return np.stack(feats).astype(np.float32)


# ─── 3. LBP ─────────────────────────────────────────────────
class LBP(BaseFeature):
    """
    Local Binary Pattern (Ojala 2002)

    各画素について 8 近傍と比較し、各位置で「中心以上=1 / 未満=0」
    の 8bit パターンを生成。画像全体のパターン分布(256bin)を特徴とする。

    回転不変版を採用すると精度が向上するが、ここでは基本版。
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins

    def extract(self, images):
        feats = []
        for img in images:
            H, W = img.shape
            code = np.zeros((H - 2, W - 2), dtype=np.uint8)
            center = img[1:-1, 1:-1]
            # 8 近傍を時計回りに比較
            offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
            for k, (dy, dx) in enumerate(offsets):
                neighbor = img[1+dy:H-1+dy, 1+dx:W-1+dx]
                code |= ((neighbor >= center).astype(np.uint8) << k)
            hist = np.bincount(code.ravel(), minlength=self.n_bins).astype(np.float32)
            hist /= hist.sum() + 1e-12          # 正規化
            feats.append(hist)
        return np.stack(feats)


# ─── 4. Edge Histogram ──────────────────────────────────────
class EdgeHist(BaseFeature):
    """
    Sobel フィルタによるエッジ強度の方向ヒストグラム。
    HOG のセル分割と正規化を省略した簡易版。
    """

    SOBEL_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    SOBEL_Y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)

    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins

    def _conv2d(self, img, kernel):
        """3x3 畳み込み (端は 0 パディング)"""
        H, W = img.shape
        pad = np.zeros((H+2, W+2), dtype=img.dtype)
        pad[1:-1, 1:-1] = img
        out = np.zeros_like(img)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * pad[i:i+H, j:j+W]
        return out

    def extract(self, images):
        feats = []
        edges = np.linspace(0, np.pi, self.n_bins + 1)
        for img in images:
            gx = self._conv2d(img, self.SOBEL_X)
            gy = self._conv2d(img, self.SOBEL_Y)
            mag = np.sqrt(gx**2 + gy**2)
            ang = np.arctan2(gy, gx) % np.pi    # [0, π)
            # 強度を重みとした方向ヒストグラム
            hist, _ = np.histogram(ang.ravel(), bins=edges,
                                    weights=mag.ravel())
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-12
            feats.append(hist)
        return np.stack(feats)


# ─── 5. DCT (Discrete Cosine Transform) ─────────────────────
class DCT(BaseFeature):
    """
    離散コサイン変換 (DCT-II) の低周波係数を特徴とする。
    JPEG 圧縮と同じ変換で、少ない係数で画像の主要情報を表現できる。

    手順:
      1. H×W 画像に 2D DCT-II を適用: F = D_H I D_W^T
         D_H[k,n] = sqrt(2/H) cos(πk(2n+1)/(2H))   k>0
         D_H[0,n] = sqrt(1/H)
      2. ジグザグスキャンで低周波成分を優先して n_components 個抽出

    精度の傾向:
      形状よりも明暗・テクスチャのパターンを捉える。
      HOG・EdgeHist より低精度だが独立した情報を持つ。
    """

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self._dct_cache: dict = {}

    def _dct_matrix(self, N: int) -> np.ndarray:
        if N not in self._dct_cache:
            n = np.arange(N)
            k = np.arange(N)
            D = np.cos(np.pi * k[:, None] * (2 * n[None, :] + 1) / (2 * N))
            D[0] /= np.sqrt(2)
            D *= np.sqrt(2.0 / N)
            self._dct_cache[N] = D
        return self._dct_cache[N]

    def _zigzag(self, H: int, W: int) -> np.ndarray:
        """対角線をジグザグに走査したときのフラット化インデックスを返す"""
        indices = []
        for s in range(H + W - 1):
            if s % 2 == 0:
                r, c = min(s, H - 1), max(0, s - (H - 1))
                while r >= 0 and c < W:
                    indices.append(r * W + c)
                    r -= 1; c += 1
            else:
                r, c = max(0, s - (W - 1)), min(s, W - 1)
                while r < H and c >= 0:
                    indices.append(r * W + c)
                    r += 1; c -= 1
        return np.array(indices)

    def extract(self, images: np.ndarray) -> np.ndarray:
        _, H, W = images.shape
        D_H = self._dct_matrix(H)
        D_W = self._dct_matrix(W)
        zz  = self._zigzag(H, W)[:self.n_components]
        feats = []
        for img in images:
            F = D_H @ img.astype(np.float64) @ D_W.T
            feats.append(F.ravel()[zz].astype(np.float32))
        return np.stack(feats)


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_feature(name: str = "hog", **kwargs) -> BaseFeature:
    table = {
        "raw":      RawPixels,
        "hog":      HOG,
        "lbp":      LBP,
        "edge":     EdgeHist,
        "dct":      DCT,
    }
    if name not in table:
        raise ValueError(f"unknown feature: {name}")
    return table[name](**kwargs)
