# recognize.py
# 使い方: python recognize.py test.jpg
import sys
import numpy as np
from lbph_core import feature_from_path

MODEL_PATH = "face_model.npz"
THRESHOLD = 0.5   # この距離より遠ければ「未登録の人」と判定


def chi_square(a, b):
    """ヒストグラム距離（小さいほど似ている）"""
    return np.sum((a - b) ** 2 / (a + b + 1e-7), axis=1)


def main():
    if len(sys.argv) < 2:
        print("使い方: python recognize.py <画像ファイル>")
        return

    # 学習ファイルを読み込むだけ（再学習なし）
    data = np.load(MODEL_PATH, allow_pickle=True)
    feats, labels = data["feats"], data["labels"]

    feat = feature_from_path(sys.argv[1])
    dist = chi_square(feat[None, :], feats)
    i = int(np.argmin(dist))
    name, d = labels[i], float(dist[i])

    if d > THRESHOLD:
        print(f"判定: 未登録の人 (最も近いのは {name}, 距離 {d:.3f})")
    else:
        print(f"判定: {name}  (距離 {d:.3f})")


if __name__ == "__main__":
    main()
