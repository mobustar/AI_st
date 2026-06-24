# train.py
# 使い方: python train.py
# dataset/<人名>/*.jpg を読み込み face_model.npz を作る
import os
import numpy as np
from lbph_core import feature_from_path

DATASET_DIR = "dataset"
MODEL_PATH = "face_model.npz"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    feats, labels = [], []
    for person in sorted(os.listdir(DATASET_DIR)):
        pdir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith(EXTS):
                path = os.path.join(pdir, fn)
                feats.append(feature_from_path(path))
                labels.append(person)
                print(f"  学習: {path} -> {person}")

    if not feats:
        print("画像が見つかりません。dataset/<人名>/ に画像を置いてください。")
        return

    feats = np.array(feats, dtype=np.float32)
    labels = np.array(labels)
    # ここで外部ファイルに保存（PyTorch不要）
    np.savez(MODEL_PATH, feats=feats, labels=labels)
    print(f"\n完了: {len(labels)}枚 / {len(set(labels))}人 を {MODEL_PATH} に保存しました")


if __name__ == "__main__":
    main()
