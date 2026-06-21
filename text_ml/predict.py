"""
==============================================================
  predict.py | 保存済み MLP モデルで文章を分類する
==============================================================
使い方:
    python3 predict.py
    python3 predict.py "スマートフォンの新機能が発表された。"

model.pt はベクトライザーを含まないため、
訓練データで同じベクトライザーを再フィットしてから推論する。
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from data       import load_dataset
from tokenizer  import get_tokenizer
from vectorizer import get_vectorizer

MODEL_PATH = Path(__file__).parent / "model.pt"


def _build_model(n_features: int, hidden_dim: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, n_classes),
    )


def load_pipeline(model_path: Path):
    """保存済みモデルと、訓練データから再フィットしたベクトライザーを返す"""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    classes    = np.array(ckpt["classes"])  # 元のクラスインデックス配列
    hidden_dim = ckpt["hidden_dim"]
    n_features = ckpt["n_features"]
    n_classes  = len(classes)

    model = _build_model(n_features, hidden_dim, n_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 訓練データでベクトライザーを再フィット (保存時と同じ設定)
    train, _ = load_dataset(seed=42)
    tokenizer  = get_tokenizer("char_ngram", ns=(2, 3))
    vectorizer = get_vectorizer("bm25", min_df=1)
    vectorizer.fit([tokenizer.tokenize(t) for t in train.texts])

    return model, classes, tokenizer, vectorizer, train.label_names


def predict_text(text: str, model, classes, tokenizer, vectorizer, label_names) -> tuple[str, list]:
    """テキストを分類し (予測ラベル, クラスごとの確率リスト) を返す"""
    tokens = tokenizer.tokenize(text)
    x = torch.tensor(vectorizer.transform([tokens]).astype(np.float32))
    with torch.no_grad():
        logits = model(x)[0]
        probs  = torch.softmax(logits, dim=0).numpy()
    pred_label = label_names[int(classes[probs.argmax()])]
    scored = sorted(
        [(label_names[int(c)], float(p)) for c, p in zip(classes, probs)],
        key=lambda x: -x[1],
    )
    return pred_label, scored


def main():
    if not MODEL_PATH.exists():
        print(f"エラー: {MODEL_PATH} が見つかりません。先に main.py を実行してください。")
        sys.exit(1)

    print("モデルを読み込んでいます...")
    model, classes, tokenizer, vectorizer, label_names = load_pipeline(MODEL_PATH)
    print("準備完了。\n")

    # コマンドライン引数があれば1件だけ処理して終了
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, scored = predict_text(text, model, classes, tokenizer, vectorizer, label_names)
        print(f"入力: {text}")
        print(f"予測: {label}\n")
        print("スコア:")
        for name, prob in scored:
            bar = "█" * int(prob * 30)
            print(f"  {name:<10} {bar:<30} {prob:.1%}")
        return

    # 対話モード
    print("テキストを入力してください (終了: Ctrl+C または空行のまま Enter)\n")
    while True:
        try:
            text = input("テキスト> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break
        if not text:
            break
        label, scored = predict_text(text, model, classes, tokenizer, vectorizer, label_names)
        print(f"  予測: {label}")
        for name, prob in scored:
            bar = "█" * int(prob * 20)
            print(f"    {name:<10} {bar:<20} {prob:.1%}")
        print()


if __name__ == "__main__":
    main()
