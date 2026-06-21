"""
==============================================================
  data.py | 学習用テキストデータセット
==============================================================
text_dataset.csv からデータを読み込み、
80:20 の訓練/評価セットに分割して返す。
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Dataset:
    """テキスト分類用データセット"""
    texts:       List[str]
    labels:      List[int]
    label_names: List[str]


_CSV_PATH = Path(__file__).parent / "text_dataset.csv"


def load_dataset(seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    text_dataset.csv を読み込み、訓練用と評価用に 80:20 で分割した
    Dataset を返す。
    """
    texts: List[str] = []
    labels: List[int] = []
    label_map: dict[int, str] = {}

    with _CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            label = int(row["label"])
            labels.append(label)
            label_map.setdefault(label, row["label_name"])

    # label_names をラベル番号順に並べる
    label_names = [label_map[i] for i in sorted(label_map)]

    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)

    n_train = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    train = Dataset(
        texts=[texts[i] for i in train_idx],
        labels=[labels[i] for i in train_idx],
        label_names=label_names,
    )
    test = Dataset(
        texts=[texts[i] for i in test_idx],
        labels=[labels[i] for i in test_idx],
        label_names=label_names,
    )
    return train, test
