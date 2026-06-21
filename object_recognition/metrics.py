"""
==============================================================
  metrics.py | 評価指標 (多クラス対応)
==============================================================
"""

import numpy as np
from typing import List


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_report(y_true, y_pred, label_names: List[str]) -> str:
    n  = len(label_names)
    cm = confusion_matrix(y_true, y_pred, n)
    eps = 1e-12

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    lines = [f"{'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>8}"]
    lines.append("-" * 42)
    for i, name in enumerate(label_names):
        lines.append(
            f"{name:<10}"
            f"{precision[i]:>8.3f}"
            f"{recall[i]:>8.3f}"
            f"{f1[i]:>8.3f}"
            f"{cm[i].sum():>8d}"
        )
    lines.append("-" * 42)
    lines.append(
        f"{'macro':<10}"
        f"{precision.mean():>8.3f}"
        f"{recall.mean():>8.3f}"
        f"{f1.mean():>8.3f}"
        f"{cm.sum():>8d}"
    )
    lines.append(f"accuracy = {accuracy(y_true, y_pred):.3f}")

    # 混同行列も表示
    lines.append("\nConfusion Matrix (row=true, col=pred):")
    header = "     " + " ".join(f"{n:>8}" for n in label_names)
    lines.append(header)
    for i, name in enumerate(label_names):
        row = f"{name:<5}" + " ".join(f"{cm[i,j]:>8d}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)
