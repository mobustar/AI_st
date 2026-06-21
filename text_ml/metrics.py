"""
==============================================================
  metrics.py | 評価指標
==============================================================
分類モデルの精度を多角的に評価する。

提供する指標:
  - accuracy       : 正解率 (全体に対する正解の割合)
  - precision/recall/f1 : クラスごと & マクロ平均
  - confusion_matrix    : 混同行列

数式:
  precision_c = TP_c / (TP_c + FP_c)
  recall_c    = TP_c / (TP_c + FN_c)
  f1_c        = 2 * P * R / (P + R)
  macro_f1    = mean(f1_c) over all classes
"""

import numpy as np
from typing import Dict, List


def confusion_matrix(y_true, y_pred, n_classes: int = None) -> np.ndarray:
    """
    混同行列 C を返す。 C[i,j] = 真ラベル i を j と予測した数。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(y_true, y_pred, n_classes: int = None) -> Dict:
    """
    クラスごとの precision/recall/F1 とマクロ平均を返す
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n  = cm.shape[0]
    eps = 1e-12

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp        # 列方向の和 - TP
    fn = cm.sum(axis=1) - tp        # 行方向の和 - TP

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision":   precision.tolist(),
        "recall":      recall.tolist(),
        "f1":          f1.tolist(),
        "macro_f1":    float(f1.mean()),
        "macro_prec":  float(precision.mean()),
        "macro_rec":   float(recall.mean()),
    }


def classification_report(y_true, y_pred, label_names: List[str]) -> str:
    """sklearn 風のテキストレポートを返す"""
    n = len(label_names)
    cm     = confusion_matrix(y_true, y_pred, n)
    metrics = precision_recall_f1(y_true, y_pred, n)

    lines = []
    lines.append(f"{'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>8}")
    lines.append("-" * 42)
    for i, name in enumerate(label_names):
        support = cm[i].sum()
        lines.append(
            f"{name:<10}"
            f"{metrics['precision'][i]:>8.3f}"
            f"{metrics['recall'][i]:>8.3f}"
            f"{metrics['f1'][i]:>8.3f}"
            f"{support:>8d}"
        )
    lines.append("-" * 42)
    lines.append(
        f"{'macro':<10}"
        f"{metrics['macro_prec']:>8.3f}"
        f"{metrics['macro_rec']:>8.3f}"
        f"{metrics['macro_f1']:>8.3f}"
        f"{cm.sum():>8d}"
    )
    lines.append(f"accuracy = {accuracy(y_true, y_pred):.3f}")
    return "\n".join(lines)
