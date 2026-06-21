"""
==============================================================
  main.py | 全アルゴリズム組み合わせ比較
==============================================================
Preprocessor × Feature × Classifier の全組み合わせを比較する。
4 × 5 × 7 = 140 通り。

データ:
  合成図形 (circle / square / triangle)
  データ拡張 (左右反転・上下反転・180度回転) で訓練データを 4 倍化

フェーズ:
  1. 全 140 通りを評価し accuracy 順にランキング表示
  2. 最良構成でクラス別詳細レポート (precision / recall / F1)

実行:
    python main.py
"""

from itertools import product

from data       import make_shapes, train_test_split, augment
from preprocess import get_preprocessor
from feature    import get_feature
from classifier import get_classifier
from pipeline   import ObjectRecognitionPipeline
from metrics    import accuracy, classification_report


PREPROCESSORS = ["noop", "minmax", "standard", "hist"]
FEATURES      = ["raw", "hog", "lbp", "edge", "dct"]
CLASSIFIERS   = ["knn", "softmax", "svm", "mlp", "gnb", "lda", "ensemble"]

_PREP_DESC = {
    "noop":     "前処理なし (画素値 [0,1] そのまま)",
    "minmax":   "MinMax正規化  各画像を [0,1] にスケーリング",
    "standard": "標準化  平均0・分散1 (モデル収束が安定しやすい)",
    "hist":     "ヒストグラム均等化  照明ムラを補正",
}
_FEAT_DESC = {
    "raw":  "Raw Pixels: 画素値を1次元に並べる (ベースライン)",
    "hog":  "HOG: 勾配方向ヒストグラム (形状認識に強い・推奨)",
    "lbp":  "LBP: 局所二値パターン (テクスチャ表現)",
    "edge": "EdgeHist: Sobelエッジの方向ヒストグラム (簡易HOG)",
    "dct":  "DCT: 離散コサイン変換係数 (周波数成分)",
}
_CLF_DESC = {
    "knn":     "k近傍法 k=5 (ユークリッド距離)",
    "softmax": "ソフトマックス回帰 (多クラスロジスティック回帰)",
    "svm":     "線形SVM One-vs-Rest",
    "mlp":     "多層パーセプトロン 2隠れ層",
    "gnb":     "ガウシアン・ナイーブベイズ",
    "lda":      "線形判別分析 (次元削減+分類)",
    "ensemble": "Voting Ensemble: Softmax + MLP + LDA の多数決",
}


def build_pipeline(prep_name: str, feat_name: str, clf_name: str):
    return ObjectRecognitionPipeline(
        preprocessor = get_preprocessor(prep_name),
        feature      = get_feature(feat_name),
        classifier   = get_classifier(clf_name),
    )


def run_all_combinations():
    """全組み合わせを評価し精度順のリストを返す"""
    ds = make_shapes(n_per_class=200, size=24, noise=0.08, seed=0)
    train_raw, test = train_test_split(ds, ratio=0.8, seed=1)
    train = augment(train_raw)   # 左右・上下反転・180度回転で4倍に拡張
    n_classes = len(train.label_names)
    total     = len(PREPROCESSORS) * len(FEATURES) * len(CLASSIFIERS)
    baseline  = 1.0 / n_classes

    print("=" * 66)
    print("  フェーズ 1/2 : 全アルゴリズム組み合わせ比較")
    print("=" * 66)
    print(f"  【処理内容】")
    print(f"    合成図形画像を 3段階のパイプラインで分類します。")
    print(f"      ① 前処理 (Preprocessor): 画素値を正規化してモデルが学習しやすくする")
    print(f"      ② 特徴抽出 (Feature)   : 画像(H×W)→固定長ベクトル (形状・エッジ情報を圧縮)")
    print(f"      ③ 分類器 (Classifier)  : ベクトル→クラス予測")
    print(f"    {len(PREPROCESSORS)}前処理 × {len(FEATURES)}特徴 × {len(CLASSIFIERS)}分類器 = {total}通りを全評価します。")
    print()
    print(f"  【データ (データ拡張あり)】")
    print(f"    合成図形画像  クラス: {' / '.join(train.label_names)}")
    print(f"    画像サイズ: {train.images.shape[1]}×{train.images.shape[2]}px グレースケール")
    print(f"    ノイズ: ガウシアン σ=0.08  (位置・サイズもランダム)")
    print(f"    元の訓練: {len(train_raw.labels)}枚 → データ拡張後: {len(train.labels)}枚 (×4)")
    print(f"    評価: {len(test.labels)}枚 (拡張なし)")
    print(f"    ランダム予測のベースライン: {baseline:.3f}  (1/{n_classes}クラス)")
    print()

    results = []
    done = 0
    for p, f, c in product(PREPROCESSORS, FEATURES, CLASSIFIERS):
        done += 1
        print(f"\r  評価中 [{done:>3}/{total}]  {p:<10}{f:<8}{c:<10}", end="", flush=True)
        try:
            pipe = build_pipeline(p, f, c)
            pipe.fit(train.images, train.labels)
            preds = pipe.predict(test.images)
            acc   = accuracy(test.labels, preds)
            results.append((p, f, c, acc))
        except Exception as e:
            results.append((p, f, c, f"ERR: {type(e).__name__}"))

    print(f"\r  評価完了 [{total}/{total}]" + " " * 40)

    valid = sorted(
        [r for r in results if isinstance(r[3], float)],
        key=lambda r: -r[3],
    )
    errs = [r for r in results if not isinstance(r[3], float)]

    print()
    print("  【結果】")
    print(f"    accuracy : 0.000=全問不正解  {baseline:.3f}=ランダム予測と同等  1.000=全問正解")
    print()
    print(f"  {'preproc':<10}{'feature':<8}{'classifier':<12}{'accuracy':>10}")
    print("  " + "-" * 40)
    for p, f, c, acc in valid:
        marker = " ←最良" if (p, f, c, acc) == valid[0] else ""
        print(f"  {p:<10}{f:<8}{c:<12}{acc:>10.3f}{marker}")
    if errs:
        print("\n  [エラーになった組み合わせ]")
        for p, f, c, msg in errs:
            print(f"    {p}/{f}/{c}: {msg}")

    best_p, best_f, best_c, best_acc = valid[0]
    print()
    print(f"  → 最良構成: {best_p} + {best_f} + {best_c}  accuracy={best_acc:.3f}")
    print(f"    前処理 : {_PREP_DESC[best_p]}")
    print(f"    特徴   : {_FEAT_DESC[best_f]}")
    print(f"    分類器 : {_CLF_DESC[best_c]}")
    return valid


def run_recommended(valid: list):
    """比較結果の最良構成でクラス別詳細レポートを表示"""
    ds = make_shapes(n_per_class=200, size=24, noise=0.08, seed=0)
    train_raw, test = train_test_split(ds, ratio=0.8, seed=1)
    train = augment(train_raw)   # 拡張データで学習

    best_p, best_f, best_c, _ = valid[0]

    print()
    print("=" * 66)
    print("  フェーズ 2/2 : 最良構成でのクラス別詳細評価")
    print("=" * 66)
    print(f"  構成: {best_p}  +  {best_f}  +  {best_c}")
    print(f"    前処理 : {_PREP_DESC[best_p]}")
    print(f"    特徴   : {_FEAT_DESC[best_f]}")
    print(f"    分類器 : {_CLF_DESC[best_c]}")
    print(f"  訓練データ: {len(train.labels)}枚 (データ拡張済み)")
    print()
    print("  学習・評価中...")

    pipe = build_pipeline(best_p, best_f, best_c)
    pipe.fit(train.images, train.labels)
    preds = pipe.predict(test.images)
    acc   = accuracy(test.labels, preds)
    print(f"  → accuracy = {acc:.3f}\n")

    print("  【クラス別レポートの見方】")
    print("    prec(適合率)  : 「このクラスと予測した」うち実際に正解の割合")
    print("    recall(再現率): 「実際にこのクラス」のうち正しく予測できた割合")
    print("    f1            : prec と recall の調和平均")
    print("    support       : 評価データ中のそのクラスのサンプル数")
    print()
    print(classification_report(test.labels, preds, train.label_names))


if __name__ == "__main__":
    valid = run_all_combinations()
    run_recommended(valid)
