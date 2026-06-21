"""
==============================================================
  main.py | 全アルゴリズム組み合わせ比較
==============================================================
Tokenizer × Vectorizer × Classifier の全組み合わせを評価する。
4 × 4 × 8 = 128 通り。

フェーズ:
  1. 全 128 通りを評価し accuracy 順にランキング表示
  2. 最良構成でクラス別詳細レポート (precision / recall / F1)
  3. PyTorch MLP モデルを保存 (model.pt)

実行:
    python main.py
"""

from itertools import product
from pathlib import Path

from data       import load_dataset
from tokenizer  import get_tokenizer
from vectorizer import get_vectorizer
from classifier import get_classifier
from pipeline   import TextClassificationPipeline
from metrics    import accuracy, classification_report


# ─── 比較する戦略の組み合わせ ────────────────────────────────
TOKENIZERS  = ["whitespace", "char_ngram", "regex", "char"]
VECTORIZERS = ["count", "tfidf", "hashing", "bm25"]
CLASSIFIERS = ["nb", "logreg", "svm", "knn", "perceptron", "ridge", "mlp", "ensemble"]

_TOK_DESC = {
    "whitespace": "空白区切り (日本語には不向き)",
    "char_ngram": "文字N-gram n=(2,3) (日本語に最も頑健)",
    "regex":      "文字種で分割 (漢字/かな/英数を別トークンに)",
    "char":       "1文字ずつ (語彙最小)",
}
_VEC_DESC = {
    "count":   "出現回数 (Bag-of-Words)",
    "tfidf":   "TF-IDF (希少語を強調)",
    "hashing": "ハッシュ固定次元 (語彙不要・衝突あり)",
    "bm25":    "BM25 (TF-IDFの改良版・飽和処理付き)",
}
_CLF_DESC = {
    "nb":         "多項ナイーブベイズ (小データに強い)",
    "logreg":     "ロジスティック回帰 + OvR",
    "svm":        "線形SVM + OvR (高次元疎ベクトルに強い)",
    "knn":        "k近傍法 k=5 (コサイン距離)",
    "perceptron": "平均化パーセプトロン + OvR",
    "ridge":      "リッジ回帰分類 + OvR (閉形式解)",
    "mlp":        "PyTorch MLP 2層 (GPU自動使用)",
    "ensemble":   "Voting Ensemble: NB + Ridge + LogReg の多数決",
}


def build_pipeline(tok_name: str, vec_name: str, clf_name: str):
    if tok_name == "char_ngram":
        tokenizer = get_tokenizer(tok_name, ns=(2, 3))
    else:
        tokenizer = get_tokenizer(tok_name)

    if vec_name == "hashing":
        vectorizer = get_vectorizer(vec_name, n_features=2**12)
    else:
        vectorizer = get_vectorizer(vec_name, min_df=1)

    classifier = get_classifier(clf_name)
    return TextClassificationPipeline(tokenizer, vectorizer, classifier)


def run_all_combinations():
    """全組み合わせを評価し、精度順のリストを返す"""
    train, test = load_dataset(seed=42)
    n_classes  = len(train.label_names)
    total      = len(TOKENIZERS) * len(VECTORIZERS) * len(CLASSIFIERS)
    baseline   = 1.0 / n_classes

    print("=" * 62)
    print("  フェーズ 1/2 : 全アルゴリズム組み合わせ比較")
    print("=" * 62)
    print(f"  【処理内容】")
    print(f"    テキストを 3段階で変換して分類します。")
    print(f"      ① トークナイザ  : テキスト → トークン列")
    print(f"      ② ベクトライザ  : トークン列 → 数値ベクトル")
    print(f"      ③ 分類器        : ベクトル → カテゴリ予測")
    print(f"    {len(TOKENIZERS)}種 × {len(VECTORIZERS)}種 × {len(CLASSIFIERS)}種 = {total}通りを全評価します。")
    print()
    print(f"  【データ】")
    print(f"    訓練: {len(train.texts)}件  評価: {len(test.texts)}件")
    print(f"    カテゴリ({n_classes}クラス): {' / '.join(train.label_names)}")
    print(f"    ランダム予測のベースライン: {baseline:.3f}  (1/{n_classes})")
    print()

    results = []
    done = 0
    for tok, vec, clf in product(TOKENIZERS, VECTORIZERS, CLASSIFIERS):
        done += 1
        print(f"\r  評価中 [{done:>3}/{total}]  {tok:<12}{vec:<10}{clf:<12}", end="", flush=True)
        try:
            pipe = build_pipeline(tok, vec, clf)
            pipe.fit(train.texts, train.labels)
            preds = pipe.predict(test.texts)
            acc   = accuracy(test.labels, preds)
            results.append((tok, vec, clf, acc))
        except Exception as e:
            results.append((tok, vec, clf, f"ERR: {type(e).__name__}"))

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
    print(f"  {'tokenizer':<12}{'vectorizer':<12}{'classifier':<14}{'accuracy':>10}")
    print("  " + "-" * 48)
    for tok, vec, clf, acc in valid:
        marker = " ←最良" if (tok, vec, clf, acc) == valid[0] else ""
        print(f"  {tok:<12}{vec:<12}{clf:<14}{acc:>10.3f}{marker}")
    if errs:
        print("\n  [エラーになった組み合わせ]")
        for tok, vec, clf, msg in errs:
            print(f"    {tok}/{vec}/{clf}: {msg}")

    best_tok, best_vec, best_clf, best_acc = valid[0]
    print()
    print(f"  → 最良構成: {best_tok} + {best_vec} + {best_clf}  accuracy={best_acc:.3f}")
    print(f"    ({_TOK_DESC[best_tok]} / {_VEC_DESC[best_vec]} / {_CLF_DESC[best_clf]})")
    return valid


def run_best_pipeline_detail(tok_name: str, vec_name: str, clf_name: str):
    """最良構成で詳細レポートを表示する"""
    train, test = load_dataset(seed=42)

    print()
    print("=" * 62)
    print("  フェーズ 2/2 : 最良構成での詳細評価")
    print("=" * 62)
    print(f"  構成: {tok_name}  +  {vec_name}  +  {clf_name}")
    print(f"    トークナイザ : {_TOK_DESC[tok_name]}")
    print(f"    ベクトライザ : {_VEC_DESC[vec_name]}")
    print(f"    分類器       : {_CLF_DESC[clf_name]}")
    print()
    print("  学習・評価中...")

    pipe = build_pipeline(tok_name, vec_name, clf_name)
    pipe.fit(train.texts, train.labels)
    preds = pipe.predict(test.texts)
    acc   = accuracy(test.labels, preds)
    print(f"  → accuracy = {acc:.3f}\n")

    print("  【クラス別レポートの見方】")
    print("    prec(適合率)  : 「このクラスと予測した」うち実際に正解の割合")
    print("    recall(再現率): 「実際にこのクラス」のうち正しく予測できた割合")
    print("    f1            : prec と recall の調和平均 (両方高いほど良い)")
    print("    support       : 評価データ中のそのクラスのサンプル数")
    print()
    print(classification_report(test.labels, preds, train.label_names))

    print()
    print("  【予測サンプル (評価データ先頭5件)】")
    for text, true, pred in zip(test.texts[:5], test.labels[:5], preds[:5]):
        mark = "○正解" if true == pred else "×誤答"
        print(f"    {mark}  真={train.label_names[true]:<8} 予={train.label_names[pred]:<8} : {text}")


def save_best_model(tok_name: str, vec_name: str):
    """最良 tokenizer+vectorizer で PyTorch MLP を学習し text_ml/ に保存する"""
    train, test = load_dataset(seed=42)

    print()
    print("=" * 62)
    print("  フェーズ 補足 : PyTorch MLP モデルの保存")
    print("=" * 62)
    print(f"  構成: {tok_name} + {vec_name} + mlp")
    print(f"  ({_TOK_DESC[tok_name]} / {_VEC_DESC[vec_name]})")
    print("  学習中...")

    pipe = build_pipeline(tok_name, vec_name, "mlp")
    pipe.fit(train.texts, train.labels)
    preds = pipe.predict(test.texts)
    acc   = accuracy(test.labels, preds)
    print(f"  評価 accuracy = {acc:.3f}")

    save_path = Path(__file__).parent / "model.pt"
    pipe.classifier.save(str(save_path))
    print(f"  → predict.py で読み込むことで任意のテキストを分類できます。")


if __name__ == "__main__":
    ranked = run_all_combinations()

    best_tok, best_vec, best_clf, _ = ranked[0]
    run_best_pipeline_detail(best_tok, best_vec, best_clf)
    save_best_model(best_tok, best_vec)
