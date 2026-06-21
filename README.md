# AI 学習ディレクトリ

NumPy を中心に実装した機械学習の 4 モジュール。  
各モジュールは独立して動作し、アルゴリズムの組み合わせを網羅的に比較できる。

| ディレクトリ | 内容 | アルゴリズム数 |
|---|---|---|
| `text_ml/` | テキスト分類（トークナイザ × TF-IDF × 分類器）| 4×4×8 = 128 通り |
| `object_recognition/` | 物体認識（前処理 × HOG特徴 × 分類器 + データ拡張）| 4×5×7 = 140 通り |
| `reinforcement_learning/` | 強化学習（環境 × 方策 × アルゴリズム + アンサンブル）| 4×4×8 = 128 通り |
| `unsupervised_learning/` | 教師なし学習（クラスタリング / 次元削減 / 異常検知 / VAE）| 8 手法 |

---

## モジュール概要

### text_ml — テキスト分類

```
入力テキスト (日本語)
    → トークナイザ (4種: whitespace / char_ngram / regex / char)
    → ベクトル化 (4種: CountVec / TF-IDF / Hashing / BM25)
    → 分類器 (8種: NB / LogReg / SVM / kNN / Perceptron / Ridge / PyTorchMLP / Ensemble)
    → 予測クラス
```

- バイナリ分類器は `OneVsRestClassifier` で多クラス対応
- `VotingEnsembleClassifier` (NB + Ridge + LogReg の多数決) で精度向上
- `predict.py` で保存済み PyTorch MLP モデルを使った対話型推論

### object_recognition — 物体認識

```
合成図形 (circle / square / triangle)
    → データ拡張 (反転・回転で 4倍化)
    → 前処理 (4種: NoOp / MinMax / Standard / HistEQ)
    → 特徴抽出 (5種: Raw / HOG / LBP / EdgeHist / DCT)
    → 分類器 (7種: kNN / Softmax / SVM / MLP / GNB / LDA / Ensemble)
    → 予測クラス
```

- データ拡張（左右反転・上下反転・180度回転）で汎化性能向上
- `VotingEnsembleClassifier` (Softmax + MLP + LDA の多数決) で精度向上

### reinforcement_learning — 強化学習

```
環境 (4種: GridWorld / CliffWalk / StochasticGrid / WindyGridWorld)
    × 方策 (4種: ε-greedy / 減衰ε-greedy / Boltzmann / UCB)
    × アルゴリズム (8種: MC / SARSA / Q / ExpSARSA / DoubleQ / DynaQ / NStepSARSA / SARSA(λ))
    → Q テーブル学習 → アンサンブル評価（7体のQ平均）
```

### unsupervised_learning — 教師なし学習

```
ラベルなしデータ
    → クラスタリング  (3種: K-means / DBSCAN / Agglomerative)
    → 次元削減        (2種: PCA / t-SNE)
    → 異常検知        (2種: Isolation Forest / LOF)
    → 生成モデル      (1種: VAE)
```

- ラベルを一切使わずにデータの構造を自動発見
- ASCII 散布図で結果を端末上に可視化
- VAE は PyTorch を使用 (CUDA 自動検出)

---

## セットアップ

```bash
# 1. 仮想環境を作成
python3 -m venv .venv

# 2. 仮想環境を有効化
source .venv/bin/activate

# 3. 依存ライブラリをインストール
pip install -r requirements.txt
```

> `requirements.txt` には `numpy` と `torch` が含まれる。  
> PyTorch MLP (`mlp` 分類器) を使わない場合は torch は不要。

---

## 実行方法

仮想環境を有効化した状態で、各ディレクトリに移動して `main.py` を実行する。

### テキスト分類

```bash
cd text_ml
python main.py
```

```
フェーズ 1/3 : 全アルゴリズム組み合わせ比較 (128通り)
  tokenizer   vectorizer  classifier     accuracy
  -----------------------------------------------
  char_ngram  bm25        ensemble          0.950
  char_ngram  tfidf       ensemble          0.933
  char_ngram  tfidf       svm               0.923
  ...

フェーズ 2/3 : 最良構成でのクラス別詳細評価

フェーズ 3/3 : PyTorch MLP でモデル保存 → model.pt
```

### 物体認識

```bash
cd object_recognition
python main.py
```

```
フェーズ 1/2 : 全アルゴリズム組み合わせ比較 (140通り)
  preproc   feature   classifier     accuracy
  -------------------------------------------
  standard  hog       ensemble          0.962
  standard  edge      softmax           0.942
  ...

フェーズ 2/2 : 最良構成でのクラス別詳細評価
```

### 強化学習

```bash
cd reinforcement_learning
python main.py
```

```
フェーズ 1/2 : 全アルゴリズム組み合わせ比較 (128通り × 3 seed)
  ┌─ 環境: gridworld
  │  policy      algorithm        mean   ± std
  │  decay_eps   dyna_q          0.832    0.021  ←最良
  ...

フェーズ 2/2 : 各環境の最良構成を精度高く再評価
  gridworld  単体: mean=0.821 ± 0.034
             アンサンブル: mean=0.843 ± 0.019  ↑改善
```

### 教師なし学習

```bash
cd unsupervised_learning
python main.py
```

```
フェーズ 1/4 : クラスタリング
  ─── blobs (球状クラスタ) ───
  K-means: sil=+0.627  DB=0.526  ARI=+0.951
  DBSCAN : sil=+0.581  DB=0.279  ARI=+0.564
  ...

フェーズ 2/4 : 次元削減 (PCA / t-SNE)
フェーズ 3/4 : 異常検知 (IsolationForest / LOF)
フェーズ 4/4 : VAE 生成モデル
```

---

## 対話型テキスト推論

`text_ml/main.py` を実行するとモデルが `model.pt` に保存される。  
その後、`predict.py` で任意の文章を分類できる。

```bash
cd text_ml
python predict.py
# または
python predict.py "この料理は最高でした"
```

---

## 仮想環境の終了

```bash
deactivate
```

---

## 学習リソース

| ファイル | 内容 |
|---|---|
| `text_ml/README.md` | トークナイザ・TF-IDF・BM25・各分類器の数式定義 |
| `text_ml/TUTORIAL.md` | OvR・アンサンブル・実験課題（コピペで動くコード付き）|
| `object_recognition/README.md` | HOG・LBP・DCT・前処理・分類器の数式定義 |
| `object_recognition/TUTORIAL.md` | データ拡張・アンサンブル・実験課題 |
| `reinforcement_learning/README.md` | MDP・各アルゴリズムの更新式・Q平均アンサンブル |
| `reinforcement_learning/TUTORIAL.md` | ベルマン方程式・SARSA vs Q-learning・実験課題 |
| `unsupervised_learning/README.md` | 各手法のアルゴリズム・指標・選択ガイド |
| `unsupervised_learning/TUTORIAL.md` | K-means++・t-SNE・LOF・VAE の数式と実装解説 |
