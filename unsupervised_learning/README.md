# 教師なし学習モジュール

ラベルなしデータから構造を発見する 4 種の教師なし学習手法を実装。
**NumPy のみ** で動作 (VAE のみ PyTorch を使用)。

---

## 概要

| フェーズ | 手法 | 目的 |
|----------|------|------|
| 1 | クラスタリング (K-means / DBSCAN / Agglomerative) | データを「似たグループ」に自動分類 |
| 2 | 次元削減 (PCA / t-SNE) | 高次元データを 2D に圧縮して可視化 |
| 3 | 異常検知 (Isolation Forest / LOF) | 正常パターンから外れた点を検出 |
| 4 | 生成モデル (VAE) | 潜在空間を学習して新しいデータを生成 |

教師あり学習との最大の違いは **ラベル (正解) を使わない** 点。
データの内部構造を自律的に発見する。

---

## ディレクトリ構成

```
unsupervised_learning/
├── data.py       データセット生成 (blobs / moons / circles / anomaly / swiss_roll)
├── cluster.py    クラスタリング 3 種
├── reduce.py     次元削減 2 種
├── anomaly.py    異常検知 2 種
├── vae.py        変分オートエンコーダ (PyTorch)
├── metrics.py    評価指標 (シルエット / ARI / 異常検知 F1 など)
├── visualize.py  ASCII 散布図・損失曲線 (matplotlib 不使用)
├── main.py       4 フェーズ一括実行
├── README.md     ← このファイル
└── TUTORIAL.md   各アルゴリズムの詳細解説
```

---

## 実行

```bash
cd unsupervised_learning
python main.py
```

PyTorch が未導入の場合は VAE フェーズのみスキップされ、残り 3 フェーズは実行される。

---

## アルゴリズム一覧

### クラスタリング (`cluster.py`)

```
Env × Algorithm で形状ごとの適性を比較
─────────────────────────────────────────────────────
アルゴリズム         │ 球状クラスタ │ 三日月形 │ 同心円
──────────────────────┼──────────────┼──────────┼───────
K-means (K-means++ )  │ ◎            │ △        │ △
DBSCAN                │ ◯            │ ◎        │ ◎
Agglomerative (Ward)  │ ◎            │ △        │ △
─────────────────────────────────────────────────────
```

**K-means**
- K-means++ 初期化で局所解リスクを低減
- `n_init` 回試行して最良 (慣性最小) を選択
- クラスタ数 K を事前指定する必要がある

**DBSCAN**
- ε-近傍の点数が `min_samples` 以上の点を「コア点」として定義
- コア点から密度連結なすべての点をクラスタに取り込む
- クラスタ数の指定不要、ノイズ点 (ラベル = -1) を自動除外

**Agglomerative Clustering (Ward 連結)**
- 各点を独立クラスタとして開始し、マージコストが最小のペアを統合
- Ward 連結: マージ時の SSE 増分をコストとして使用
- 計算量 O(n³) のため n ≲ 1000 推奨

---

### 次元削減 (`reduce.py`)

**PCA (主成分分析)**
- データの分散が最大になる方向 (主成分) を SVD で計算
- 線形変換のため `inverse_transform` で元次元に再構成可能
- `explained_variance_ratio_` で各主成分の情報量を確認できる

```
X (n×d)  →  [PCA]  →  X_low (n×k)  →  [inverse_transform]  →  X_recon (n×d)
```

**t-SNE**
- 高次元の「近さ」(ガウス確率) と低次元の「近さ」(t 分布確率) の
  KL ダイバージェンスを勾配降下で最小化
- t 分布の裾の厚さが「crowding problem」を解決する
- 再構成不可。可視化専用
- O(n²) 実装のため n ≲ 500 推奨

---

### 異常検知 (`anomaly.py`)

**Isolation Forest**
- ランダム分割木 (孤立木) を構築し、各点が孤立するまでのパス長を測る
- 外れ値は少ない分割で孤立する → パスが短い → スコア小
- グローバルな外れ値 (全体の傾向からの逸脱) の検出が得意

**LOF (Local Outlier Factor)**
- 自分と近傍の局所密度 (lrd) を比較する比率で外れ度を測る
- `LOF ≫ 1` の点は周囲より密度が低い → 外れ値
- 局所的な外れ値 (密度が異なる複数クラスタが混在する場合) に強い

```
predict(X, contamination=0.1)
  → 異常スコアの下位 10% を -1 (異常)、残りを +1 (正常) として返す
```

---

### VAE (`vae.py`)

変分オートエンコーダ (Variational Autoencoder)。PyTorch 使用。

```
Encoder: x → Linear → ReLU → (μ, log σ²)
                                   ↓ 再パラメータ化: z = μ + σ·ε
Decoder: z → Linear → ReLU → Linear → Sigmoid → x̂
```

**ELBO 損失** (Evidence Lower BOund):
```
loss = 再構成損失 (BCE) + β × KL(N(μ,σ²) || N(0,I))
```
- 再構成損失: 入力と出力の近さ
- KL 正則化: 潜在空間を標準正規分布に近づける

**再パラメータ化トリック**: `z = μ + σ·ε`、`ε ~ N(0,I)`
- z のサンプリングを μ と σ だけで表すことでバックプロパゲーションを可能にする

---

## 評価指標 (`metrics.py`)

| 指標 | 範囲 | 解釈 |
|------|------|------|
| シルエットスコア | -1 ~ +1 | 高いほど良い |
| Davies-Bouldin 指数 | 0 ~ ∞ | 小さいほど良い |
| ARI (調整ランド指数) | -1 ~ +1 | 1=真ラベルと完全一致 |
| Precision (異常検知) | 0 ~ 1 | 予測異常のうち本物の割合 |
| Recall (異常検知) | 0 ~ 1 | 本物の異常のうち検出できた割合 |
| F1 (異常検知) | 0 ~ 1 | Precision と Recall の調和平均 |
| Trustworthiness | 0 ~ 1 | 近傍関係の保存度 |

---

## 依存ライブラリ

```
numpy       (必須: 全モジュール)
torch       (任意: vae.py のみ。pip install torch)
```
