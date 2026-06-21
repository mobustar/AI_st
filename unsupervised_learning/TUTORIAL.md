# 教師なし学習 チュートリアル

このチュートリアルでは、教師なし学習の 4 種の手法を順番に学ぶ。

---

## Step 0: 教師あり学習との違い

| | 教師あり学習 | 教師なし学習 |
|--|--|--|
| 学習データ | 入力 + **正解ラベル** | 入力のみ |
| 目標 | ラベルの予測 | データの**構造**の発見 |
| 評価 | 正解率・F1 | シルエット・ARI など |
| 代表例 | 分類・回帰 | クラスタリング・次元削減・異常検知 |

教師なし学習の難しさは「答えが定義されていない」こと。
何が「良いクラスタリング」かはデータと目的による。

---

## Step 1: データの生成 (`data.py`)

```python
from data import make_blobs, make_moons, make_anomaly

# 球状クラスタ (K-means が得意)
ds = make_blobs(n_samples=300, n_clusters=3, cluster_std=0.8, seed=42)
print(ds.X.shape)   # (300, 2)
print(ds.y.shape)   # (300,)  ← 真のラベル (評価用のみ。学習には使わない)

# 三日月形 (DBSCAN が得意)
ds_moons = make_moons(n_samples=300, noise=0.1, seed=42)

# 外れ値入りデータ (異常検知用)
ds_anom = make_anomaly(n_normal=200, n_outliers=20, seed=42)
# y=1: 正常  y=-1: 外れ値
```

**重要**: `ds.y` は評価用の真ラベル。学習時は `ds.X` のみ使う。

---

## Step 2: クラスタリング (`cluster.py`)

### 2-1. K-means

```python
from cluster import get_cluster

km = get_cluster("kmeans", k=3, n_init=10, seed=0)
km.fit(ds.X)

print(km.labels_)     # 各点のクラスタラベル [0, 2, 1, ...]
print(km.inertia_)    # 慣性 (小さいほど良い)
print(km.centroids_)  # 重心座標 shape (3, 2)
```

**K-means++ 初期化の仕組み**:
1. 最初の重心をランダムに選ぶ
2. 以後、既存重心からの距離² に比例した確率で次の重心を選ぶ
3. 「遠い点」が選ばれやすいため、重心が散らばり局所解を回避しやすい

```
距離²が大きい点 → 確率が高い → 次の重心に選ばれやすい
```

**K の選び方 (エルボー法)**:
```python
inertias = []
for k in range(1, 8):
    m = get_cluster("kmeans", k=k, seed=0)
    m.fit(ds.X)
    inertias.append(m.inertia_)
# 急激に下がらなくなる「肘」の k を選ぶ
```

### 2-2. DBSCAN

```python
db = get_cluster("dbscan", eps=0.4, min_samples=5)
db.fit(ds.X)

print(db.n_clusters_)     # 発見されたクラスタ数 (K を指定しなくても自動)
print((db.labels_ == -1).sum())  # ノイズ点の数
```

**3 種類の点の定義**:
- **コア点**: ε 近傍内に `min_samples` 個以上の点がある
- **境界点**: コア点の ε 近傍にあるが、自分はコア点でない
- **ノイズ点**: いずれのクラスタにも属さない (ラベル = -1)

**パラメータ選択のヒント**:
- `eps` が小さすぎる → すべてノイズになる
- `eps` が大きすぎる → すべて 1 クラスタになる
- `min_samples = 2 * 次元数` が経験則

### 2-3. Agglomerative (Ward 連結)

```python
ag = get_cluster("agglomerative", n_clusters=3)
ag.fit(ds.X)
print(ag.labels_)
```

**Ward 連結のコスト計算**:
```
2 クラスタ A, B をマージしたときの SSE 増分:
  cost = (n_A × n_B) / (n_A + n_B) × ||μ_A - μ_B||²

この cost が最小のペアを選んでマージを繰り返す。
```

### 2-4. 評価指標

```python
from metrics import silhouette_score, davies_bouldin_score, adjusted_rand_index

sil = silhouette_score(ds.X, km.labels_)
# +1: 完璧なクラスタリング / 0: クラスタ境界上 / -1: 誤割当

db  = davies_bouldin_score(ds.X, km.labels_)
# 0 に近いほど良い (クラスタ間が広く、クラスタ内が密)

ari = adjusted_rand_index(ds.y, km.labels_)
# 真ラベルとの一致度。1=完全一致 / 0=ランダムと同等
```

**シルエット係数の計算**:
```
各点 i について:
  a(i) = 同クラスタ内の他点との平均距離 (小さいほど密集)
  b(i) = 最近傍の異クラスタとの平均距離 (大きいほど分離)
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

全点の s(i) を平均したものがシルエットスコア
```

---

## Step 3: 次元削減 (`reduce.py`)

### 3-1. PCA (主成分分析)

```python
from reduce import get_reducer

pca = get_reducer("pca", n_components=2)
X_2d = pca.fit_transform(X_high_dim)  # 高次元 → 2D

# 各主成分が全分散の何割を説明するか
print(pca.explained_variance_ratio_)  # e.g., [0.85, 0.10]

# 再構成 (低次元 → 元の次元)
X_recon = pca.inverse_transform(X_2d)
```

**PCA の数学 (SVD を使った実装)**:
```
① 中心化: X_c = X - mean(X)
② SVD:    X_c = U Σ V^T
③ 主成分方向: V の列 (右特異ベクトル)
④ 射影:    X_low = X_c @ V[:, :k]

分散説明率 = σ_i² / Σ_j σ_j²
```

主成分方向は「データが最も広がっている方向」。
第 1 主成分が最大分散方向、第 2 主成分はそれと直交する最大分散方向。

### 3-2. t-SNE

```python
tsne = get_reducer("tsne", n_components=2, perplexity=30, n_iter=1000, seed=42)
X_2d = tsne.fit_transform(X_high_dim)
# ※ 再構成 (inverse_transform) はできない
```

**t-SNE のアルゴリズム**:
```
① 高次元での類似度 p_ij (ガウスカーネル):
   p_j|i = exp(-||x_i-x_j||² / 2σ_i²) / 正規化定数
   p_ij  = (p_j|i + p_i|j) / 2n  (対称化)

② 低次元での類似度 q_ij (t 分布):
   q_ij = (1 + ||y_i-y_j||²)^{-1} / 正規化定数

③ KL ダイバージェンスを最小化:
   C = Σ p_ij * log(p_ij / q_ij)
```

**t 分布を使う理由**:
ガウス分布は裾が薄いため、低次元で離れた点を引き寄せすぎる (crowding problem)。
t 分布は裾が厚いため、クラスタ間のギャップを自然に広げる。

**PCA vs t-SNE の使い分け**:
| | PCA | t-SNE |
|--|--|--|
| 変換の種類 | 線形 | 非線形 |
| 再構成 | 可能 | 不可 |
| グローバル構造 | 保存 | 局所優先 |
| 計算量 | O(n×d²) | O(n²) |
| 用途 | 前処理・再構成 | 可視化 |

---

## Step 4: 異常検知 (`anomaly.py`)

```python
from anomaly import get_anomaly
from metrics import anomaly_metrics

# 正常データのみで学習 (ラベルは使わない)
X_train = ds_anom.X[ds_anom.y == 1]
ifo = get_anomaly("isolation_forest", n_estimators=100, seed=0)
ifo.fit(X_train)

# 全データで予測
# contamination: 異常と見なす割合
y_pred = ifo.predict(ds_anom.X, contamination=0.1)
# +1 = 正常 / -1 = 異常

# 評価
m = anomaly_metrics(ds_anom.y, y_pred)
print(f"F1={m['f1']:.3f}  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}")
```

### 4-1. Isolation Forest の仕組み

```
1. ランダムに特徴量を選ぶ
2. 選んだ特徴量の値をランダムな値で分割
3. 外れ値は少ない分割で孤立する (短いパス)
4. 正常データは多くの分割が必要 (長いパス)

異常スコア = 2^(-平均パス長 / c(n))
c(n): n 点の BST の平均パス長の期待値
```

正規化スコアが 1 に近いほど外れ値、0.5 付近は不明確。

### 4-2. LOF の仕組み

```
局所到達可能密度 (lrd):
  lrd_k(x) = 1 / mean(reach-dist_k(x, o))  ← 大きいほど密度高い

LOF スコア:
  LOF_k(x) = mean(lrd_k(o)) / lrd_k(x)   ← 近傍 o の密度 / 自分の密度

LOF >> 1: 周囲より密度が低い → 外れ値
LOF ≈ 1: 周囲と密度が同じ → 正常
```

**reach-dist の定義**:
```
reach-dist_k(x, o) = max(k-dist(o), dist(x, o))
```
o の k 近傍距離より短い距離は k 近傍距離で「切り上げ」することで
近距離点へのスコアの不安定性を防ぐ。

### 4-3. 評価指標

```
Precision = TP / (TP + FP)  # 予測した異常のうち本物の割合
Recall    = TP / (TP + FN)  # 本物の異常のうち検出できた割合
F1        = 2 * P * R / (P + R)
```

TP: 本物の異常 → 異常と予測 (True Positive)
FP: 正常点 → 異常と予測 (False Positive → 誤警報)
FN: 本物の異常 → 正常と予測 (False Negative → 見逃し)

---

## Step 5: 変分オートエンコーダ (`vae.py`)

### 5-1. 通常のオートエンコーダとの違い

```
通常のオートエンコーダ:
  x → Encoder → z (1点) → Decoder → x̂

VAE:
  x → Encoder → (μ, σ²) → z ~ N(μ,σ²) → Decoder → x̂
                  ↑確率分布          ↑サンプリング
```

VAE の潜在変数 z は点ではなく「分布」。
これにより潜在空間が連続・滑らかになり、
z をサンプリングして新しいデータを生成できる。

### 5-2. 再パラメータ化トリック

サンプリング `z ~ N(μ, σ²)` はそのままでは微分できない。

```python
# サンプリングをこう書き換える:
eps = torch.randn_like(sigma)   # ε ~ N(0,I)
z   = mu + sigma * eps          # z = μ + σ・ε

# eps は定数扱い → μ と σ に対して勾配が流れる
```

### 5-3. ELBO 損失

```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
     = -(再構成損失)  - (KL 正則化)

再構成損失: BCE(x̂, x) ← 入力を忠実に再現
KL 正則化 : -0.5 Σ (1 + log σ² - μ² - σ²) ← 潜在空間を N(0,I) に近づける
```

KL 正則化がなければ VAE は「特定の点に情報を押し込む」通常の AE になる。
KL 正則化により潜在空間が整理され、補間・生成が可能になる。

### 5-4. 使い方

```python
from vae import VAE, VAETrainer
import numpy as np

# データを [0, 1] に正規化 (Sigmoid 出力に合わせる)
X = (X - X.min()) / (X.max() - X.min() + 1e-8)

# モデルと学習
model   = VAE(input_dim=2, hidden_dim=64, latent_dim=2)
trainer = VAETrainer(model, lr=1e-3, batch_size=64)
losses  = trainer.train(X, n_epochs=200, beta=1.0, verbose_every=50)

# 新しいデータの生成 (z ~ N(0,I) からサンプリング)
X_gen = model.sample(100)

# 潜在空間への埋め込み (可視化用)
Z = model.encode_numpy(X)   # shape (n, latent_dim)
```

**β-VAE について**:
β > 1 にすると KL 正則化の重みが増え、潜在変数の「絡み合い」が減る
(各次元が独立した意味を持ちやすくなる)。再構成精度は下がる。

---

## まとめ: アルゴリズム選択ガイド

### クラスタリング

| データの形状 | 推奨アルゴリズム |
|--|--|
| 球状・等サイズ | K-means |
| 任意形状・ノイズあり | DBSCAN |
| 階層構造を知りたい | Agglomerative |
| K が不明 | DBSCAN (自動推定) |

### 次元削減

| 目的 | 推奨 |
|--|--|
| 前処理・再構成・解釈 | PCA |
| 可視化のみ | t-SNE |
| 線形構造の保存 | PCA |
| 非線形構造 (多様体) | t-SNE |

### 異常検知

| 状況 | 推奨 |
|--|--|
| グローバルな外れ値 | Isolation Forest |
| 密度差のある複数クラスタ混在 | LOF |
| 大規模データ (n > 1000) | Isolation Forest |
| 小規模・局所外れ値 | LOF |
