# object_recognition

NumPy **のみ**で実装した画像分類のフルパイプライン。
3 種類の図形（円・正方形・三角形）を、前処理 → 特徴抽出 → 分類器 の組み合わせで認識する。

---

## パイプライン概要

```
入力画像 (24×24 グレースケール)
        │
        ▼
┌───────────────────┐
│   Preprocessor    │  画素値の正規化 (4 種)
│ noop / minmax /   │
│ standard / hist   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ FeatureExtractor  │  画像 → 固定長ベクトル (4 種)
│ raw / hog /       │
│ lbp / edge        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    Classifier     │  多クラス分類 (7 種)
│ knn / softmax /   │
│ svm / mlp /       │
│ gnb / lda /       │
│ ensemble          │
└────────┬──────────┘
         │
         ▼
   予測クラス
  circle / square / triangle
```

各コンポーネントは**戦略パターン**で実装されており、名前を指定するだけで差し替えられる。

---

## ディレクトリ構成

```
object_recognition/
├── data.py        # 合成図形データセット生成 + データ拡張 (augment)
├── preprocess.py  # 前処理 4 種（NoOp / MinMax / Standard / HistEQ）
├── feature.py     # 特徴抽出 5 種（Raw / HOG / LBP / EdgeHist / DCT）
├── classifier.py  # 多クラス分類器 7 種（kNN / Softmax / SVM / MLP / GNB / LDA / Ensemble）
├── metrics.py     # 評価指標（混同行列・Precision / Recall / F1）
├── pipeline.py    # 上記を統合する ObjectRecognitionPipeline
└── main.py        # 全 140 通り比較 + 推奨構成の詳細評価
```

---

## データセット（data.py）

### 生成される図形

```
circle       square      triangle
  ████         ████         █
 ██████       ██████       ███
████████     ████████     █████
 ██████       ██████       ███
  ████         ████          █
```

| 項目 | 値 |
|------|-----|
| クラス | circle (0) / square (1) / triangle (2) |
| 画像サイズ | 24×24 px グレースケール（値域 [0, 1]）|
| デフォルト枚数 | 各クラス 200 枚（計 600 枚）|
| ランダム化 | 位置・サイズ・ガウシアンノイズ（σ=0.08）|
| 訓練/評価分割 | 80:20 |
| データ拡張後 | 元の訓練データの 4 倍（480枚 → 1920枚）|

### データ拡張（augment 関数）

```
図形は反転・回転後も同じクラス（円は回転しても円）のため、
同じラベルで拡張データを作れる。

各画像から 4 枚を生成:
  1. 元画像                img
  2. 左右反転              img[:, ::-1]
  3. 上下反転              img[::-1, :]
  4. 180度回転（上下左右） img[::-1, ::-1]

訓練データを 4 倍に増やすことで:
  - 位置・向き変動への汎化性能が向上する
  - モデルが「反転しても同じ図形」という不変性を学べる

注意: テストデータには拡張を適用しない（実際の評価条件を模擬）。
```

---

## Preprocessor（preprocess.py）— 4 種

画素値の分布を整えて後段モデルの収束・精度を向上させる。

```
元データ（0付近に偏った分布）      Standard（平均0/分散1）
  ████                               ─────┼──────────
  ████████    →    MinMax →          μ=0
  ████████████  [0,1]に伸張
```

---

### NoOp（変換なし）

**定義**:

```
f(x) = x

すべての画素値を float32 にキャストして返すだけ。
変換は一切行わない。
```

**いつ使うか**: データが既に `[0, 1]` 正規化済みの場合。比較実験で「前処理なし」のベースラインを作る場合。

---

### MinMaxScaler（最小最大正規化）

**定義**:

```
【学習】
  x_min = min_{i,j} x_{ij}       （訓練データ全体の最小画素値）
  x_max = max_{i,j} x_{ij}       （訓練データ全体の最大画素値）

【変換】
  x'_{ij} = (x_{ij} − x_min) / (x_max − x_min)

結果: x'_{ij} ∈ [0, 1]
数値安定のため分母に ε=1e-12 を加算: x' = (x − x_min) / max(x_max − x_min, ε)
```

**利点**: 結果が `[0, 1]` に収まる。直感的。
**欠点**: 外れ値 1 つでスケール全体が歪む。

---

### StandardScaler（標準化）

**定義**:

```
【学習】
  μ = (1/N) Σ_{i,j} x_{ij}       （全画素の平均）
  σ = √( (1/N) Σ_{i,j} (x_{ij} − μ)² )  （全画素の標準偏差）

【変換】
  x'_{ij} = (x_{ij} − μ) / (σ + ε)     ε=1e-12（0除算防止）

結果: E[x'] ≈ 0、Var[x'] ≈ 1
```

全画素を 1 つの統計量で正規化する（画像個別ではなくデータセット全体の統計）。
クラス間で同じスケール感が保たれる。

**いつ使うか**: **一般的な用途で最も推奨**。線形モデル・MLP・SVM との相性が良い。

---

### HistogramEqualizer（ヒストグラム均等化）

**定義**:

```
各画像 I を独立に処理する。

【量子化】
  k_{ij} = clip(round(I_{ij} × (B−1)), 0, B−1)   B=256 ビン

【ヒストグラム】
  h[k] = | {(i,j) : k_{ij} = k} |   （各値の画素数）

【累積分布関数 (CDF)】
  CDF[k] = (Σ_{m=0}^{k} h[m]) / (H × W)    （[0, 1] に正規化）

【変換】
  I'_{ij} = CDF[k_{ij}]

結果: 画素値の分布が [0, 1] 上でほぼ均一になる
```

**利点**: 照明変化・暗部つぶれに対して頑健。
**いつ使うか**: 実世界の画像で照明条件がばらつく場合。

> **ファクトリ関数**: `get_preprocessor("standard")`

---

## FeatureExtractor（feature.py）— 4 種

画像（H×W）を固定長ベクトルに変換する。**精度に最も大きく影響するコンポーネント**。

| 名前 | クラス | 特徴ベクトル長 | 精度（図形認識）|
|------|--------|:-----------:|:--------:|
| `raw` | `RawPixels` | 576 | △ ベースライン |
| `hog` | `HOG` | ブロック数 × 18 | ◎ 形状に強い |
| `lbp` | `LBP` | 256 | △ テクスチャに強い |
| `edge` | `EdgeHist` | 16 | ◎ **軽量・高精度** |
| `dct` | `DCT` | n_components | ○ 周波数情報 |

---

### RawPixels（生画素）

**定義**:

```
入力: I ∈ ℝ^{H×W}

出力: φ(I) = vec(I) ∈ ℝ^{H×W}   （行方向に flatten）

φ(I)_k = I_{⌊k/W⌋, k mod W}   k = 0, 1, ..., H×W−1
```

**利点**: 実装が最もシンプル。全情報を保持。
**欠点**: 位置不変性がない。次元数が H×W=576 と大きい。

---

### HOG（Histogram of Oriented Gradients）

**定義**:

```
入力: I ∈ ℝ^{H×W}
パラメータ: n_bins=9, cell_size=4, block_size=2

【Step 1: 勾配計算（中央差分）】
  g_y[y,x] = I[y+1,x] − I[y−1,x]      （端は 0 で補完）
  g_x[y,x] = I[y,x+1] − I[y,x−1]

  m[y,x] = √(g_x² + g_y²)              （勾配の大きさ）
  θ[y,x] = atan2(g_y, g_x) mod π       （方向 ∈ [0, π)）

【Step 2: セルヒストグラム】
  セル (p, q) の範囲: rows [p·c, (p+1)·c),  cols [q·c, (q+1)·c)   c = cell_size

  ビン幅: Δ = π / n_bins

  h[p,q,b] = Σ_{(y,x)∈cell(p,q)}  m[y,x] · 𝟙[b·Δ ≤ θ[y,x] < (b+1)·Δ]

  （magnitude を重みとして方向ビンに加算）

【Step 3: ブロック正規化（L2-Hys）】
  ブロック (r, s) を構成するセルのヒストグラムを連結:
    v = [h[r,s,:], h[r,s+1,:], h[r+1,s,:], h[r+1,s+1,:]] ∈ ℝ^{4·n_bins}

  L2 正規化:  v ← v / √(‖v‖² + ε²)
  クリップ:   v ← min(v, 0.2)
  再正規化:   v ← v / √(‖v‖² + ε²)       （ε=1e-6）

【Step 4: 最終特徴】
  全ブロックの正規化済みベクトルを連結
  ブロック数: (cy − block+1) × (cx − block+1)   cy=H/c, cx=W/c

出力次元: ブロック数 × block_size² × n_bins
```

**利点**: 形状・輪郭認識に強い。局所コントラスト変化に頑健。
**いつ使うか**: 形のエッジが重要な認識タスク。

---

### LBP（Local Binary Pattern）

**定義**:

```
入力: I ∈ ℝ^{H×W}

【Step 1: LBP コード計算】
  8 近傍の相対座標（時計回り）:
    Δ = [(−1,−1),(−1,0),(−1,1),(0,1),(1,1),(1,0),(1,−1),(0,−1)]

  LBP[y,x] = Σ_{k=0}^{7}  𝟙[I[y+Δk_r, x+Δk_c] ≥ I[y,x]] · 2^k

  LBP[y,x] ∈ {0, 1, ..., 255}
  （有効範囲: y ∈ [1, H−2], x ∈ [1, W−2]）

【Step 2: ヒストグラム】
  H_LBP[b] = |{(y,x) : LBP[y,x] = b}|   b = 0, 1, ..., 255

【Step 3: 正規化】
  φ(I) = H_LBP / ( Σ_b H_LBP[b] + ε )  ∈ ℝ^{256}   （L1 正規化）
```

**利点**: 照明変化（輝度のオフセット）に不変。計算が軽い。
**欠点**: 大域的な形状情報を持たない。図形認識では HOG に劣る。

---

### EdgeHist（エッジ方向ヒストグラム）

**定義**:

```
入力: I ∈ ℝ^{H×W}
パラメータ: n_bins=16

【Step 1: Sobel フィルタによる勾配計算】
  K_x = [[-1,0,1],[-2,0,2],[-1,0,1]]   （水平エッジ検出）
  K_y = [[-1,-2,-1],[0,0,0],[1,2,1]]   （垂直エッジ検出）

  g_x = I * K_x   （3×3 畳み込み、端は 0 パディング）
  g_y = I * K_y

  m[y,x] = √(g_x[y,x]² + g_y[y,x]²)
  θ[y,x] = atan2(g_y[y,x], g_x[y,x]) mod π   ∈ [0, π)

【Step 2: 重み付きヒストグラム】
  ビン幅: Δ = π / n_bins

  H[b] = Σ_{y,x}  m[y,x] · 𝟙[b·Δ ≤ θ[y,x] < (b+1)·Δ]

【Step 3: 正規化】
  φ(I) = H / ( Σ_b H[b] + ε )  ∈ ℝ^{n_bins}   （L1 正規化）
```

**利点**: 16 次元と非常に短い。図形の方向分布を的確に捉える。
**いつ使うか**: 形が単純でエッジ方向の分布がクラスごとに明確なタスク。**推奨手法**。

---

### DCT（離散コサイン変換）

**定義**:

```
入力: I ∈ ℝ^{H×W}
パラメータ: n_components=32

【Step 1: 2D DCT-II】
  変換行列 D_N ∈ ℝ^{N×N}:
    D_N[k, n] = sqrt(2/N) · cos(π k (2n+1) / (2N))   k = 1,...,N−1
    D_N[0, n] = sqrt(1/N)                              k = 0

  F = D_H · I · D_W^T     ∈ ℝ^{H×W}   （H 方向→W 方向の順に変換）

【Step 2: ジグザグスキャン】
  F の左上 (低周波) から対角線を折れ曲がりながら走査し
  n_components 個の係数を抽出:

  zigzag 順: (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),...

  φ(I) = [ F[zigzag[0]], F[zigzag[1]], ..., F[zigzag[n−1]] ] ∈ ℝ^n
```

**利点**: JPEG 圧縮と同じ変換。少数係数で画像の主要情報を保持。
**欠点**: 図形認識では HOG・EdgeHist より精度は低い傾向。

> **ファクトリ関数**: `get_feature("edge")`

---

## Classifier（classifier.py）— 7 種

特徴ベクトルからクラスを予測する。すべて**3 クラス対応**（circle / square / triangle）。

---

### kNN（k 近傍法）

**定義**:

```
訓練データ: {(x_i, y_i)}_{i=1}^{N}   x_i ∈ ℝ^d, y_i ∈ {0,...,K−1}

【コサイン距離】
  dist_cos(x, x_i) = 1 − (x · x_i) / (‖x‖ · ‖x_i‖)

  ‖x‖ = 0 の場合は分母を 1 とする（ゼロベクトル保護）

【k 近傍の取得】
  N_k(x) = k 個の最近傍インデックスの集合
          = argmin_{S⊆[N], |S|=k} Σ_{i∈S} dist_cos(x, x_i)

【距離重み付き予測】
  w_i = 1 / (dist_cos(x, x_i) + ε)   ε=1e-9

  ŷ = argmax_{c∈{0,...,K−1}}  Σ_{i∈N_k(x)}  w_i · 𝟙[y_i = c]
```

**利点**: 学習不要。非線形境界を自然に表現できる。
**いつ使うか**: データが少なくベースラインを素早く作りたいとき。

---

### SoftmaxRegression（多項ロジスティック回帰）

**定義**:

```
パラメータ: W ∈ ℝ^{d×K}, b ∈ ℝ^K   （d=特徴次元、K=クラス数）

【スコア計算】
  z_c = w_c · x + b_c   c = 0,...,K−1

【Softmax（log-sum-exp で数値安定化）】
  z' = z − max_c z_c                   （オーバーフロー防止）
  P(y=c | x) = exp(z'_c) / Σ_{k=0}^{K−1} exp(z'_k)

【損失関数（交差エントロピー + L2 正則化）】
  L(W, b) = −(1/N) Σ_{i=1}^{N} log P(y=y_i | x_i)  +  (λ/2) ‖W‖_F²

  ‖W‖_F² = Σ_{j,c} W_{jc}²   （フロベニウスノルムの 2 乗）

【勾配（フルバッチ）】
  P_i ∈ ℝ^K : P_i[c] = P(y=c | x_i) （softmax 確率ベクトル）
  E_i ∈ ℝ^K : E_i = P_i − one_hot(y_i)

  ∂L/∂W = (1/N) Xᵀ E  +  λW
  ∂L/∂b = (1/N) Σ_i E_i

  （X ∈ ℝ^{N×d}, E ∈ ℝ^{N×K}）

【更新】
  W ← W − lr · ∂L/∂W
  b ← b − lr · ∂L/∂b

【早期停止】
  |L_t − L_{t-1}| < tol のとき停止
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `lr` | 0.5 | 学習率 |
| `reg` | 1e-4 | L2 正則化係数 λ |
| `epochs` | 800 | 最大エポック数 |
| `tol` | 1e-7 | 早期停止閾値 |

**利点**: 確率値を出力できる。解釈性が高い。**推奨**。

---

### OvRLinearSVM（One-vs-Rest 線形 SVM）

**定義**:

```
K クラスそれぞれについて 2 値 SVM を独立に学習する。

【2 値 SVM の設定（クラス c vs それ以外）】
  y_pm[i] = +1  if y_i = c,  else −1

【ヒンジ損失 + L2 正則化】
  L(w, b) = (1/N) Σ_{i=1}^{N} max(0, 1 − y_pm[i] · (w · x_i + b))  +  (λ/2) ‖w‖²

【Pegasos 風 SGD（平均化付き）】
  t: 累積ステップ数  （エポック × N から始まり 1 ずつ増加）

  学習率: η_t = 1 / (λ · t)

  margin_i = y_pm[i] · (w · x_i + b)

  更新ルール:
    if margin_i < 1:          （マージン不足）
      w ← (1 − η_t λ) w  +  η_t · y_pm[i] · x_i
      b ← b  +  η_t · y_pm[i]
    else:                     （十分なマージン）
      w ← (1 − η_t λ) w

  平均化:
    w_avg ← w_avg + w   （全ステップを累積）
    b_avg ← b_avg + b

  最終パラメータ: ŵ = w_avg / T,  b̂ = b_avg / T   （T=総ステップ数）

【多クラス予測】
  score_c(x) = ŵ_c · x + b̂_c

  ŷ = argmax_{c ∈ {0,...,K−1}} score_c(x)
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `reg` | 1e-3 | 正則化係数 λ |
| `epochs` | 50 | SGD エポック数 |

**利点**: 高次元疎ベクトルに強い。理論的な汎化保証がある。

---

### MLPClassifier（2 層 MLP）

**定義**:

```
パラメータ: W1 ∈ ℝ^{d×H}, b1 ∈ ℝ^H, W2 ∈ ℝ^{H×K}, b2 ∈ ℝ^K
           （d=入力次元、H=隠れ層サイズ、K=クラス数）

【He 初期化】
  W1 ~ N(0, √(2/d))     （ReLU の「半分がゼロ」を補正）
  W2 ~ N(0, √(2/H))

【順伝播】
  Z1 = X W1 + 1 b1ᵀ   ∈ ℝ^{N×H}
  H1 = max(Z1, 0)       ∈ ℝ^{N×H}   （ReLU 活性化関数）
  Z2 = H1 W2 + 1 b2ᵀ   ∈ ℝ^{N×K}
  P  = Softmax(Z2)      ∈ ℝ^{N×K}   （log-sum-exp 安定化）

【損失関数】
  L = −(1/N) Σ_i log P[i, y_i]  +  (λ/2)(‖W1‖_F² + ‖W2‖_F²)

【逆伝播】
  dZ2 = (P − one_hot(y)) / N         ∈ ℝ^{N×K}
  dW2 = H1ᵀ dZ2  +  λ W2
  db2 = sum_rows(dZ2)
  dH1 = dZ2 W2ᵀ                      ∈ ℝ^{N×H}
  dZ1 = dH1 ⊙ 𝟙[Z1 > 0]             ∈ ℝ^{N×H}   （ReLU の勾配）
  dW1 = Xᵀ dZ1  +  λ W1
  db1 = sum_rows(dZ1)

【Adam 更新（全パラメータ θ ∈ {W1,b1,W2,b2} に適用）】
  m_t = β₁ m_{t-1}  +  (1−β₁) g_t           （1次モーメント、β₁=0.9）
  v_t = β₂ v_{t-1}  +  (1−β₂) g_t²          （2次モーメント、β₂=0.999）
  m̂_t = m_t / (1 − β₁^t)                     （バイアス補正）
  v̂_t = v_t / (1 − β₂^t)
  θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)     （α=学習率、ε=1e-8）

【予測】
  ŷ = argmax_c P[c]
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `hidden` | 64 | 隠れ層のユニット数 H |
| `lr` | 1e-3 | Adam 学習率 α |
| `reg` | 1e-4 | L2 正則化係数 λ |
| `epochs` | 300 | 最大エポック数 |

**利点**: 非線形パターンを学習できる。データが多いほど精度が上がりやすい。

---

### GaussianNBClassifier（ガウシアンナイーブベイズ）

**定義**:

```
訓練データ: {(x_i, y_i)}   x_i ∈ ℝ^d（連続値特徴）、y_i ∈ {0,...,K−1}

【学習（各クラス c）】
  μ_{cj}  = (1/N_c) Σ_{y_i=c} x_{ij}           （特徴 j の平均）
  σ²_{cj} = (1/N_c) Σ_{y_i=c} (x_{ij} − μ_{cj})²  （特徴 j の分散）
  σ²_{cj} ← σ²_{cj} + var_smoothing × max_{c',j} σ²_{c'j}   （ゼロ分散防止）

【対数尤度 (アンダーフロー防止)】
  log P(x | y=c) = Σ_j log N(x_j ; μ_{cj}, σ²_{cj})
                 = −0.5 Σ_j [ log(2π σ²_{cj})  +  (x_j − μ_{cj})² / σ²_{cj} ]

【予測】
  ŷ = argmax_c [ log P(c)  +  log P(x | y=c) ]
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `var_smoothing` | 1e-9 | ゼロ分散回避のための平滑化係数 |

**利点**: 学習・推論が高速。連続値特徴に直接適用できる。
**欠点**: 特徴間の独立性仮定が強い。

---

### LDAClassifier（線形判別分析）

**定義**:

```
【学習】
  μ_c = クラス c の平均ベクトル ∈ ℝ^d

  クラス内散布行列:
    Σ_W = Σ_c Σ_{y_i=c} (x_i − μ_c)(x_i − μ_c)^T ∈ ℝ^{d×d}

  正則化付き逆行列 (ムーア・ペンローズ擬似逆行列):
    Σ_W^{-1} = pinv(Σ_W + reg · I)

【線形判別関数】
  δ_c(x) = x^T Σ_W^{-1} μ_c
            − 0.5 μ_c^T Σ_W^{-1} μ_c
            + log P(c)

  直感: 距離の公平な比較のためにクラス内分散で調整した空間への投影

【予測】
  ŷ = argmax_c δ_c(x)

  行列形式:
    W = Σ_W^{-1} M^T   (d×K)    M = [μ_0 ; μ_1 ; ...] ∈ ℝ^{K×d}
    bias_c = −0.5 μ_c^T W[:,c] + log P(c)
    scores = X W + bias  ∈ ℝ^{N×K}
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `reg` | 1e-4 | Σ_W の正則化係数 |

**利点**: 線形分類器の中で最も原理的。クラス間の距離構造を最大化する方向を学習。
**欠点**: クラス内共分散が等しいという仮定がある。

---

### VotingEnsembleClassifier（投票アンサンブル）

3 種の分類器（Softmax / MLP / LDA）の予測を**多数決 (Hard Voting)** で統合する。

**定義**:

```
構成分類器:
  clf_1 = SoftmaxRegression(lr=0.5, reg=1e-4, epochs=800)
  clf_2 = MLPClassifier(hidden=64, lr=1e-3, epochs=300)
  clf_3 = LDAClassifier(reg=1e-4)

【学習】
  各分類器を同じ訓練データで独立に学習する。

【予測 (Hard Voting)】
  preds_k[i] = clf_k.predict(x_i)   (各分類器の予測クラス)

  for サンプル i:
    candidates = {preds_1[i], preds_2[i], preds_3[i]}
    ŷ[i] = most_common(candidates)

【3 分類器の役割分担】
  Softmax: 確率的分類。勾配降下で重みを学習。
  MLP    : 非線形分類。ReLU 隠れ層で複雑な境界を学習。
  LDA    : 統計的分類。クラス内分散を正規化して線形境界を学習。
  → 3 者が異なる誤りを犯すため多数決で相殺しやすい。

【同票（3 者がすべて異なる予測）の場合】
  numpy.unique が辞書順に返す最初のクラスが選ばれる（circle=0, square=1, triangle=2）。
```

**アンサンブルが有効な状況**:

```
  個別精度     多数決の結果
  clf_1: ○   clf_2: ×   clf_3: ○   → ○ (多数決で正解)
  clf_1: ×   clf_2: ○   clf_3: ○   → ○ (多数決で正解)
  clf_1: ×   clf_2: ×   clf_3: ×   → × (全員間違い → 多数決でも不正解)

分類器間の予測相関が低いほど多数決の効果が高い。
```

**利点**: 単一分類器の弱点を他の分類器が補い合う。過学習しにくい。
**欠点**: 訓練・推論時間が 3 倍程度になる。構成分類器が相関していると効果が薄い。

> **ファクトリ関数**: `get_classifier("softmax")`

---

## 評価指標（metrics.py）

**定義**:

```
混同行列: C ∈ ℤ^{K×K}   C[i,j] = 真ラベル i を j と予測した数

TP_c = C[c,c]
FP_c = Σ_{i≠c} C[i,c]   （他クラスを c と誤予測した数）
FN_c = Σ_{j≠c} C[c,j]   （c を他クラスと誤予測した数）

Precision_c = TP_c / (TP_c + FP_c + ε)
Recall_c    = TP_c / (TP_c + FN_c + ε)
F1_c        = 2 · Precision_c · Recall_c / (Precision_c + Recall_c + ε)

Macro-Precision = (1/K) Σ_c Precision_c
Macro-F1        = (1/K) Σ_c F1_c

Accuracy        = Σ_c C[c,c] / Σ_{i,j} C[i,j]
```

```
classification_report の出力例:

class        prec  recall      f1  support
------------------------------------------
circle      0.950   0.942   0.946       53
square      0.934   0.941   0.937       51
triangle    0.958   0.960   0.959       56
------------------------------------------
macro       0.947   0.948   0.947      160
accuracy = 0.944

Confusion Matrix (row=true, col=pred):
          circle   square triangle
circle        50        2        1
square         1       48        2
triangle       1        1       54
```

---

## 使い方

### 基本

```python
from data       import make_shapes, train_test_split
from preprocess import get_preprocessor
from feature    import get_feature
from classifier import get_classifier
from pipeline   import ObjectRecognitionPipeline
from metrics    import classification_report

ds = make_shapes(n_per_class=200, size=24, noise=0.08, seed=0)
train, test = train_test_split(ds, ratio=0.8, seed=1)

pipe = ObjectRecognitionPipeline(
    preprocessor = get_preprocessor("standard"),
    feature      = get_feature("edge"),
    classifier   = get_classifier("softmax"),
)
pipe.fit(train.images, train.labels)
preds = pipe.predict(test.images)
print(classification_report(test.labels, preds, train.label_names))
```

### 全 140 通り比較（4前処理 × 5特徴 × 7分類器）

```bash
python main.py
```

---

## 推奨構成

```
standard  +  hog  +  ensemble   → データ拡張あり・最高精度クラス（main.py の自動選択）

理由:
  standard  : クラス間で公平な統計的正規化、収束が安定
  hog       : 勾配方向ヒストグラムで図形の形状・輪郭を精度よく記述
  ensemble  : Softmax + MLP + LDA の多数決で誤りを相殺

シンプルな一本構成:
  standard  +  edge  +  softmax  → 軽量・高速・高精度のベースライン
  standard  +  hog   +  lda     → 線形判別分析で HOG 特徴を効率よく分類
```

---

## 精度向上のための設計選択

| 箇所 | 工夫 | 理由 |
|------|------|------|
| HOG | L2-Hys 正規化（クリップ→再正規化）| 局所コントラスト変化に頑健 |
| Softmax | log-sum-exp + L2 正則化 + 早期停止 | オーバーフロー回避・過学習抑制 |
| SVM | Pegasos 平均化 SGD + η=1/(λt) 減衰 | 平均化で汎化性能向上 |
| MLP | He 初期化 + Adam + 早期停止 | ReLU 層と相性よく収束が安定 |
| kNN | コサイン距離 + 距離重み付き投票 | 高次元特徴では L2 距離より精度が高い |
| GaussianNB | var_smoothing でゼロ分散を防止 | 対数計算でアンダーフロー回避 |
| LDA | pinv による正則化 + 事前確率を bias に反映 | 特異行列でも安定して解ける |
| DCT | ジグザグスキャンで低周波を優先抽出 | 人間の視覚特性に対応した係数順 |
| 全体 | フルバッチ最適化 | ミニバッチより収束が滑らかで精度重視 |

---

## 依存ライブラリ

```
numpy のみ（外部ライブラリ不要）
Python 3.x
```
