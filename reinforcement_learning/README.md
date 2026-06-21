# reinforcement_learning

NumPy **のみ**で実装した Q テーブルベース強化学習のフルセット。
4 種類の環境に対し、方策・アルゴリズムの組み合わせを比較できる。

---

## 強化学習の基本サイクル

```
         ┌─────────────────────────────────┐
         │            Agent                │
         │  ┌──────────┐  ┌────────────┐  │
         │  │  Policy  │  │ Algorithm  │  │
         │  │ (行動選択)│  │(Q テーブル)│  │
         │  └──────────┘  └────────────┘  │
         └───────┬──────────────┬──────────┘
                 │ action a      │ update Q(s,a)
                 ▼              ▲
         ┌───────────────┐      │
         │      Env      │──────┘
         │  (状態・報酬)  │  reward r, next_state s'
         └───────────────┘
```

| コンポーネント | 役割 | 選択肢 |
|--------------|------|--------|
| **Env** | 状態・行動・報酬を定義する環境 | 4 種 |
| **Policy** | Q 値から行動を選ぶ探索戦略 | 4 種 |
| **Algorithm** | Q テーブルの更新ルール | 8 種 |

4 × 4 × 8 = **128 通り**を 3 seed 平均で比較できる。

---

## ディレクトリ構成

```
reinforcement_learning/
├── env.py         # 環境 4 種
├── policy.py      # 行動選択ポリシー 4 種
├── algorithm.py   # 学習アルゴリズム 8 種
├── trainer.py     # 複数 seed 評価トレーナー
└── main.py        # 全 128 通り比較 + 推奨構成の詳細評価
```

---

## 共通記号

```
S : 状態空間（有限離散）
A : 行動空間   |A| = n_actions = 4
s ∈ S : 現在の状態
a ∈ A : 選択した行動
r ∈ ℝ : 報酬
s'    : 遷移後の状態
γ ∈ [0,1) : 割引率（デフォルト 0.95）
α ∈ (0,1] : 学習率（デフォルト 0.1）
Q : S×A → ℝ : 行動価値テーブル（初期値 0）
```

---

## Env（env.py）— 4 種

すべて OpenAI Gym 風のインタフェース: `reset()` → `step(action)` → `(next_state, reward, done)`
行動は 4 方向: `0=上, 1=右, 2=下, 3=左`。状態は `row × 幅 + col` の整数で表現。

---

### GridWorld（4×4 格子）

```
┌───┬───┬───┬───┐
│ S │   │   │   │   S = スタート (0,0)
├───┼───┼───┼───┤   G = ゴール  (3,3)
│   │   │   │   │
├───┼───┼───┼───┤   報酬: ゴール到達 = +1.0
│   │   │   │   │         各ステップ  = −0.04
├───┼───┼───┼───┤         100 ステップ超過 → 強制終了
│   │   │   │ G │
└───┴───┴───┴───┘
```

**定義**:

```
状態: s = row × 4 + col   s ∈ {0,...,15}
遷移: 行動 a = (Δrow, Δcol) に従って移動
      壁に当たると静止（clip で実装）

報酬関数:
  R(s, a, s') = +1.0   if s' = ゴール(15)
              = −0.04  otherwise

終了条件:
  s' = ゴール  または  ステップ数 ≥ 100
```

---

### CliffWalk（4×12 格子）

Sutton & Barto の古典的問題。SARSA と Q-learning の挙動の違いを観察できる。

```
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  │  │  row 0
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  row 1
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  row 2
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│ S│XX│XX│XX│XX│XX│XX│XX│XX│XX│XX│ G│  row 3
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Q-learning の最適経路（崖ギリギリ）: 報酬 ≈ −13
  S→→→→→→→→→→→G

SARSA の学習経路（安全な迂回）: 報酬 ≈ −15〜−17
  S↑→→→→→→→→→↓G
```

**定義**:

```
状態: s = row × 12 + col   s ∈ {0,...,47}
崖:   Cliff = {(3, col) : 1 ≤ col ≤ 10}

報酬関数:
  R(s, a, s') = −1.0    if s' = ゴール(3,11)   かつ 終了
              = −100.0  if s' ∈ Cliff  →  スタート(3,0) にリセット
              = −1.0    otherwise

終了条件:
  s' = ゴール  または  ステップ数 ≥ 200
```

---

### StochasticGrid（4×4 滑り付き格子）

```
┌───┬───┬───┬───┐
│ S │   │   │   │   X = 穴 {(1,1),(2,3)}: 報酬 −1、終了
├───┼───┼───┼───┤   G = ゴール (3,3)   : 報酬 +1、終了
│   │ X │   │   │   slip_prob = 0.2
├───┼───┼───┼───┤   → 20% の確率で隣接行動にランダムにずれる
│   │   │   │ X │
├───┼───┼───┼───┤
│   │   │   │ G │
└───┴───┴───┴───┘
```

**定義**:

```
確率的遷移:
  u ~ Uniform(0, 1)
  if u < slip_prob:
    a' = (a + choice({−1, +1})) mod 4   （左右どちらかにずれる）
  else:
    a' = a

報酬関数:
  R(s, a, s') = +1.0  if s' = ゴール
              = −1.0  if s' ∈ Holes
              =  0.0  otherwise

終了条件:
  s' = ゴール  または  s' ∈ Holes  または  ステップ数 ≥ 100
```

---

### WindyGridWorld（7×10 風あり格子）

Sutton & Barto の Example 6.5。列ごとに「風」があり、上方向に押される。

```
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │↑ │↑ │↑↑│↑↑│↑ │  │  row 0..2 (矢印は風の影響)
│ S│  │  │  │  │  │  │ G│  │  │  row 3  S=(3,0), G=(3,7)
│  │  │  │  │  │  │  │  │  │  │  row 4..6
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
風(列): 0  0  0  1  1  1  2  2  1  0
```

**定義**:

```
確定的遷移 + 風の効果:
  ny = clip(row + Δrow_action − WIND[col], 0, H−1)
  nx = clip(col + Δcol_action,            0, W−1)

報酬関数:
  R(s, a, s') = −1.0   （全ステップ一定）

終了条件:
  s' = ゴール(3,7)  または  ステップ数 ≥ 500
```

**特徴**: 風があると最短経路が zigzag になる。SARSA(λ) が credit assignment を効率よく伝播させる。

| 環境名 | クラス | 状態数 | 行動数 | 主な用途 |
|--------|--------|:----:|:----:|----------|
| `gridworld` | `GridWorld` | 16 | 4 | アルゴリズムの基本動作確認 |
| `cliffwalk` | `CliffWalk` | 48 | 4 | SARSA vs Q-learning の比較 |
| `stochastic` | `StochasticGrid` | 16 | 4 | Double Q の最大化バイアス除去の検証 |
| `windy` | `WindyGridWorld` | 70 | 4 | n-step / eligibility traces の効果検証 |

> **ファクトリ関数**: `get_env("gridworld")`

---

## Policy（policy.py）— 4 種

Q 値から具体的な行動を選ぶ方策。**探索（Exploration）** と **活用（Exploitation）** のバランスを担う。

```
← 探索重視 ────────────────────── 活用重視 →
  完全ランダム   EpsilonGreedy   argmax のみ
  ε=1.0          ε=0.1           ε=0.0
```

---

### EpsilonGreedy（ε-greedy）

**定義**:

```
u ~ Uniform(0, 1)

π(s) = { 一様ランダム a ~ Uniform(A)   if u < ε
        { argmax_{a'} Q(s, a')          otherwise

同点処理: max Q(s,·) を達成する行動が複数ある場合、
          その中からランダムに 1 つを選ぶ（偏り防止）
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `epsilon` | 0.1 | 探索確率 ε |

**いつ使うか**: 比較用ベースライン。シンプルな決定的環境。

---

### DecayEpsilonGreedy（減衰 ε-greedy）

**定義**:

```
EpsilonGreedy と同じ行動選択式を使い、
エピソード終了時に ε を減衰させる。

エピソード終了時の更新:
  ε_{t+1} = max(ε_t × decay, ε_min)

初期値 ε_0 から始まり、最小値 ε_min に漸近する。

t エピソード後の ε の値:
  ε_t = max(ε_0 × decay^t, ε_min)
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `epsilon` | 1.0 | 初期探索確率 ε_0 |
| `decay` | 0.995 | 幾何減衰係数 |
| `epsilon_min` | 0.01 | 下限 ε_min |

**いつ使うか**: **全環境で推奨**。探索→活用の自然な移行。

---

### Boltzmann（ソフトマックス探索）

**定義**:

```
Q(s, a) / τ を各行動のスコアとして softmax 確率を計算する。

【log-sum-exp による数値安定化】
  q = Q(s, ·) / τ                    （スコアベクトル）
  q' = q − max_{a'} q(a')            （最大値を引いてオーバーフロー防止）

【確率分布】
  P(a | s) = exp(q'(a)) / Σ_{a'} exp(q'(a'))

【行動選択】
  a ~ Categorical(P(· | s))          （確率分布からサンプリング）

τ → ∞ : P(a|s) → 1/|A|             （一様分布 = 完全探索）
τ → 0  : P(a*|s) → 1               （argmax Q = 完全活用）
          （a* = argmax Q(s,·)）
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `tau` | 0.5 | 温度 τ（大きいほど探索的）|

**いつ使うか**: Q 値の差を確率に反映させた滑らかな探索。

---

### UCB1（上限信頼区間）

**定義**:

```
N(s)   : 状態 s を訪問した総回数
N(s,a) : 状態 s で行動 a を選んだ回数（初期値 0）

【未試行の行動がある場合】
  untried = {a ∈ A : N(s, a) = 0}
  if untried ≠ ∅:
    a ~ Uniform(untried)               （未試行を優先して選ぶ）

【全行動試行済みの場合】
  UCB(s, a) = Q(s, a)  +  c · √( ln N(s) / N(s, a) )

  a* = argmax_{a ∈ A} UCB(s, a)

  同点処理: UCB 最大の行動が複数あればランダムに 1 つ

【訪問回数の更新】
  N(s, a*) ← N(s, a*) + 1
  N(s)     ← N(s) + 1
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `c` | 1.4 | 探索ボーナスの重み（√2 ≈ 1.41 が理論的推奨）|

**いつ使うか**: 全 (s,a) を偏りなく試行したい確率的環境。

> **ファクトリ関数**: `get_policy("decay_eps", epsilon=1.0, decay=0.995, epsilon_min=0.05)`

---

## Algorithm（algorithm.py）— 8 種

Q テーブルの更新ルール。すべてに共通する TD 更新式:

```
Q(s, a) ← Q(s, a)  +  α · [target − Q(s, a)]
                              ↑ TD 誤差（δ）

target の計算方法がアルゴリズムごとに異なる。
```

---

### MonteCarlo（モンテカルロ法）

**定義**:

```
エピソード τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T) を収集する。

【収益（リターン）の計算】
  G_T     = 0
  G_{t}   = r_t  +  γ · G_{t+1}     t = T−1, T−2, ..., 0

  （後ろから逐次計算: G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...）

【Every-Visit 更新】
  各時刻 t について（同じ (s,a) が複数回出現しても全て更新）:
    Q(s_t, a_t) ← Q(s_t, a_t)  +  α · [G_t − Q(s_t, a_t)]
```

**利点**: TD のブートストラップバイアスがない（実際の報酬のみ使用）。
**欠点**: エピソードが終わるまで更新できない。分散が大きい。

---

### SARSA（On-policy TD(0)）

**定義**:

```
On-policy: 学習に使う方策 π と評価する方策が同じ

【1 ステップの更新】
  状態 s で方策 π により行動 a を選ぶ
  環境を 1 ステップ実行: (s, a) → (r, s')
  次状態 s' でも方策 π により次の行動 a' を先に選ぶ

  target = r  +  γ · Q(s', a') · 𝟙[not done]

  Q(s, a) ← Q(s, a)  +  α · [target − Q(s, a)]

  (s, a) ← (s', a')   （タプルを引き継いで次ステップへ）

名前の由来: State, Action, Reward, (next)State, (next)Action
```

**利点**: 実際に実行する方策（探索含む）の価値を正確に学べる。安全な経路を選ぶ傾向。
**いつ使うか**: 失敗コストが高く安全性を重視する場面。

---

### QLearning（Off-policy TD(0)）

**定義**:

```
Off-policy: 行動選択は方策 π（ε-greedy 等）で行うが、
            更新には別の（最適）方策の価値を使う

【1 ステップの更新】
  状態 s で方策 π により行動 a を選ぶ
  環境を 1 ステップ実行: (s, a) → (r, s')

  target = r  +  γ · max_{a'} Q(s', a') · 𝟙[not done]
                  ↑ 次状態での最善行動の Q 値（Greedy）

  Q(s, a) ← Q(s, a)  +  α · [target − Q(s, a)]

  s ← s'
```

**利点**: 探索方策に依存せず最適方策を学べる。CliffWalk で最高報酬を実現。
**いつ使うか**: 最終的な最高パフォーマンスを目指す場面。

---

### ExpectedSARSA（期待値 SARSA）

**定義**:

```
SARSA の次行動 a' のサンプリングを期待値に置き換える。

【ε-greedy 方策の期待 Q 値】
  max_actions = {a' : Q(s', a') = max_{a''} Q(s', a'')}
  n_max = |max_actions|

  π(a' | s') = ε / |A|                      （ランダム成分）
             + (1 − ε) / n_max              （greedy 成分、同点は均等配分）
               （ただし a' ∈ max_actions の場合のみ後項を加算）

  E_π[Q(s', ·)] = Σ_{a'} π(a' | s') · Q(s', a')

【更新】
  target = r  +  γ · E_π[Q(s', ·)] · 𝟙[not done]

  Q(s, a) ← Q(s, a)  +  α · [target − Q(s, a)]
```

**利点**: 1 つの a' をサンプルする SARSA より分散が小さく安定して収束する。

---

### DoubleQLearning（ダブル Q 学習）

**定義**:

```
Q_A, Q_B ∈ ℝ^{|S|×|A|} を独立に保持する（初期値 0）。

【行動選択】
  Q_mean(s, a) = (Q_A(s, a) + Q_B(s, a)) / 2
  a = Policy.select(Q_mean, s)

【更新（50% の確率でいずれかを選択）】
  u ~ Uniform(0, 1)

  if u < 0.5:      （Q_A を更新）
    a_best = argmax_{a'} Q_A(s', a')         （行動選択は Q_A）
    target = r  +  γ · Q_B(s', a_best) · 𝟙[not done]   （評価は Q_B）
    Q_A(s, a) ← Q_A(s, a)  +  α · [target − Q_A(s, a)]

  else:             （Q_B を更新）
    a_best = argmax_{a'} Q_B(s', a')
    target = r  +  γ · Q_A(s', a_best) · 𝟙[not done]
    Q_B(s, a) ← Q_B(s, a)  +  α · [target − Q_B(s, a)]

【最大化バイアスの除去】
  通常の Q-learning:
    E[max_a Q(s',a)] ≥ max_a E[Q(s',a)]     （ Jensen の不等式）
    → Q 推定にノイズがあると max 操作で過大評価が生じる

  Double Q:
    E[Q_B(s', argmax Q_A(s',·))] = max_a E[Q(s',a)]  （近似的に）
    → 行動選択と価値評価を分離することでバイアスが相殺される
```

**利点**: 確率的環境で Q-learning より安定した収束。最大化バイアスがない。

---

### DynaQ（モデルベース + モデルフリーの統合）

**定義**:

```
model : (S×A) → (ℝ × S × {True,False}) の決定的遷移モデル
observed_pairs : これまで観測した (s,a) ペアのリスト

【1 ステップごとの処理】

  ① 実環境での直接更新（Q-Learning と同じ）:
       target = r  +  γ · max_{a'} Q(s', a') · 𝟙[not done]
       Q(s, a) ← Q(s, a)  +  α · [target − Q(s, a)]

  ② モデルへの保存:
       model[(s, a)] ← (r, s', done)
       if (s, a) ∉ observed_pairs: observed_pairs.append((s, a))

  ③ プランニング（n_planning 回繰り返す）:
       (ps, pa) ~ Uniform(observed_pairs)
       (pr, ps', pdone) = model[(ps, pa)]
       ptarget = pr  +  γ · max_{a'} Q(ps', a') · 𝟙[not pdone]
       Q(ps, pa) ← Q(ps, pa)  +  α · [ptarget − Q(ps, pa)]

【サンプル効率の改善】
  実環境 1 ステップにつき n_planning 回の追加更新が行われるため、
  実エピソード数 T に対して実質 T × (1 + n_planning) 回の Q 更新が行われる。
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `n_planning` | 20 | 1 ステップあたりのプランニング回数 |

**利点**: 実サンプルが少ない状況でも高速に収束。
**欠点**: 決定的モデルのため確率的環境では最後の観測のみを記憶する。

---

### NStepSARSA（n-step SARSA）

**定義**:

```
n ステップ先まで実際の報酬を展開してから Q を更新する。

【n-step リターン】
  G_{t:t+n} = Σ_{k=0}^{n-1} γ^k r_{t+k+1}  +  γ^n Q(s_{t+n}, a_{t+n})

  ※ t+n ≥ T (終端) の場合: 該当する Q 項は省略し報酬だけを使う

【更新】
  Q(s_t, a_t) ← Q(s_t, a_t)  +  α [G_{t:t+n} − Q(s_t, a_t)]

  更新は t+n 時点で行われる (n ステップ遅れ)。

【n と他手法の関係】
  n=1     ≡ SARSA
  n=∞    ≡ Monte Carlo  (エピソード全体を使う)
  1<n<∞  : バイアス↓・分散↑ のトレードオフ
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `n` | 4 | ルックアヘッドのステップ数 |

**利点**: n を調整することで SARSA と MC の中間のバイアス-分散トレードオフを選べる。
**実装**: リングバッファ（サイズ n+1）で遷移を保持し、インデックスを `mod (n+1)` で管理。

---

### SARSALambda（SARSA(λ)）

**定義**:

```
適格性トレース e(s,a) ∈ ℝ^{|S|×|A|} （エピソード開始時にゼロ初期化）

【各ステップの処理】

  ① TD 誤差:
       δ = r  +  γ Q(s', a')  −  Q(s, a)

  ② トレース更新 (Accumulating traces):
       e(s, a) ← γλ e(s, a) + 1   （訪問した (s,a)）

  ③ 全 (s,a) の Q と e を一括更新:
       Q(s'', a'') ← Q(s'', a'')  +  α δ e(s'', a'')
       e(s'', a'') ← γλ e(s'', a'')

【λ の役割】
  λ=0 : e がすぐ消えるので SARSA(0) と等価
  λ=1 : e が長く残り、遠い過去まで credit assignment が届く（MC に近い）
  実用的には λ ∈ [0.7, 0.9] が高精度な傾向

【比較】
  n-step SARSA : 固定 n ステップの正確な展開
  SARSA(λ)     : λ^k の指数重みで全 n に対する加重平均 (forward view)
```

| パラメータ | デフォルト | 意味 |
|-----------|:-------:|------|
| `lam` | 0.8 | トレース減衰率 λ |

**利点**: 1 ステップごとに全 (s,a) を更新できるため、n-step SARSA より収束が早い傾向。
**いつ使うか**: 遷移経路が長い環境 (WindyGridWorld 等) での効率的な学習。

> **ファクトリ関数**: `get_algorithm("dyna_q", n_states, n_actions, n_planning=10)`

---

## Trainer（trainer.py）

複数 seed で学習・評価し、平均報酬と標準偏差を返す。さらにアンサンブル評価も提供する。

### train_and_evaluate（複数 seed 評価）

**定義**:

```
for seed in range(n_seeds):
  env  = env_factory(seed)
  algo = algo_factory(n_states, n_actions, seed)
  pol  = policy_factory(seed)
  algo.train(env, pol, n_episodes)   # 学習

  eval_env = env_factory(seed + 10000)   # 学習とは別 seed で評価
  eval_r   = evaluate_greedy(eval_env, algo, n_eval_episodes)
  eval_rewards.append(eval_r)

mean_eval = (1/n_seeds) Σ_k eval_rewards[k]
std_eval  = √( (1/n_seeds) Σ_k (eval_rewards[k] − mean_eval)² )

evaluate_greedy:
  貪欲方策 π_greedy(s) = argmax_a Q(s,a) で n_eval_episodes 回実行し平均を返す。
  訓練中の ε-greedy ノイズを含まない純粋な性能を測定する。

評価用に別 seed を使う理由:
  確率的環境 (stochastic) では同じ seed だと乱数が同じになり評価に偏りが出る。
  seed + 10000 で完全に異なる乱数列で評価する。
```

### ensemble_train_and_evaluate（Q テーブルアンサンブル）

複数エージェントを異なる seed で学習させ、Q テーブルを**平均**して評価する。

**定義**:

```
【学習フェーズ（n_agents 体を独立に訓練）】
  for seed in range(n_agents):
    env  = env_factory(seed)
    algo = algo_factory(n_states, n_actions, seed)
    pol  = policy_factory(seed)
    algo.train(env, pol, n_episodes)
    Qs.append(algo.Q.copy())   # shape: (n_states, n_actions)

【Q テーブルの平均】
  Q_ensemble = (1/n_agents) Σ_k Q_k   ∈ ℝ^{|S|×|A|}

  平均化の効果:
    各エージェントは異なる乱数 seed で初期化・探索するため、
    局所解に陥る方向が異なる。
    平均することで個々の偏りが打ち消しあい、
    より安定した Q 値の推定が得られる。

【評価フェーズ】
  averaged_agent.greedy_action(s) = argmax_a Q_ensemble(s, a)

  for seed in range(5):
    eval_env = env_factory(seed + 20000)
    eval_rewards.append(evaluate_greedy(eval_env, averaged_agent, n_eval_episodes))

  mean_eval = mean(eval_rewards)
  std_eval  = std(eval_rewards)

【単体 vs アンサンブルの比較】
  単体 (n_seeds=5, n_episodes=800):  複数回の独立学習の平均
  アンサンブル (n_agents=7, n_episodes=800): Q テーブルを平均した合成エージェント

  → アンサンブルの方が mean が高い or std が小さい場合、
    Q テーブル平均化の効果があったことを示す。
```

| 関数 | 引数 | 返り値 |
|------|------|-------|
| `train_and_evaluate` | `n_episodes`, `n_eval_episodes`, `n_seeds` | `(mean, std, train_curves)` |
| `ensemble_train_and_evaluate` | `n_episodes`, `n_eval_episodes`, `n_agents` | `(mean, std)` |

---

## 使い方

### 基本

```python
from env       import get_env
from policy    import get_policy
from algorithm import get_algorithm

env  = get_env("cliffwalk")
pol  = get_policy("decay_eps", epsilon=1.0, decay=0.995, epsilon_min=0.05)
algo = get_algorithm("q", env.n_states, env.n_actions, alpha=0.1, gamma=0.95)

rewards = algo.train(env, pol, n_episodes=500)

state = env.reset()
done  = False
total = 0
while not done:
    a = algo.greedy_action(state)
    state, r, done = env.step(a)
    total += r
print(f"合計報酬: {total}")
```

### 全 128 通り比較（4環境 × 4方策 × 8アルゴリズム）

```bash
python main.py
```

---

## 環境ごとの推奨構成

| 環境 | 推奨 Policy | 推奨 Algorithm | 理由 |
|------|------------|----------------|------|
| `gridworld` | `decay_eps` | `dyna_q` | プランニングで少ない実サンプルでも素早く収束 |
| `cliffwalk` | `decay_eps` | `q` | Off-policy で崖際の最短経路（最高報酬）を学習 |
| `stochastic` | `decay_eps` | `double_q` | 最大化バイアスを除去して確率的変動に対処 |
| `windy` | `decay_eps` | `sarsa_lambda` | 長い経路を λ トレースで効率よく credit assignment |

---

## 精度向上のための設計選択

| 箇所 | 工夫 | 理由 |
|------|------|------|
| EpsilonGreedy | 同点 argmax でランダム選択 | 特定行動への偏りを回避 |
| Boltzmann | log-sum-exp で安定化 | 大きな Q 値でのオーバーフロー防止 |
| UCB1 | 未試行行動を ∞ ボーナスで優先 | 全 (s,a) ペアを保証的に試行 |
| DoubleQ | QA/QB 交互更新 | 理論的に正しい不偏推定 |
| NStepSARSA | mod-(n+1) リングバッファ | メモリ効率よく任意 n ステップを保持 |
| SARSALambda | e 行列の一括スカラー更新 | NumPy のブロードキャストで O(|S||A|) を高速化 |
| Trainer | 複数 seed で平均 ± 標準偏差 | 統計的に信頼できる比較 |

---

## 依存ライブラリ

```
numpy のみ（外部ライブラリ不要）
Python 3.x
```
