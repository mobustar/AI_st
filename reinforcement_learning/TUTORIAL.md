# 強化学習 完全チュートリアル

> **このディレクトリで学べること**  
> 「試行錯誤しながら報酬を最大化する」エージェントの仕組みを数式から理解する。  
> 環境 → 方策 → アルゴリズム の 3 層構造がどう噛み合うかを体験する。

---

## 強化学習とは何か

機械学習の三種類:
```
教師あり学習: 正解ラベルがある → 予測誤差を最小化
教師なし学習: ラベルなし → 構造を発見
強化学習:     試行の結果に報酬 → 累積報酬を最大化
```

### マルコフ決定過程 (MDP) — 強化学習の数学的な土台

強化学習の問題は **MDP (Markov Decision Process)** として定式化される:

$$\langle \mathcal{S},\ \mathcal{A},\ \mathcal{P},\ \mathcal{R},\ \gamma \rangle$$

| 記号 | 意味 | このコードでの例 |
|------|------|----------------|
| $\mathcal{S}$ | 状態集合 | 格子の位置 (0〜15) |
| $\mathcal{A}$ | 行動集合 | 上・右・下・左 |
| $\mathcal{P}(s'\|s,a)$ | 遷移確率 | 決定的=1 / 確率的=スリップ |
| $\mathcal{R}(s,a)$ | 即時報酬 | ゴール=+1 / 普通=-0.04 |
| $\gamma \in [0,1)$ | 割引率 | 0.95 |

**マルコフ性**: 次の状態 $s'$ は**現在の状態 $s$ と行動 $a$ だけ**で決まる (過去に依存しない)。

### 目標: 累積割引報酬の最大化

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

$\gamma < 1$ の理由: 将来の報酬を少し割り引く (不確実性 + 早く解決したい)。

---

## 全体マップ

```
[env.py]      環境 (状態・報酬・遷移)
    ↕ step(action) → (next_state, reward, done)

[policy.py]   方策 (行動選択)
    ↕ select(Q, state) → action

[algorithm.py] 学習アルゴリズム (Q テーブルの更新)
    → train(env, policy, n_episodes) → rewards リスト

[trainer.py]  複数 seed で実行 → 平均評価
```

---

## Step 1 — 価値関数とベルマン方程式

### 行動価値関数 Q(s,a)

「状態 $s$ で行動 $a$ を取った後、最適に行動し続けたときの期待累積報酬」:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t=s, a_t=a\right]$$

### ベルマン最適方程式

最適 Q 関数 $Q^*$ は次の再帰式を満たす:

$$Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

この式を反復的に解くのが Q-learning の本質。

### Q テーブルの形

```python
Q: shape = (n_states, n_actions)

例 (4x4 GridWorld, 4 行動):
     上    右    下    左
s=0  0.0  0.1  0.2  0.0
s=1  0.0  0.3  0.1  0.1
...
s=15 0.0  0.0  0.0  0.0  ← ゴール
```

---

## Step 2 — 環境  `env.py`

### 2-1. GridWorld

```
(0,0)→→→(0,3)
 ↓         ↓
(3,0)→→→(3,3)★ゴール
```

- 各ステップ: -0.04 の報酬 (短い経路を促す)
- ゴール到達: +1.0

状態エンコード: `state = row * W + col` → 0〜15 の整数

### 2-2. CliffWalk (Sutton & Barto の古典問題)

```
S _ _ _ _ _ _ _ _ _ _ G   ← row=3
  崖 崖 崖 崖 崖 崖 崖 崖 崖 崖
```

崖に落ちると: 報酬 -100, スタートに戻る / 通常移動: -1  
→ **SARSA vs Q-learning の違いを最も明確に示す環境**

| アルゴリズム | 学習する経路 | 理由 |
|------------|------------|------|
| SARSA (on-policy) | 安全な上回り (-17 程度) | 自分の ε-greedy 方策で稼働するとき崖落ちリスクを考慮 |
| Q-learning (off-policy) | 崖際の最短経路 (-13) | max で最適方策を仮定するため崖のリスクを過小評価 |

### 2-3. StochasticGrid (スリップあり)

行動の 20% が意図と異なる隣接行動に変わる:

```python
if rng.random() < slip_prob:
    action = (action + rng.choice([-1, 1])) % 4
```

→ 決定的な Q-learning の最大化バイアスが顕在化するため、Double Q-learning との比較に最適。

### 2-4. WindyGridWorld

各列に「風」があり、行動のたびに上方向に押される:

```
風強さ: 0 0 0 1 1 1 2 2 1 0  (列ごと)
```

列 6〜7 では 2 マス上に押されるため、真っすぐ進めない。  
長期的な credit assignment が必要 → SARSA(λ) が有利。

---

## Step 3 — 行動選択方策  `policy.py`

### 探索 vs 活用のトレードオフ

- **活用 (exploitation)**: 現在の Q が高い行動を選ぶ → 安全だが局所解に陥る
- **探索 (exploration)**: ランダムに行動する → 良い経路を発見できるが非効率

### 3-1. ε-greedy

$$\pi(s) = \begin{cases} \text{ランダム} & \text{確率 } \varepsilon \\ \arg\max_a Q(s,a) & \text{確率 } 1-\varepsilon \end{cases}$$

シンプルだが $\varepsilon$ の設定が重要。

### 3-2. 減衰 ε-greedy (推奨)

学習初期は探索 ($\varepsilon$ が大きい)、後期は活用 ($\varepsilon$ が小さい):

$$\varepsilon_{t+1} = \max(\varepsilon_t \cdot \text{decay},\ \varepsilon_{\min})$$

`decay=0.995` なら 1000 エピソードで $\varepsilon$ は $1.0 \times 0.995^{1000} \approx 0.007$ になる。

### 3-3. Boltzmann (Softmax)

$$P(a \mid s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)}$$

> $\tau$ は**温度パラメータ**  
> $\tau \to \infty$: 一様ランダム (探索重視)  
> $\tau \to 0$: 決定的 greedy (活用重視)

ε-greedy より「Q 値の差に比例した探索確率」になるため、情報をより活用した探索ができる。

### 3-4. UCB1 (Upper Confidence Bound)

$$a^* = \arg\max_a \left[Q(s,a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s,a)}}\right]$$

> $N(s)$: 状態 $s$ の訪問回数  $N(s,a)$: $(s,a)$ の訪問回数  
> $c$: 探索の強さ (通常 $\sqrt{2}$ か $1.4$)

**直感**: まだあまり試していない行動 ($N(s,a)$ が小さい) にボーナスを与える。  
確率的バンディット問題で最適な探索保証がある原理的な方法。

---

## Step 4 — 学習アルゴリズム  `algorithm.py`

### 4-1. モンテカルロ法 (MonteCarlo)

エピソード全体を経験してから Q を更新する。

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-t-1} r_T$$

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[G_t - Q(s_t, a_t)\right]$$

**特徴**:
- バイアスなし (実際の報酬を使う)
- 分散が大きい (エピソード全体の確率的変動が入る)
- 途中での更新がないため、長いエピソードでは収束が遅い

```python
# algorithm.py の MonteCarlo.train() より
G = 0.0
for s, a, r in reversed(episode):
    G = r + self.gamma * G          # 後ろから累積報酬を計算
    self.Q[s, a] += self.alpha * (G - self.Q[s, a])
```

---

### 4-2. SARSA (On-policy TD)

1 ステップごとに Q を更新。**実際に取る次の行動 $a'$** を使う。

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[\underbrace{r + \gamma Q(s', a')}_{\text{TD ターゲット}} - Q(s,a)\right]$$

> $(s, a, r, s', a')$ → SARSA という名前の由来

**"on-policy"** の意味: 自分が実際に使う方策 $\pi$ で $a'$ を選ぶ。  
→ 崖際の「実際のリスク」を学習に反映できる。

TD 誤差 (temporal difference error):

$$\delta = r + \gamma Q(s', a') - Q(s, a)$$

$\delta > 0$: 実際より良かった → Q を上げる  
$\delta < 0$: 実際より悪かった → Q を下げる

---

### 4-3. Q-Learning (Off-policy TD) ★最重要

SARSA の $a'$ を **max** に変える:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\right]$$

**"off-policy"** の意味: 実際の方策に関係なく、**最適行動 (greedy)** を仮定して更新。  
→ 現在の方策が悪くても、理論上の最適 Q に収束できる。

| | SARSA | Q-learning |
|--|-------|-----------|
| 更新に使う $a'$ | 実際に選ぶ行動 | argmax の行動 |
| 崖際での振る舞い | 安全な迂回を学ぶ | 最短経路を学ぶ |
| 収束先 | $Q^{\pi}$ (現方策の価値) | $Q^*$ (最適価値) |

---

### 4-4. Expected SARSA

SARSA の $a'$ を**方策の期待値**に置き換える:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s,a)\right]$$

ε-greedy 方策の場合、期待値の計算:

$$\mathbb{E}[Q(s', \cdot)] = \frac{\varepsilon}{|\mathcal{A}|} \sum_{a'} Q(s', a') + (1-\varepsilon) \max_{a'} Q(s', a')$$

**SARSA より分散が低い** → より安定した学習が可能。  
Q-learning と SARSA の中間的な性質を持つ。

---

### 4-5. Double Q-Learning ← 確率的環境で最強

Q-learning の**最大化バイアス**問題を解決する:

#### 最大化バイアスとは

$$\mathbb{E}\left[\max_a Q(s', a)\right] \geq \max_a \mathbb{E}[Q(s', a)]$$

Q 値には推定誤差がある → max を取ると誤差の大きい方向に引っ張られる → 過大評価。

#### 解決策: 2 つの Q テーブルを使う

```
Q_A で「どの行動が良いか」を選択
Q_B で「その行動の価値」を評価
(50% の確率で A/B を入れ替え)
```

$$Q_A(s,a) \leftarrow Q_A(s,a) + \alpha \left[r + \gamma Q_B(s',\ \arg\max_{a'} Q_A(s', a')) - Q_A(s,a)\right]$$

**直感**: 「選択」と「評価」を別のテーブルに任せることで、互いのバイアスを打ち消す。

---

### 4-6. Dyna-Q (モデルベース + モデルフリー)

```
実環境からのサンプル ──→ Q を直接更新 (モデルフリー)
                   └──→ モデルに保存
モデル (過去の経験) ──→ Q を追加更新 (プランニング) ×n_planning
```

$$\text{実サンプル 1 回} + \text{モデルからのシミュレーション n\_planning 回}$$

**サンプル効率が飛躍的に向上**: 実環境とのインタラクションが少なくても学習が進む。

```python
# algorithm.py の DynaQ.train() より
# 1) 実環境で Q を更新
target = r + gamma * Q[ns].max()
Q[s, a] += alpha * (target - Q[s, a])

# 2) モデルを保存
model[(s, a)] = (r, ns, done)

# 3) モデルから n_planning 回プランニング
for _ in range(n_planning):
    ps, pa = random_from_observed_pairs
    pr, pns, pdone = model[(ps, pa)]
    Q[ps, pa] += alpha * (pr + gamma * Q[pns].max() - Q[ps, pa])
```

---

### 4-7. n-step SARSA

モンテカルロ (無限ステップ) と SARSA (1 ステップ) の間を連続的に補間する:

$$G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n Q(s_{t+n}, a_{t+n})$$

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[G_{t:t+n} - Q(s_t, a_t)\right]$$

- $n=1$: SARSA と等価 (高バイアス / 低分散)
- $n=\infty$: Monte Carlo と等価 (低バイアス / 高分散)

適切な $n$ はタスク依存 (このコードでは $n=4$ がデフォルト)。

---

### 4-8. SARSA(λ) — 適格性トレース ★最も洗練された手法

**n-step の限界**: どの $n$ が最適か事前にわからない。  
**解決**: $\lambda$ で複数の $n$ を**指数加重平均**する。

#### 適格性トレース $e(s,a)$

各 $(s,a)$ ペアへの「責任度」を記録するテーブル:

$$e(s_t, a_t) \leftarrow e(s_t, a_t) + 1 \quad \text{(訪問時に加算)}$$
$$e(s, a) \leftarrow \gamma\lambda \cdot e(s, a) \quad \text{(毎ステップ減衰)}$$

#### 更新

$$\delta = r + \gamma Q(s', a') - Q(s, a) \quad \text{(TD 誤差)}$$

$$Q(s, a) \leftarrow Q(s, a) + \alpha \delta \cdot e(s, a) \quad \text{全 (s,a) について同時に}$$

**直感**: 最近通ったルートほど $e$ が大きい → TD 誤差を最近の経路に強く反映 = 長距離の credit assignment。

```
ゴール ←(報酬)← s3 ← s2 ← s1 ← ...

λ=0: s3 だけ更新
λ=0.8: s3 に強く / s2 に中程度 / s1 に弱く ... 更新
λ=1.0: 全経路に均等に更新 (≈モンテカルロ)
```

---

## Step 5 — 実行してみる

### 全組み合わせ比較

```bash
cd reinforcement_learning
python main.py
```

```
=== Env: gridworld ===
policy      algo          mean     ± std
------------------------------------------
decay_eps   dyna_q        0.832     0.021
decay_eps   sarsa_lambda  0.821     0.018
...
```

### 1 つの構成を試す

```python
import numpy as np

# --- 4x4 GridWorld 環境 ---
class GridWorld:
    H, W, GOAL = 4, 4, 15
    ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]  # 上右下左

    def reset(self):
        self.state = 0
        return 0

    def step(self, a):
        r, c = divmod(self.state, self.W)
        dr, dc = self.ACTIONS[a]
        self.state = max(0,min(self.H-1,r+dr))*self.W + max(0,min(self.W-1,c+dc))
        done = (self.state == self.GOAL)
        return self.state, (1.0 if done else -0.04), done

# --- Q-learning (減衰 ε-greedy) ---
def train(n_episodes=400, alpha=0.1, gamma=0.95, eps_start=1.0, decay=0.995, seed=0):
    env = GridWorld()
    Q   = np.zeros((16, 4))
    rng = np.random.default_rng(seed)
    eps = eps_start
    rewards = []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        for _ in range(200):
            a = rng.integers(4) if rng.random() < eps else Q[s].argmax()
            ns, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[ns].max() - Q[s, a])
            s, total = ns, total + r
            if done:
                break
        rewards.append(total)
        eps = max(0.05, eps * decay)
    return Q, rewards

Q, rewards = train(n_episodes=400)
mean_r = np.mean(rewards[-100:])
std_r  = np.std(rewards[-100:])
print(f"平均報酬 (最後100エピソード): {mean_r:.3f} ± {std_r:.3f}")

# 最適方策を表示
DIR = ["↑", "→", "↓", "←"]
print("\n最適方策:")
for r in range(4):
    print(" ".join("G " if r*4+c == 15 else DIR[Q[r*4+c].argmax()]+" " for c in range(4)))
```

---

## Step 6 — 実験課題

### 課題 1: α (学習率) の影響

```python
# 上の GridWorld クラス定義に続けて実行
import numpy as np

def run_sarsa(alpha, n_episodes=400, gamma=0.95, seed=0):
    env, Q = GridWorld(), np.zeros((16, 4))
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        a = rng.integers(4) if rng.random() < 0.1 else Q[s].argmax()
        for _ in range(200):
            ns, r, done = env.step(a)
            na = rng.integers(4) if rng.random() < 0.1 else Q[ns].argmax()
            Q[s, a] += alpha * (r + gamma * Q[ns, na] - Q[s, a])
            s, a, total = ns, na, total + r
            if done:
                break
        rewards.append(total)
    return np.mean(rewards[-100:])

# alpha = 0.01, 0.1, 0.5 で比較
print("α (学習率) の比較:")
for alpha in [0.01, 0.1, 0.5]:
    print(f"  alpha={alpha:.2f} → 平均報酬={run_sarsa(alpha):.3f}")
```

- 小さい α: 収束が遅いが安定
- 大きい α: 収束が速いが振動する可能性

### 課題 2: γ (割引率) の影響

```python
# 上の GridWorld / train 関数に続けて実行
print("γ (割引率) の比較:")
for gamma in [0.5, 0.9, 0.99]:
    _, rewards = train(n_episodes=400, gamma=gamma)
    print(f"  gamma={gamma} → 平均報酬={np.mean(rewards[-100:]):.3f}")
```

- γ が小さい: 近い報酬を重視 → 近視眼的
- γ が大きい: 将来の報酬も重視 → 長期的な計画を立てる

### 課題 3: Dyna-Q の n_planning の影響

```python
# 上の GridWorld クラス定義に続けて実行
def run_dyna_q(n_planning, n_episodes=400, alpha=0.1, gamma=0.95, seed=0):
    env, Q, model, seen = GridWorld(), np.zeros((16, 4)), {}, []
    rng = np.random.default_rng(seed)
    eps, rewards = 1.0, []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        for _ in range(200):
            a = rng.integers(4) if rng.random() < eps else Q[s].argmax()
            ns, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[ns].max() - Q[s, a])
            model[(s, a)] = (r, ns)
            if (s, a) not in seen:
                seen.append((s, a))
            for _ in range(n_planning):
                ps, pa = seen[rng.integers(len(seen))]
                pr, pns = model[(ps, pa)]
                Q[ps, pa] += alpha * (pr + gamma * Q[pns].max() - Q[ps, pa])
            s, total = ns, total + r
            if done:
                break
        rewards.append(total)
        eps = max(0.05, eps * 0.995)
    return np.mean(rewards[-100:])

print("Dyna-Q n_planning の比較:")
for n in [0, 5, 20, 50]:
    print(f"  n_planning={n:2d} → 平均報酬={run_dyna_q(n):.3f}")
```

n_planning が大きいほどサンプル効率が良いが、計算時間が増える。

### 課題 4: SARSA vs Q-learning on CliffWalk

```python
import numpy as np

class CliffWalk:
    H, W, START, GOAL = 4, 12, 36, 47
    ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]

    def reset(self):
        self.state = self.START
        return self.state

    def step(self, a):
        r, c = divmod(self.state, self.W)
        dr, dc = self.ACTIONS[a]
        self.state = max(0,min(self.H-1,r+dr))*self.W + max(0,min(self.W-1,c+dc))
        if self.state // self.W == 3 and 1 <= self.state % self.W <= 10:
            return self.START, -100, False  # 崖
        done = (self.state == self.GOAL)
        return self.state, (0 if done else -1), done

def run_cliff(algo, n_episodes=500, alpha=0.1, gamma=0.95, eps=0.1, seed=0):
    env, Q = CliffWalk(), np.zeros((48, 4))
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        a = rng.integers(4) if rng.random() < eps else Q[s].argmax()
        for _ in range(500):
            ns, r, done = env.step(a)
            na = rng.integers(4) if rng.random() < eps else Q[ns].argmax()
            if algo == "sarsa":
                Q[s, a] += alpha * (r + gamma * Q[ns, na] - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * Q[ns].max() - Q[s, a])
            s, a, total = ns, na, total + r
            if done:
                break
        rewards.append(total)
    return np.mean(rewards[-100:])

print("CliffWalk: SARSA vs Q-learning")
print(f"  SARSA      → 平均報酬={run_cliff('sarsa'):.1f}  (安全な迂回路)")
print(f"  Q-learning → 平均報酬={run_cliff('q'):.1f}  (崖際の最短路)")
```

SARSA は安全経路 (-17 付近) を、Q は最短経路 (-13 付近) を学ぶはずです。

### 課題 5: Double Q の有効性確認

```python
import numpy as np

class StochasticGrid:
    H, W, GOAL = 4, 4, 15
    ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]

    def __init__(self, slip=0.2, seed=0):
        self.slip = slip
        self.rng  = np.random.default_rng(seed)

    def reset(self):
        self.state = 0
        return 0

    def step(self, a):
        if self.rng.random() < self.slip:
            a = (a + self.rng.choice([-1, 1])) % 4
        r, c = divmod(self.state, self.W)
        dr, dc = self.ACTIONS[a]
        self.state = max(0,min(self.H-1,r+dr))*4 + max(0,min(self.W-1,c+dc))
        done = (self.state == self.GOAL)
        return self.state, (1.0 if done else -0.04), done

def run_q_stochastic(n_episodes=500, alpha=0.1, gamma=0.95, seed=0):
    env, Q = StochasticGrid(seed=seed), np.zeros((16, 4))
    rng = np.random.default_rng(seed)
    eps, rewards = 1.0, []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        for _ in range(200):
            a = rng.integers(4) if rng.random() < eps else Q[s].argmax()
            ns, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[ns].max() - Q[s, a])
            s, total = ns, total + r
            if done:
                break
        rewards.append(total)
        eps = max(0.05, eps * 0.995)
    return np.mean(rewards[-100:])

def run_double_q(n_episodes=500, alpha=0.1, gamma=0.95, seed=0):
    env = StochasticGrid(seed=seed)
    QA, QB = np.zeros((16, 4)), np.zeros((16, 4))
    rng = np.random.default_rng(seed)
    eps, rewards = 1.0, []
    for _ in range(n_episodes):
        s, total = env.reset(), 0.0
        for _ in range(200):
            a = rng.integers(4) if rng.random() < eps else (QA+QB)[s].argmax()
            ns, r, done = env.step(a)
            if rng.random() < 0.5:
                QA[s, a] += alpha * (r + gamma * QB[ns, QA[ns].argmax()] - QA[s, a])
            else:
                QB[s, a] += alpha * (r + gamma * QA[ns, QB[ns].argmax()] - QB[s, a])
            s, total = ns, total + r
            if done:
                break
        rewards.append(total)
        eps = max(0.05, eps * 0.995)
    return np.mean(rewards[-100:])

print("確率的環境: Q-learning vs Double Q-learning")
print(f"  Q-learning   → 平均報酬={run_q_stochastic():.3f}")
print(f"  Double Q     → 平均報酬={run_double_q():.3f}")
```

---

## Step 7 — Q テーブルアンサンブル  `trainer.py`

### なぜアンサンブルが有効か

Q-learning などは乱数 seed（初期値・探索順）に依存する。  
同じアルゴリズムでも seed が変わると最終的な Q テーブルが異なることがある。

```
seed=0 → Q_0   (ある局所解 or 最適解)
seed=1 → Q_1   (別の局所解 or 最適解)
...
seed=6 → Q_6

Q_ensemble = (Q_0 + Q_1 + ... + Q_6) / 7
```

個々の Q テーブルの「偏り」が打ち消しあい、より安定した方策が得られる。

### 評価方法

```
単体評価:
  n_seeds=5 の独立学習 → 各 seed の評価報酬の mean ± std

アンサンブル評価:
  n_agents=7 の Q テーブルを平均 → 5 パターンの評価環境で評価
  → mean, std を単体と比較

アンサンブルの mean が単体の mean より高い → Q 平均化の効果あり
アンサンブルの std が単体の std より低い  → より安定した方策
```

### 実際に確認する

```bash
python main.py
```

フェーズ 2 の出力例:
```
gridworld  (4×4格子 スタート(0,0)→ゴール(3,3))
  方策: decay_eps    学習: dyna_q
  単体 (800ep×5seed) : mean=  0.821 ± 0.034
  アンサンブル (7体平均): mean=  0.843 ± 0.019  ↑改善
```

→ アンサンブルで mean 上昇・std 減少 = 正しく機能している。

---

## まとめ: アルゴリズム選択ガイド

| 状況 | 推奨アルゴリズム | 理由 |
|------|----------------|------|
| 決定的な小環境 | Q-learning | 最適価値に直接収束 |
| 崖・罰則がある環境 | SARSA | 方策のリスクを考慮 |
| 確率的な環境 | Double Q-learning | 最大化バイアスを除去 |
| サンプルが少ない | Dyna-Q | モデルで仮想経験を増幅 |
| 長いエピソード | SARSA(λ) | 長距離の credit assignment |
| 最初の実験 | Q-learning + 減衰ε-greedy | シンプルかつ理論的に正しい |

### 数式まとめ

| アルゴリズム | TD ターゲット |
|------------|-------------|
| Monte Carlo | $G_t$ (実際の累積報酬) |
| SARSA | $r + \gamma Q(s', a')$ ($a'$ は実際の行動) |
| Q-learning | $r + \gamma \max_{a'} Q(s', a')$ |
| Expected SARSA | $r + \gamma \mathbb{E}_\pi[Q(s', \cdot)]$ |
| Double Q | $r + \gamma Q_B(s', \arg\max_{a'} Q_A(s', a'))$ |
| Dyna-Q | Q-learning + モデルからのプランニング |
| n-step SARSA | $\sum_{k=0}^{n-1}\gamma^k r_{t+k+1} + \gamma^n Q(s_{t+n}, a_{t+n})$ |
| SARSA(λ) | $\delta \cdot e(s,a)$ を全ペアに一括伝播 |
