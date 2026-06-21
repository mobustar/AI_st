"""
==============================================================
  algorithm.py | 強化学習アルゴリズム (戦略パターン)
==============================================================
Q テーブルベースの 8 つのアルゴリズム。すべて共通インタフェース:
    algo.train(env, policy, n_episodes) → rewards
    algo.greedy_action(state) → 学習後の貪欲方策

選択アルゴリズム:
  1. MonteCarlo       - エピソード終了後に G_t で更新 (バイアスなし・分散大)
  2. SARSA            - On-policy TD(0)。安全な経路を選ぶ傾向
  3. QLearning        - Off-policy TD(0)。最適方策 Q* に直接収束
  4. ExpectedSARSA    - 次状態の期待値で更新 (SARSA より分散が低く安定)
  5. DoubleQLearning  - 2 つの Q テーブルで最大化バイアスを除去
  6. DynaQ            - モデル学習 + プランニングでサンプル効率を向上
  7. NStepSARSA       - n ステップ先まで実報酬を展開 (MC と SARSA の中間)
  8. SARSALambda      - 適格性トレースで長距離の credit assignment を実現

精度の傾向:
  決定的:       Q-Learning ≈ SARSA ≈ ExpectedSARSA ≈ DynaQ
  確率的:       DoubleQLearning > ExpectedSARSA > Q-Learning
  少サンプル:   DynaQ が最良 (実環境 1 ステップで n_planning 回追加更新)
  長エピソード: SARSALambda が最良 (λ トレースで遠い過去まで更新)

数式まとめ (TD ターゲット):
  MonteCarlo:      G_t = Σ γ^k r_{t+k}           ※ エピソード全体
  SARSA:           r + γ Q(s', a')                ※ a' は実際にとる行動
  Q-learning:      r + γ max_a' Q(s', a')
  Expected SARSA:  r + γ Σ_a' π(a'|s') Q(s', a')
  Double Q:        r + γ Q_B(s', argmax_a' Q_A(s', a'))
  n-step SARSA:    Σ_{k=0}^{n-1} γ^k r_{t+k+1} + γ^n Q(s_{t+n}, a_{t+n})
  SARSA(λ):        Q(s,a) += α δ e(s,a)  ※ δ=TD誤差、e=適格性トレース
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List
import numpy as np

from env    import BaseEnv
from policy import BasePolicy


class BaseAlgorithm(ABC):
    """共通基底"""

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.95):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.Q = np.zeros((n_states, n_actions))

    @abstractmethod
    def train(self, env: BaseEnv, policy: BasePolicy, n_episodes: int) -> List[float]:
        ...

    def greedy_action(self, state: int) -> int:
        return int(self.Q[state].argmax())


# ─── 1. Monte Carlo (Every-Visit) ───────────────────────────
class MonteCarlo(BaseAlgorithm):
    """
    エピソード終了後にリターン G_t = Σ γ^k r_{t+k+1} を計算し
    各 (s,a) ペアに対して Q を G_t に向けて更新する。
        Q(s,a) ← Q(s,a) + α [G_t - Q(s,a)]
    """

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            episode = []                 # [(s, a, r), ...]
            state = env.reset()
            done  = False
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                episode.append((state, a, r))
                state = ns

            # 後ろから累積リターン G を計算して更新
            G = 0.0
            for s, a, r in reversed(episode):
                G = r + self.gamma * G
                self.Q[s, a] += self.alpha * (G - self.Q[s, a])

            rewards.append(sum(t[2] for t in episode))
            policy.on_episode_end()
        return rewards


# ─── 2. SARSA (On-policy TD) ────────────────────────────────
class SARSA(BaseAlgorithm):
    """
    on-policy TD(0):
        Q(s,a) ← Q(s,a) + α [r + γ Q(s', a') - Q(s,a)]
    a' は方策が実際に選ぶ行動。安全な方策を学ぶ傾向。
    """

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            action = policy.select(self.Q, state)
            done  = False
            total = 0.0
            while not done:
                ns, r, done = env.step(action)
                na = 0 if done else policy.select(self.Q, ns)
                target = r + (0.0 if done else self.gamma * self.Q[ns, na])
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                state, action = ns, na
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 3. Q-Learning (Off-policy TD) ──────────────────────────
class QLearning(BaseAlgorithm):
    """
    off-policy TD(0):
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
    最大化バイアスがあるため、確率的環境では Q を過大評価することがある。
    """

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done  = False
            total = 0.0
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                target = r + (0.0 if done else self.gamma * self.Q[ns].max())
                self.Q[state, a] += self.alpha * (target - self.Q[state, a])
                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 4. Expected SARSA ──────────────────────────────────────
class ExpectedSARSA(BaseAlgorithm):
    """
    SARSA の分散を抑える派生:
        Q(s,a) ← Q(s,a) + α [r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a)]

    π は ε-greedy を仮定して期待値を計算する。
    SARSA より低分散で精度が高くなる傾向。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        super().__init__(n_states, n_actions, alpha, gamma)
        self.epsilon = epsilon

    def _expected_q(self, state):
        """ε-greedy 方策に対する Q の期待値"""
        q = self.Q[state]
        n = len(q)
        max_a = q.argmax()
        # 同点を考慮した greedy 確率
        max_actions = np.flatnonzero(q == q.max())
        n_max = len(max_actions)
        prob = np.full(n, self.epsilon / n)         # ランダム成分
        prob[max_actions] += (1 - self.epsilon) / n_max
        return float((prob * q).sum())

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                target = r + (0.0 if done else self.gamma * self._expected_q(ns))
                self.Q[state, a] += self.alpha * (target - self.Q[state, a])
                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 5. Double Q-Learning ───────────────────────────────────
class DoubleQLearning(BaseAlgorithm):
    """
    Q_A, Q_B の 2 つのテーブルを交互に更新し、最大化バイアスを除去する。
    どちらを更新するかは 50% で選択。
        Q_A(s,a) ← Q_A(s,a) + α [r + γ Q_B(s', argmax_a' Q_A(s',a')) - Q_A(s,a)]

    確率的環境で Q-Learning より明確に良い精度を出す。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, seed=0):
        # 親の Q 初期化を流用しつつ、追加で QA / QB を持つ
        super().__init__(n_states, n_actions, alpha, gamma)
        self.QA = np.zeros((n_states, n_actions))
        self.QB = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    def _sync_Q(self):
        """policy / greedy_action 用に平均値を Q に反映する"""
        self.Q = (self.QA + self.QB) / 2.0

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                self._sync_Q()                 # 行動選択時は最新の平均を使う
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)
                if self.rng.random() < 0.5:
                    # QA を更新: 最大化は QA で、評価は QB で
                    best_a = int(self.QA[ns].argmax())
                    target = r + (0.0 if done else self.gamma * self.QB[ns, best_a])
                    self.QA[state, a] += self.alpha * (target - self.QA[state, a])
                else:
                    best_a = int(self.QB[ns].argmax())
                    target = r + (0.0 if done else self.gamma * self.QA[ns, best_a])
                    self.QB[state, a] += self.alpha * (target - self.QB[state, a])
                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        self._sync_Q()                          # 学習終了後も同期しておく
        return rewards


# ─── 6. Dyna-Q (model-based + model-free) ───────────────────
class DynaQ(BaseAlgorithm):
    """
    Dyna-Q (Sutton 1990):
      実環境からのサンプルで Q を更新するだけでなく、
      観測した遷移を「モデル」として保存し、それを使って
      n_planning 回の追加 Q 更新 (planning) を行う。

      → サンプル効率が大幅に向上。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 n_planning: int = 20, seed: int = 0):
        super().__init__(n_states, n_actions, alpha, gamma)
        self.n_planning = n_planning
        self.rng = np.random.default_rng(seed)
        # 決定的モデル: model[(s,a)] = (r, s')
        self.model: dict = {}
        self.observed_pairs: list = []

    def train(self, env, policy, n_episodes):
        rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            total = 0.0
            while not done:
                a = policy.select(self.Q, state)
                ns, r, done = env.step(a)

                # 1) 直接的 Q 学習更新
                target = r + (0.0 if done else self.gamma * self.Q[ns].max())
                self.Q[state, a] += self.alpha * (target - self.Q[state, a])

                # 2) モデル保存
                if (state, a) not in self.model:
                    self.observed_pairs.append((state, a))
                self.model[(state, a)] = (r, ns, done)

                # 3) プランニング (n_planning 回)
                if self.observed_pairs:
                    for _p in range(self.n_planning):
                        idx = int(self.rng.integers(len(self.observed_pairs)))
                        ps, pa = self.observed_pairs[idx]
                        pr, pns, pdone = self.model[(ps, pa)]
                        ptarget = pr + (0.0 if pdone else self.gamma * self.Q[pns].max())
                        self.Q[ps, pa] += self.alpha * (ptarget - self.Q[ps, pa])

                state = ns
                total += r
            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── 7. n-step SARSA ────────────────────────────────────────
class NStepSARSA(BaseAlgorithm):
    """
    n-step SARSA (Sutton & Barto 7.2 章)。

    n ステップ先まで実際の報酬を展開してから Q を更新する。

    n-step リターン:
        G_{t:t+n} = Σ_{k=0}^{n-1} γ^k r_{t+k+1}  +  γ^n Q(s_{t+n}, a_{t+n})
                    ※ 終端ステップ以降は Q=0

    更新:
        Q(s_t, a_t) ← Q(s_t, a_t) + α [G_{t:t+n} − Q(s_t, a_t)]

    n=1 のとき SARSA と等価。
    n が大きいほど MC に近づき、バイアスが減って分散が増す。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, n: int = 4):
        super().__init__(n_states, n_actions, alpha, gamma)
        self.n = n

    def train(self, env: BaseEnv, policy: BasePolicy, n_episodes: int) -> List[float]:
        rewards_list = []
        M = self.n + 1   # リングバッファサイズ

        for _ in range(n_episodes):
            s0 = env.reset()
            a0 = policy.select(self.Q, s0)

            states  = [0]   * M
            actions = [0]   * M
            rews    = [0.0] * M
            states[0]  = s0
            actions[0] = a0

            T     = float("inf")
            t     = 0
            total = 0.0

            while True:
                if t < T:
                    ns, r, done = env.step(actions[t % M])
                    total += r
                    rews[(t + 1) % M]   = r
                    states[(t + 1) % M] = ns
                    if done:
                        T = t + 1
                    else:
                        actions[(t + 1) % M] = policy.select(self.Q, ns)

                tau = t - self.n + 1
                if tau >= 0:
                    t_cap = int(T) if T != float("inf") else t + 1
                    G = sum(
                        (self.gamma ** (i - tau - 1)) * rews[i % M]
                        for i in range(tau + 1, min(tau + self.n, t_cap) + 1)
                    )
                    if T == float("inf") or tau + self.n < int(T):
                        G += (self.gamma ** self.n) * self.Q[
                            states[(tau + self.n) % M],
                            actions[(tau + self.n) % M],
                        ]
                    s_tau = states[tau % M]
                    a_tau = actions[tau % M]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])

                if tau == T - 1:
                    break
                t += 1

            rewards_list.append(total)
            policy.on_episode_end()
        return rewards_list


# ─── 8. SARSA(λ) — 適格性トレース ────────────────────────────
class SARSALambda(BaseAlgorithm):
    """
    SARSA(λ): 適格性トレース (eligibility traces) 付き SARSA。

    適格性トレース e(s,a) は過去に通過した (s,a) ペアの「責任度」を表し、
    TD 誤差 δ を全 (s,a) に e(s,a) の重みで一括伝播させる。

    Accumulating traces (蓄積型):
        e(s_t, a_t) ← γλ e(s_t, a_t) + 1   （訪問時は 1 を加算）

    TD 誤差と更新:
        δ = r  +  γ Q(s', a')  −  Q(s, a)
        Q(s, a) ← Q(s, a)  +  α δ e(s, a)   for all (s, a)
        e(s, a) ← γλ e(s, a)                 for all (s, a)

    λ=0 : SARSA(0) と等価 (ブートストラップのみ)
    λ=1 : Monte Carlo に近い動作 (長い credit assignment)
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, lam: float = 0.8):
        super().__init__(n_states, n_actions, alpha, gamma)
        self.lam = lam

    def train(self, env: BaseEnv, policy: BasePolicy, n_episodes: int) -> List[float]:
        rewards = []
        for _ in range(n_episodes):
            e      = np.zeros((self.n_states, self.n_actions))
            state  = env.reset()
            action = policy.select(self.Q, state)
            done   = False
            total  = 0.0

            while not done:
                ns, r, done = env.step(action)
                na = 0 if done else policy.select(self.Q, ns)

                delta = r + (0.0 if done else self.gamma * self.Q[ns, na]) - self.Q[state, action]
                e[state, action] += 1.0          # accumulating traces

                self.Q += self.alpha * delta * e
                e      *= self.gamma * self.lam

                state, action = ns, na
                total += r

            rewards.append(total)
            policy.on_episode_end()
        return rewards


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_algorithm(name: str, n_states: int, n_actions: int,
                  **kwargs) -> BaseAlgorithm:
    table = {
        "mc":           MonteCarlo,
        "sarsa":        SARSA,
        "q":            QLearning,
        "esarsa":       ExpectedSARSA,
        "double_q":     DoubleQLearning,
        "dyna_q":       DynaQ,
        "n_step_sarsa": NStepSARSA,
        "sarsa_lambda": SARSALambda,
    }
    if name not in table:
        raise ValueError(f"unknown algorithm: {name}")
    return table[name](n_states=n_states, n_actions=n_actions, **kwargs)
