"""
==============================================================
  policy.py | 行動選択ポリシー (戦略パターン)
==============================================================
Q 値から行動を選ぶ方策。探索 (exploration) と
活用 (exploitation) のバランスを取る役割を担う。

選択アルゴリズム:
  1. EpsilonGreedy       - ε で一様ランダム、それ以外は貪欲
  2. DecayEpsilonGreedy  - エピソードごとに ε を減衰
  3. Boltzmann           - softmax(Q/τ) の確率分布で選択
  4. UCB1                - 訪問回数を考慮した上限信頼区間

精度の傾向:
  確率的環境では Boltzmann / UCB のほうが偏り少なく学習が安定。
  小さな決定的環境では EpsilonGreedy で十分。
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def select(self, Q: np.ndarray, state: int) -> int: ...

    def on_episode_end(self):
        """エピソード終了時のフック (デフォルト: 何もしない)"""


# ─── 1. ε-greedy ────────────────────────────────────────────
class EpsilonGreedy(BasePolicy):
    """確率 ε でランダム、1-ε で argmax_a Q(s,a)"""

    def __init__(self, epsilon: float = 0.1, seed: int = 0):
        self.epsilon = epsilon
        self.rng     = np.random.default_rng(seed)

    def select(self, Q, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(Q.shape[1]))
        # 同点処理: 最大値が複数あったらランダムに 1 つ選ぶ (バイアス回避)
        q = Q[state]
        max_q = q.max()
        candidates = np.flatnonzero(q == max_q)
        return int(self.rng.choice(candidates))


# ─── 2. ε 減衰版 ────────────────────────────────────────────
class DecayEpsilonGreedy(EpsilonGreedy):
    """
    ε を ε_min を下限に幾何減衰させる。
    探索→活用への滑らかな移行が学習後期の安定性を高める。
    """

    def __init__(self, epsilon: float = 1.0,
                 decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 seed: int = 0):
        super().__init__(epsilon, seed)
        self.decay        = decay
        self.epsilon_min  = epsilon_min

    def on_episode_end(self):
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)


# ─── 3. Boltzmann (Softmax) ─────────────────────────────────
class Boltzmann(BasePolicy):
    """
    P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)

    温度 τ が大きいほど一様、小さいほど決定的。
    log-sum-exp で数値安定化。
    """

    def __init__(self, tau: float = 1.0, seed: int = 0):
        self.tau = tau
        self.rng = np.random.default_rng(seed)

    def select(self, Q, state):
        q = Q[state] / max(self.tau, 1e-8)
        q -= q.max()                         # オーバーフロー回避
        p  = np.exp(q)
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))


# ─── 4. UCB1 ────────────────────────────────────────────────
class UCB1(BasePolicy):
    """
    Upper Confidence Bound:
        a* = argmax_a [ Q(s,a) + c · sqrt(ln(N(s)) / N(s,a)) ]

    まだ試行回数が少ない行動にボーナスを与えて探索を促す。
    確率的環境で偏り少なく収束させる原理的な方法。
    """

    def __init__(self, c: float = 1.4, n_states: int = None,
                 n_actions: int = None, seed: int = 0):
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.N_sa: np.ndarray = None
        self.N_s:  np.ndarray = None
        if n_states is not None and n_actions is not None:
            self._init_counts(n_states, n_actions)

    def _init_counts(self, n_states: int, n_actions: int):
        self.N_sa = np.zeros((n_states, n_actions), dtype=np.int64)
        self.N_s  = np.zeros(n_states,             dtype=np.int64)

    def select(self, Q, state):
        # 遅延初期化 (Q.shape を使う)
        if self.N_sa is None:
            self._init_counts(*Q.shape)

        # まだ試行していない行動があればそれを優先
        untried = np.flatnonzero(self.N_sa[state] == 0)
        if untried.size > 0:
            a = int(self.rng.choice(untried))
        else:
            ln_Ns = np.log(max(self.N_s[state], 1))
            ucb = Q[state] + self.c * np.sqrt(ln_Ns / self.N_sa[state])
            max_v = ucb.max()
            candidates = np.flatnonzero(ucb == max_v)
            a = int(self.rng.choice(candidates))

        # 選んだ行動の試行回数を更新
        self.N_sa[state, a] += 1
        self.N_s[state]     += 1
        return a


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_policy(name: str = "decay_eps", **kwargs) -> BasePolicy:
    table = {
        "eps":       EpsilonGreedy,
        "decay_eps": DecayEpsilonGreedy,
        "boltzmann": Boltzmann,
        "ucb":       UCB1,
    }
    if name not in table:
        raise ValueError(f"unknown policy: {name}")
    return table[name](**kwargs)
