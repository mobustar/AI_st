"""
==============================================================
  env.py | 強化学習用環境 (戦略パターン)
==============================================================
全環境は OpenAI Gym 風の最小インタフェース:
    state = env.reset()
    next_state, reward, done = env.step(action)

選択環境:
  1. GridWorld      - 4×4 格子。各ステップ -0.04、ゴール +1.0
  2. CliffWalk      - 4×12 格子。Sutton & Barto 古典問題 (SARSA vs Q の差が顕著)
  3. StochasticGrid - 4×4 格子。20% の確率でスリップする確率的環境
  4. WindyGridWorld - 7×10 格子。列ごとに風があり上方向に押される

格子座標系: (row, col), 行動: 0=上, 1=右, 2=下, 3=左
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseEnv(ABC):
    n_actions: int = 4

    @property
    @abstractmethod
    def n_states(self) -> int: ...

    @abstractmethod
    def reset(self) -> int: ...

    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool]: ...


# ─── 1. シンプル GridWorld ──────────────────────────────────
class GridWorld(BaseEnv):
    """
    4x4 格子。スタート (0,0) → ゴール (3,3) を目指す。
    各ステップで -0.04 の報酬 (短経路を促す)、ゴールで +1。
    """

    H, W = 4, 4
    GOAL = (3, 3)
    ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps

    @property
    def n_states(self) -> int:
        return self.H * self.W

    def reset(self) -> int:
        self.pos    = (0, 0)
        self._steps = 0
        return self._encode(self.pos)

    def _encode(self, pos):
        return pos[0] * self.W + pos[1]

    def step(self, action: int):
        self._steps += 1
        dy, dx = self.ACTIONS[action]
        ny = int(np.clip(self.pos[0] + dy, 0, self.H - 1))
        nx = int(np.clip(self.pos[1] + dx, 0, self.W - 1))
        self.pos = (ny, nx)

        if self.pos == self.GOAL:
            return self._encode(self.pos), 1.0, True
        if self._steps >= self.max_steps:
            return self._encode(self.pos), -0.04, True
        return self._encode(self.pos), -0.04, False


# ─── 2. CliffWalk (Sutton & Barto Example 6.6) ──────────────
class CliffWalk(BaseEnv):
    """
    4x12 の崖歩き問題。
        S _ _ _ _ _ _ _ _ _ _ G
        ↑ 最下段 (1..10 列) は崖。落ちると S にリセット、報酬 -100
        通常移動の報酬は -1
    SARSA は安全な経路 (上回り) を、Q-learning は崖際の最短経路を学習する。
    """

    H, W = 4, 12
    START = (3, 0)
    GOAL  = (3, 11)
    ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps

    @property
    def n_states(self) -> int:
        return self.H * self.W

    def reset(self) -> int:
        self.pos    = self.START
        self._steps = 0
        return self._encode(self.pos)

    def _encode(self, pos):
        return pos[0] * self.W + pos[1]

    def _is_cliff(self, pos):
        return pos[0] == self.H - 1 and 1 <= pos[1] <= self.W - 2

    def step(self, action: int):
        self._steps += 1
        dy, dx = self.ACTIONS[action]
        ny = int(np.clip(self.pos[0] + dy, 0, self.H - 1))
        nx = int(np.clip(self.pos[1] + dx, 0, self.W - 1))
        new_pos = (ny, nx)

        if self._is_cliff(new_pos):
            # 崖に落ちたらスタートに戻り大きな負報酬
            self.pos = self.START
            return self._encode(self.pos), -100.0, False
        self.pos = new_pos
        if self.pos == self.GOAL:
            return self._encode(self.pos), -1.0, True
        if self._steps >= self.max_steps:
            return self._encode(self.pos), -1.0, True
        return self._encode(self.pos), -1.0, False


# ─── 3. 確率的グリッド (滑り付き) ───────────────────────────
class StochasticGrid(BaseEnv):
    """
    GridWorld と似ているが、確率 slip_prob で意図と異なる隣接行動が起きる。
    確率的環境では Q-learning の最大化バイアスが顕著に現れるため、
    Double Q-learning との比較に有用。

    地図: 4x4
        S _ _ _
        _ X _ _    X = 穴 (報酬 -1, 終了)
        _ _ _ X
        _ _ _ G    G = ゴール (報酬 +1, 終了)
    """

    H, W = 4, 4
    START = (0, 0)
    GOAL  = (3, 3)
    HOLES = {(1, 1), (2, 3)}
    ACTIONS = [(-1,0),(0,1),(1,0),(0,-1)]

    def __init__(self, slip_prob: float = 0.2, max_steps: int = 100, seed: int = 0):
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    @property
    def n_states(self) -> int:
        return self.H * self.W

    def reset(self) -> int:
        self.pos    = self.START
        self._steps = 0
        return self._encode(self.pos)

    def _encode(self, pos):
        return pos[0] * self.W + pos[1]

    def step(self, action: int):
        self._steps += 1
        # 滑り判定: slip_prob で左右どちらかに行動が変わる
        if self.rng.random() < self.slip_prob:
            action = (action + self.rng.choice([-1, 1])) % 4

        dy, dx = self.ACTIONS[action]
        ny = int(np.clip(self.pos[0] + dy, 0, self.H - 1))
        nx = int(np.clip(self.pos[1] + dx, 0, self.W - 1))
        self.pos = (ny, nx)

        if self.pos == self.GOAL:
            return self._encode(self.pos), 1.0, True
        if self.pos in self.HOLES:
            return self._encode(self.pos), -1.0, True
        if self._steps >= self.max_steps:
            return self._encode(self.pos), 0.0, True
        return self._encode(self.pos), 0.0, False


# ─── 4. Windy GridWorld (Sutton & Barto Example 6.5) ─────────
class WindyGridWorld(BaseEnv):
    """
    風のある格子世界 (Sutton & Barto Example 6.5)。

    7×10 の格子。各列に「風」が設定されており、
    エージェントが行動するたびに風の強さ分だけ上方向 (row 減少) に押される。

    ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
    │  │  │  │  │  │  │  │  │  │  │  row 0
    │  │  │  │  │  │  │  │  │  │  │  row 1
    │  │  │  │  │  │  │  │  │  │  │  row 2
    │ S│  │  │  │  │  │  │ G│  │  │  row 3
    │  │  │  │  │  │  │  │  │  │  │  row 4
    │  │  │  │  │  │  │  │  │  │  │  row 5
    │  │  │  │  │  │  │  │  │  │  │  row 6
    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
    風(列): 0  0  0  1  1  1  2  2  1  0

    各ステップ: −1 の報酬 / ゴール到達でエピソード終了
    """

    H, W   = 7, 10
    START  = (3, 0)
    GOAL   = (3, 7)
    WIND   = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps

    @property
    def n_states(self) -> int:
        return self.H * self.W

    def reset(self) -> int:
        self.pos    = self.START
        self._steps = 0
        return self._encode(self.pos)

    def _encode(self, pos):
        return pos[0] * self.W + pos[1]

    def step(self, action: int):
        self._steps += 1
        dy, dx = self.ACTIONS[action]
        wind = self.WIND[self.pos[1]]
        ny = int(np.clip(self.pos[0] + dy - wind, 0, self.H - 1))
        nx = int(np.clip(self.pos[1] + dx,        0, self.W - 1))
        self.pos = (ny, nx)
        if self.pos == self.GOAL:
            return self._encode(self.pos), -1.0, True
        if self._steps >= self.max_steps:
            return self._encode(self.pos), -1.0, True
        return self._encode(self.pos), -1.0, False


# ─── ファクトリ関数 ─────────────────────────────────────────
def get_env(name: str = "gridworld", **kwargs) -> BaseEnv:
    table = {
        "gridworld":  GridWorld,
        "cliffwalk":  CliffWalk,
        "stochastic": StochasticGrid,
        "windy":      WindyGridWorld,
    }
    if name not in table:
        raise ValueError(f"unknown env: {name}")
    return table[name](**kwargs)
