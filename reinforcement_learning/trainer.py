"""
==============================================================
  trainer.py | 強化学習トレーナー
==============================================================
Env × Policy × Algorithm を組み合わせて学習・評価する。

提供する関数:
  train_and_evaluate()        - 複数 seed で独立学習し mean ± std を返す
  ensemble_train_and_evaluate() - 複数エージェントの Q テーブルを平均し
                                  合成エージェントを評価する

Q テーブルアンサンブルの原理:
  n_agents 個のエージェントをそれぞれ異なる seed で学習させる。
  各エージェントの Q テーブルを要素平均することで、
  個々の偶然性による局所解への偏りを打ち消し、
  より安定した方策を得る。
"""

from typing import List, Tuple
import numpy as np

from env       import BaseEnv
from policy    import BasePolicy
from algorithm import BaseAlgorithm


class _AveragedQAgent:
    """N エージェントの Q テーブルを平均した合成エージェント"""
    def __init__(self, Q: np.ndarray):
        self.Q = Q

    def greedy_action(self, state: int) -> int:
        return int(self.Q[state].argmax())


def evaluate_greedy(env: BaseEnv, algo: BaseAlgorithm,
                    n_episodes: int = 50) -> float:
    """
    学習後の貪欲方策で平均報酬を測る。
    確率的環境では複数試行の平均が必要。
    """
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = algo.greedy_action(state)
            state, r, done = env.step(a)
            ep_r += r
        total += ep_r
    return total / n_episodes


def train_and_evaluate(env_factory, policy_factory, algo_factory,
                        n_episodes: int = 500,
                        n_eval_episodes: int = 50,
                        n_seeds: int = 3) -> Tuple[float, float, List[List[float]]]:
    """
    異なる seed で n_seeds 回学習を行い、評価平均と標準偏差を返す。

    Args:
        env_factory:    seed → BaseEnv  を返す callable
        policy_factory: seed → BasePolicy
        algo_factory:   (n_states, n_actions, seed) → BaseAlgorithm

    Returns:
        mean_eval: 平均評価報酬
        std_eval:  評価報酬の標準偏差
        all_train_curves: 各 seed の訓練報酬曲線
    """
    eval_rewards = []
    train_curves = []
    for seed in range(n_seeds):
        env  = env_factory(seed)
        algo = algo_factory(env.n_states, env.n_actions, seed)
        pol  = policy_factory(seed)
        train_curve = algo.train(env, pol, n_episodes)
        train_curves.append(train_curve)

        # 評価用に環境をリセットして貪欲方策で報酬計測
        eval_env = env_factory(seed + 10000)
        eval_r   = evaluate_greedy(eval_env, algo, n_eval_episodes)
        eval_rewards.append(eval_r)

    return float(np.mean(eval_rewards)), float(np.std(eval_rewards)), train_curves


def ensemble_train_and_evaluate(env_factory, policy_factory, algo_factory,
                                n_episodes: int = 800,
                                n_eval_episodes: int = 50,
                                n_agents: int = 7) -> Tuple[float, float]:
    """
    N エージェントを別 seed で学習し Q テーブルを平均して評価する。

    単一エージェントは初期化の偶然性で局所解に陥ることがあるが、
    複数エージェントの Q テーブルを平均することで偏りを打ち消し
    より安定した高品質な方策を得る。

    Returns:
        (mean_eval, std_eval) : アンサンブル方策の平均報酬と複数評価の標準偏差
    """
    Qs = []
    for seed in range(n_agents):
        env  = env_factory(seed)
        algo = algo_factory(env.n_states, env.n_actions, seed)
        pol  = policy_factory(seed)
        algo.train(env, pol, n_episodes)
        Qs.append(algo.Q.copy())

    ensemble_agent = _AveragedQAgent(np.mean(Qs, axis=0))

    eval_rewards = []
    for seed in range(5):
        eval_env = env_factory(seed + 20000)
        eval_rewards.append(evaluate_greedy(eval_env, ensemble_agent, n_eval_episodes))

    return float(np.mean(eval_rewards)), float(np.std(eval_rewards))
