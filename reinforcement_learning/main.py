"""
==============================================================
  main.py | 強化学習: 全組み合わせの比較実行
==============================================================
Env × Policy × Algorithm の全組み合わせを 3 seed 平均で評価する。
4 × 4 × 8 = 128 通り。

フェーズ:
  1. 全 128 通りを評価し環境ごとに mean ± std をランキング表示
  2. 各環境の最良構成を精度高く再評価
     - 単体: 800 エピソード × 5 seed で mean ± std
     - アンサンブル: 7 体の Q テーブルを平均した合成エージェントで評価

実行:
    python main.py
"""

from itertools import product

from env       import get_env
from policy    import get_policy
from algorithm import get_algorithm
from trainer   import train_and_evaluate, ensemble_train_and_evaluate


# ─── 比較する戦略 ───────────────────────────────────────────
ENVS       = ["gridworld", "cliffwalk", "stochastic", "windy"]
POLICIES   = ["eps", "decay_eps", "boltzmann", "ucb"]
ALGORITHMS = ["mc", "sarsa", "q", "esarsa", "double_q", "dyna_q",
              "n_step_sarsa", "sarsa_lambda"]

_ENV_DESC = {
    "gridworld":  ("4×4格子  スタート(0,0)→ゴール(3,3)",
                   "各ステップ-0.04 / ゴール+1  理論最適≈+0.72(7ステップ到達)"),
    "cliffwalk":  ("4×12格子  崖歩き問題 (Sutton & Barto)",
                   "通常ステップ-1 / 崖落下-100  最短路≈-13 / 安全路≈-17"),
    "stochastic": ("4×4格子  20%確率でスリップ・穴あり",
                   "ゴール+1 / 穴-1 / タイムアウト0  確率的なので値に幅が出る"),
    "windy":      ("7×10格子  列ごとに上向きの風 (強さ0~2)",
                   "毎ステップ-1  速くゴールするほど高い(少ない負の値が良い)"),
}
_POL_DESC = {
    "eps":        "ε-greedy ε=0.10 固定  (常に10%でランダム探索)",
    "decay_eps":  "ε-greedy ε:1.0→0.05 減衰  (序盤は探索、終盤は活用)",
    "boltzmann":  "Boltzmann/Softmax τ=0.5  (Q値の差に応じて確率的に選択)",
    "ucb":        "UCB c=1.4  (選択回数が少ない行動を優先して探索)",
}
_ALGO_DESC = {
    "mc":           "モンテカルロ法 (エピソード完了後に報酬を遡って更新)",
    "sarsa":        "SARSA (オンポリシー TD学習)",
    "q":            "Q-learning (オフポリシー TD学習  最大Q値で更新)",
    "esarsa":       "Expected SARSA (次状態の期待値で更新)",
    "double_q":     "Double Q-learning (最大化バイアスを2つのQテーブルで除去)",
    "dyna_q":       "Dyna-Q (実経験+モデルベースプランニング10回/ステップ)",
    "n_step_sarsa": "n-step SARSA n=4 (4ステップ先まで報酬を集めて更新)",
    "sarsa_lambda": "SARSA(λ) λ=0.8 (資格トレース: 過去の状態にも遡って更新)",
}


def make_factories(env_name, pol_name, algo_name):
    """seed を受け取って各部品を生成する factory を返す"""

    def env_factory(seed):
        if env_name == "stochastic":
            return get_env(env_name, seed=seed)
        return get_env(env_name)

    def policy_factory(seed):
        if pol_name == "eps":
            return get_policy(pol_name, epsilon=0.1, seed=seed)
        if pol_name == "decay_eps":
            return get_policy(pol_name, epsilon=1.0, decay=0.995,
                              epsilon_min=0.05, seed=seed)
        if pol_name == "boltzmann":
            return get_policy(pol_name, tau=0.5, seed=seed)
        if pol_name == "ucb":
            return get_policy(pol_name, c=1.4, seed=seed)
        raise ValueError(pol_name)

    def algo_factory(n_states, n_actions, seed):
        common = dict(alpha=0.1, gamma=0.95)
        if algo_name == "esarsa":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, epsilon=0.1)
        if algo_name == "double_q":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, seed=seed)
        if algo_name == "dyna_q":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, n_planning=10, seed=seed)
        if algo_name == "n_step_sarsa":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, n=4)
        if algo_name == "sarsa_lambda":
            return get_algorithm(algo_name, n_states, n_actions,
                                 **common, lam=0.8)
        return get_algorithm(algo_name, n_states, n_actions, **common)

    return env_factory, policy_factory, algo_factory


def run_all_combinations(n_episodes: int = 400, n_seeds: int = 3):
    """全組み合わせを評価し結果リストを返す"""
    total = len(ENVS) * len(POLICIES) * len(ALGORITHMS)

    print("=" * 66)
    print("  フェーズ 1/2 : 全アルゴリズム組み合わせ比較")
    print("=" * 66)
    print(f"  【処理内容】")
    print(f"    エージェントが環境の中で試行錯誤を繰り返し、")
    print(f"    より多くの報酬を得られる行動方針(ポリシー)を自動学習します。")
    print(f"      ① 環境 (Env)    : エージェントが行動するフィールド")
    print(f"      ② 方策 (Policy) : 状態を見てどの行動を選ぶかの探索戦略")
    print(f"      ③ アルゴリズム  : Q値テーブルをどう更新するかの学習則")
    print(f"    {len(ENVS)}環境 × {len(POLICIES)}方策 × {len(ALGORITHMS)}アルゴリズム = {total}通り")
    print(f"    各組み合わせを {n_seeds} seed で学習し平均報酬を評価します。")
    print()
    print(f"  【評価方法】")
    print(f"    訓練後に貪欲方策(完全に学習済みの行動のみ)で30エピソード実行し、")
    print(f"    その平均報酬を評価スコアとします。スコアが高いほど良い学習結果です。")
    print()

    rows = []
    done = 0
    for env_name, pol_name, algo_name in product(ENVS, POLICIES, ALGORITHMS):
        done += 1
        print(f"\r  学習中 [{done:>3}/{total}]  {env_name:<12}{pol_name:<12}{algo_name:<14}",
              end="", flush=True)
        envF, polF, algoF = make_factories(env_name, pol_name, algo_name)
        try:
            mean, std, _ = train_and_evaluate(
                envF, polF, algoF,
                n_episodes=n_episodes,
                n_eval_episodes=30,
                n_seeds=n_seeds,
            )
            rows.append((env_name, pol_name, algo_name, mean, std))
        except Exception as e:
            rows.append((env_name, pol_name, algo_name, None, type(e).__name__))

    print(f"\r  学習完了 [{total}/{total}]" + " " * 50)

    # 環境ごとに結果を表示
    for env_name in ENVS:
        short_desc, reward_desc = _ENV_DESC[env_name]
        env_rows = sorted(
            [r for r in rows if r[0] == env_name and isinstance(r[3], float)],
            key=lambda r: -r[3],
        )
        best = env_rows[0] if env_rows else None

        print()
        print(f"  ┌─ 環境: {env_name}  ({short_desc})")
        print(f"  │  報酬の目安: {reward_desc}")
        print(f"  │  mean=平均評価報酬  std=seed間のばらつき(小さいほど安定)")
        print(f"  │")
        print(f"  │  {'policy':<12}{'algorithm':<16}{'mean':>10}{'± std':>10}")
        print(f"  │  " + "-" * 48)
        for _, pol, algo, mean, std in env_rows:
            marker = " ←最良" if best is not None and (pol, algo) == (best[1], best[2]) else ""
            print(f"  │  {pol:<12}{algo:<16}{mean:>10.3f}{std:>10.3f}{marker}")
        if best:
            print(f"  └→ 最良: {best[1]} × {best[2]}  mean={best[3]:.3f}")
            print(f"     方策: {_POL_DESC[best[1]]}")
            print(f"     学習: {_ALGO_DESC[best[2]]}")

    return rows


def run_recommended(rows):
    """各環境の比較結果から最良構成を詳細評価する"""
    print()
    print("=" * 66)
    print("  フェーズ 2/2 : 各環境の最良構成を精度高く再評価")
    print("=" * 66)
    print(f"  【処理内容】")
    print(f"    フェーズ1で環境ごとに最も平均報酬が高かった構成を選び、")
    print(f"    seed数5・評価50エピソードでより正確に再測定します。")
    print()

    for env_name in ENVS:
        env_rows = sorted(
            [r for r in rows if r[0] == env_name and isinstance(r[3], float)],
            key=lambda r: -r[3],
        )
        if not env_rows:
            continue
        _, pol_name, algo_name, _, _ = env_rows[0]
        short_desc, reward_desc = _ENV_DESC[env_name]
        envF, polF, algoF = make_factories(env_name, pol_name, algo_name)

        # ① 単体エージェント (seed=5, episode=800) で再評価
        mean_single, std_single, _ = train_and_evaluate(
            envF, polF, algoF,
            n_episodes=800,
            n_eval_episodes=50,
            n_seeds=5,
        )

        # ② アンサンブル (7エージェントのQテーブルを平均, episode=800)
        mean_ens, std_ens = ensemble_train_and_evaluate(
            envF, polF, algoF,
            n_episodes=800,
            n_eval_episodes=50,
            n_agents=7,
        )

        print(f"  {env_name:<12} ({short_desc})")
        print(f"    方策: {pol_name:<12}  学習: {algo_name}")
        print(f"    目安: {reward_desc}")
        print(f"    単体 (800ep×5seed) : mean={mean_single:7.3f} ± {std_single:.3f}")
        print(f"    アンサンブル (7体平均): mean={mean_ens:7.3f} ± {std_ens:.3f}"
              + ("  ↑改善" if mean_ens > mean_single else ""))
        print()


if __name__ == "__main__":
    rows = run_all_combinations(n_episodes=600, n_seeds=3)
    run_recommended(rows)
