"""Train every algorithm, save learning curves, and print a comparison table.

Runs two experiments:

  1. CliffWalking  — Q-learning vs SARSA vs Expected-SARSA. Reproduces the
     classic on-policy / off-policy result (safe vs optimal path).
  2. GridWorld     — all four algorithms (incl. REINFORCE) on a shared task.

Each run is averaged over several seeds. Figures go to figures/, and a
markdown comparison table is written to figures/results.md so the README can
quote real numbers.

Usage:  python train.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe
import matplotlib.pyplot as plt
import numpy as np

from rllab.agents import ExpectedSarsa, QLearning, Reinforce, Sarsa
from rllab.envs import CliffWalking, GridWorld
from rllab.train_loop import (
    evaluate_greedy,
    greedy_path,
    train_reinforce,
    train_td,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def smooth(x: np.ndarray, w: int = 10) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def run_td_multi(agent_cls, env_fn, episodes, **kw):
    """Train a TD agent over several seeds; return (mean_curve, final_evals)."""
    curves, evals = [], []
    for seed in SEEDS:
        env = env_fn()
        agent = agent_cls(env.n_states, env.n_actions, seed=seed, **kw)
        curves.append(train_td(env, agent, episodes))
        evals.append(evaluate_greedy(env, agent, episodes=20))
    return np.mean(curves, axis=0), np.array(evals)


def run_reinforce_multi(env_fn, episodes, **kw):
    curves, evals = [], []
    for seed in SEEDS:
        env = env_fn()
        agent = Reinforce(env.n_states, env.n_actions, seed=seed, **kw)
        curves.append(train_reinforce(env, agent, episodes))
        evals.append(evaluate_greedy(env, agent, episodes=20))
    return np.mean(curves, axis=0), np.array(evals)


def cliff_path_summary(agent_cls, **kw):
    """Describe the greedy path on the cliff, averaged over seeds.

    Returns (median path length, median gap-to-cliff over seeds). The gap is the
    smallest number of rows between the greedy path's interior cells and the
    cliff row — Q-learning hugs the edge (gap 1), SARSA stays back (gap > 1).
    """
    cliff_row = CliffWalking().rows - 1
    lengths, gaps = [], []
    for seed in SEEDS:
        env = CliffWalking()
        agent = agent_cls(env.n_states, env.n_actions, seed=seed, **kw)
        train_td(env, agent, episodes=500)
        path = greedy_path(env, agent)
        interior = [(r, c) for (r, c) in path if 0 < c < env.cols - 1]
        gap = min((cliff_row - r) for (r, c) in interior) if interior else cliff_row
        lengths.append(len(path) - 1)
        gaps.append(gap)
    return int(np.median(lengths)), int(np.median(gaps))


# ---------------------------------------------------------------------------
# Experiment 1: CliffWalking
# ---------------------------------------------------------------------------
def experiment_cliff():
    print("\n=== CliffWalking: Q-learning vs SARSA vs Expected-SARSA ===")
    episodes = 500
    kw = dict(alpha=0.5, gamma=1.0, epsilon=0.1)

    q_curve, q_eval = run_td_multi(QLearning, CliffWalking, episodes, **kw)
    s_curve, s_eval = run_td_multi(Sarsa, CliffWalking, episodes, **kw)
    e_curve, e_eval = run_td_multi(ExpectedSarsa, CliffWalking, episodes, **kw)

    plt.figure(figsize=(8, 4.5))
    for curve, label in [
        (q_curve, "Q-learning (off-policy)"),
        (s_curve, "SARSA (on-policy)"),
        (e_curve, "Expected-SARSA"),
    ]:
        plt.plot(smooth(curve), label=label, linewidth=1.6)
    plt.ylim(-200, 0)
    plt.xlabel("Episode (10-episode moving average)")
    plt.ylabel("Sum of rewards during episode")
    plt.title("CliffWalking — learning curves (mean of 5 seeds)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cliff_learning_curves.png"), dpi=130)
    plt.close()

    results = []
    for name, ev, cls in [
        ("Q-learning", q_eval, QLearning),
        ("SARSA", s_eval, Sarsa),
        ("Expected-SARSA", e_eval, ExpectedSarsa),
    ]:
        plen, gap = cliff_path_summary(cls, **kw)
        results.append((name, ev.mean(), ev.std(), plen, gap))
        print(
            f"  {name:16s} greedy return = {ev.mean():7.2f} +/- {ev.std():4.2f}"
            f"  | greedy path: {plen} steps, gap-to-cliff = {gap} row(s)"
        )
    return results


# ---------------------------------------------------------------------------
# Experiment 2: GridWorld (all four algorithms)
# ---------------------------------------------------------------------------
def experiment_gridworld():
    print("\n=== GridWorld (5x5): all four algorithms ===")

    def env_fn():
        return GridWorld(rows=5, cols=5, walls=((1, 1), (2, 1), (3, 1)),
                         max_steps=200)

    td_eps = 400
    td_kw = dict(alpha=0.5, gamma=0.99, epsilon=0.1)

    q_curve, q_eval = run_td_multi(QLearning, env_fn, td_eps, **td_kw)
    s_curve, s_eval = run_td_multi(Sarsa, env_fn, td_eps, **td_kw)
    e_curve, e_eval = run_td_multi(ExpectedSarsa, env_fn, td_eps, **td_kw)

    # REINFORCE needs more episodes and a softer feature signal; use the 4x4
    # variant where vanilla policy gradient converges to the optimal path.
    def env_fn_pg():
        return GridWorld(rows=4, cols=4, walls=((1, 1),), max_steps=100)

    r_curve, r_eval = run_reinforce_multi(
        env_fn_pg, 1500, alpha=0.02, gamma=0.99, entropy_beta=0.01
    )

    plt.figure(figsize=(8, 4.5))
    for curve, label in [
        (q_curve, "Q-learning"),
        (s_curve, "SARSA"),
        (e_curve, "Expected-SARSA"),
    ]:
        plt.plot(smooth(curve), label=label, linewidth=1.6)
    plt.xlabel("Episode (10-episode moving average)")
    plt.ylabel("Sum of rewards during episode")
    plt.title("GridWorld 5x5 — TD learning curves (mean of 5 seeds)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "gridworld_learning_curves.png"), dpi=130)
    plt.close()

    # REINFORCE on its own axes (different env + episode budget)
    plt.figure(figsize=(8, 4.5))
    plt.plot(smooth(r_curve, 25), label="REINFORCE (policy gradient)",
             color="#b45309", linewidth=1.6)
    plt.xlabel("Episode (25-episode moving average)")
    plt.ylabel("Sum of rewards during episode")
    plt.title("GridWorld 4x4 — REINFORCE learning curve (mean of 5 seeds)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reinforce_learning_curve.png"), dpi=130)
    plt.close()

    results = []
    for name, ev in [
        ("Q-learning", q_eval),
        ("SARSA", s_eval),
        ("Expected-SARSA", e_eval),
        ("REINFORCE", r_eval),
    ]:
        results.append((name, ev.mean(), ev.std()))
        print(f"  {name:16s} greedy return = {ev.mean():7.2f} +/- {ev.std():4.2f}")
    return results


def write_results_md(cliff, grid):
    lines = ["# Results (real runs, mean of 5 seeds)", ""]
    lines.append("## CliffWalking (4x12)")
    lines.append("")
    lines.append("| Algorithm | Greedy return | Greedy path | Gap to cliff |")
    lines.append("|---|---:|---:|---:|")
    for name, mean, std, plen, gap in cliff:
        lines.append(f"| {name} | {mean:.2f} +/- {std:.2f} | {plen} steps | {gap} row(s) |")
    lines.append("")
    lines.append("## GridWorld (5x5 for TD, 4x4 for REINFORCE)")
    lines.append("")
    lines.append("| Algorithm | Greedy return |")
    lines.append("|---|---:|")
    for name, mean, std in grid:
        lines.append(f"| {name} | {mean:.2f} +/- {std:.2f} |")
    lines.append("")
    with open(os.path.join(FIG_DIR, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {os.path.join(FIG_DIR, 'results.md')}")


def main():
    np.random.seed(0)
    cliff = experiment_cliff()
    grid = experiment_gridworld()
    write_results_md(cliff, grid)
    print("\nFigures written to figures/. Done.")


if __name__ == "__main__":
    main()
