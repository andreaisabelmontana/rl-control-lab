"""Learning tests: convergence, on-policy vs off-policy, policy gradient."""

import numpy as np

from rllab.agents import ExpectedSarsa, QLearning, Reinforce, Sarsa
from rllab.envs import CliffWalking, GridWorld
from rllab.train_loop import (
    evaluate_greedy,
    greedy_path,
    train_reinforce,
    train_td,
)


def _random_baseline_return(env, episodes=200, seed=0):
    """Average return of a uniformly random policy."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(episodes):
        env.reset()
        total, done, steps = 0.0, False, 0
        while not done and steps < env.max_steps:
            a = int(rng.integers(env.n_actions))
            _, r, done, _ = env.step(a)
            total += r
            steps += 1
        totals.append(total)
    return float(np.mean(totals))


def test_q_learning_converges_near_optimal_gridworld():
    env = GridWorld(rows=5, cols=5, walls=((1, 1), (2, 1), (3, 1)))
    agent = QLearning(env.n_states, env.n_actions, alpha=0.5, gamma=0.99,
                      epsilon=0.1, seed=0)
    returns = train_td(env, agent, episodes=400)

    # learning improves: last 50 episodes far better than first 50
    early = returns[:50].mean()
    late = returns[-50:].mean()
    assert late > early + 5

    # greedy policy beats a random policy by a clear margin
    greedy = evaluate_greedy(env, agent, episodes=20)
    random_ret = _random_baseline_return(env)
    assert greedy > random_ret + 10

    # optimal shortest path on a 5x5 with a wall column is 8 steps -> return -8.
    # the greedy return should be at/near optimal (allow a little slack).
    assert greedy >= -10
    path = greedy_path(env, agent)
    assert env.index_to_pos(env._idx(env.goal)) == path[-1]


def test_sarsa_safer_than_qlearning_on_cliff():
    # Q-learning learns the optimal risky path hugging the cliff edge (row 3->2);
    # SARSA learns a safer path further from the edge. Compare how close each
    # greedy path comes to the cliff.
    def train(agent_cls, seed):
        env = CliffWalking()
        agent = agent_cls(env.n_states, env.n_actions, alpha=0.5, gamma=1.0,
                          epsilon=0.1, seed=seed)
        train_td(env, agent, episodes=500)
        return env, agent

    def min_row_distance_to_cliff(env, agent):
        """Smallest gap (in rows) between the greedy path and the cliff row."""
        path = greedy_path(env, agent)
        cliff_row = env.rows - 1
        # distance of each visited interior cell to the cliff row
        dists = [cliff_row - r for (r, c) in path
                 if 0 < c < env.cols - 1]
        return min(dists) if dists else cliff_row

    q_env, q_agent = train(QLearning, seed=1)
    s_env, s_agent = train(Sarsa, seed=1)

    q_gap = min_row_distance_to_cliff(q_env, q_agent)
    s_gap = min_row_distance_to_cliff(s_env, s_agent)

    # Q-learning hugs the cliff (gap 1); SARSA keeps further away (gap > 1).
    assert q_gap == 1
    assert s_gap > q_gap


def test_qlearning_higher_eval_return_than_sarsa_on_cliff():
    # Greedy paths: Q-learning's optimal path is shorter (return -13),
    # SARSA's safe path is longer (more negative). Greedy eval should reflect it.
    def final_greedy(agent_cls, seed=2):
        env = CliffWalking()
        agent = agent_cls(env.n_states, env.n_actions, alpha=0.5, gamma=1.0,
                          epsilon=0.1, seed=seed)
        train_td(env, agent, episodes=500)
        return evaluate_greedy(env, agent, episodes=5)

    q = final_greedy(QLearning)
    s = final_greedy(Sarsa)
    # optimal cliff path return is -13; allow exact-ish
    assert q >= -15
    assert q >= s  # Q-learning's greedy path is at least as good


def test_expected_sarsa_learns_cliff():
    env = CliffWalking()
    agent = ExpectedSarsa(env.n_states, env.n_actions, alpha=0.5, gamma=1.0,
                          epsilon=0.1, seed=3)
    returns = train_td(env, agent, episodes=500)
    assert returns[-50:].mean() > returns[:50].mean()
    assert evaluate_greedy(env, agent, episodes=5) > -50


def test_reinforce_improves_average_return():
    env = GridWorld(rows=4, cols=4, walls=((1, 1),), max_steps=100)
    agent = Reinforce(env.n_states, env.n_actions, alpha=0.02, gamma=0.99,
                      entropy_beta=0.01, seed=0)
    returns = train_reinforce(env, agent, episodes=1500)

    # average return improves over training
    early = returns[:100].mean()
    late = returns[-100:].mean()
    assert late > early + 4

    # the learned policy beats a random policy by a clear margin
    random_ret = _random_baseline_return(env)
    assert late > random_ret + 20

    # and the greedy policy converges to the optimal short path (return -5)
    greedy = evaluate_greedy(env, agent, episodes=20)
    assert greedy >= -6


def test_reinforce_converges_across_seeds():
    # The default (undiscounted-gradient) REINFORCE reliably reaches the
    # optimal path on every seed; the strict gamma^t variant does not.
    for seed in range(5):
        env = GridWorld(rows=4, cols=4, walls=((1, 1),), max_steps=100)
        agent = Reinforce(env.n_states, env.n_actions, alpha=0.02, gamma=0.99,
                          entropy_beta=0.01, seed=seed)
        train_reinforce(env, agent, episodes=1500)
        assert evaluate_greedy(env, agent, episodes=10) >= -6


def test_reinforce_policy_is_valid_distribution():
    env = GridWorld()
    agent = Reinforce(env.n_states, env.n_actions, seed=0)
    probs = agent.policy(0)
    assert probs.shape == (env.n_actions,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)
