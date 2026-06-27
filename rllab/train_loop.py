"""Shared training loops.

`train_td` drives the value-based agents (Q-learning, SARSA, Expected-SARSA);
`train_reinforce` drives the Monte-Carlo policy-gradient agent. Both return the
per-episode total reward so learning curves are logged identically.
"""

from __future__ import annotations

import numpy as np


def train_td(env, agent, episodes: int) -> np.ndarray:
    """Train an on/off-policy TD agent. Returns array of episode returns."""
    returns = np.empty(episodes, dtype=np.float64)

    for ep in range(episodes):
        state = env.reset()
        action = agent.act(state)
        total = 0.0
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            total += reward

            if agent.on_policy and not done:
                next_action = agent.act(next_state)
                agent.update(state, action, reward, next_state, done, next_action)
                state, action = next_state, next_action
            else:
                agent.update(state, action, reward, next_state, done)
                state = next_state
                if not done:
                    action = agent.act(state)

        returns[ep] = total

    return returns


def train_reinforce(env, agent, episodes: int) -> np.ndarray:
    """Train a REINFORCE agent. Returns array of episode returns."""
    returns = np.empty(episodes, dtype=np.float64)

    for ep in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        agent.update_episode(states, actions, rewards)
        returns[ep] = float(sum(rewards))

    return returns


def evaluate_greedy(env, agent, episodes: int = 20) -> float:
    """Average return of the agent's greedy policy (no exploration)."""
    totals = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            action = agent.greedy(state)
            state, reward, done, _ = env.step(action)
            total += reward
            steps += 1
        totals.append(total)
    return float(np.mean(totals))


def greedy_path(env, agent, max_steps: int = 200):
    """Return the list of (row, col) cells visited under the greedy policy."""
    state = env.reset()
    path = [env.index_to_pos(state)]
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.greedy(state)
        state, _, done, _ = env.step(action)
        path.append(env.index_to_pos(state))
        steps += 1
    return path
