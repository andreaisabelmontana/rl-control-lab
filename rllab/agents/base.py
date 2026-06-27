"""Shared machinery for the tabular value-based agents.

Q-learning, SARSA and Expected-SARSA differ only in how they compute the TD
target. Everything else — the Q-table, epsilon-greedy action selection, the
update step size — lives here so the agents stay a few lines each.
"""

from __future__ import annotations

import numpy as np


class TabularAgent:
    """Base class for epsilon-greedy, Q-table agents."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    # -- policy --------------------------------------------------------------

    def act(self, state: int) -> int:
        """Epsilon-greedy action selection (ties broken at random)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(state)

    def greedy(self, state: int) -> int:
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    # -- learning ------------------------------------------------------------

    def td_target(self, reward, next_state, done, next_action=None):
        """Return the TD target. Overridden by each algorithm."""
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done, next_action=None):
        target = self.td_target(reward, next_state, done, next_action)
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return td_error

    # on-policy agents (SARSA family) need the next action chosen up front
    on_policy: bool = False
