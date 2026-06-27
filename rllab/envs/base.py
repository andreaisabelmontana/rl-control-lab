"""Shared interface for tabular environments.

Every environment exposes a Gym-style API implemented from scratch:

    state = env.reset()
    next_state, reward, done, info = env.step(action)

States are integers in [0, n_states). Actions are integers in [0, n_actions).
This keeps the agents environment-agnostic: any tabular agent can run on any
environment without touching the algorithm code.
"""

from __future__ import annotations

import random


class TabularEnv:
    """Base class for discrete-state / discrete-action environments."""

    n_states: int
    n_actions: int

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def reset(self) -> int:
        """Return the initial state (an integer index)."""
        raise NotImplementedError

    def step(self, action: int):
        """Apply ``action``; return ``(next_state, reward, done, info)``."""
        raise NotImplementedError

    # -- helpers used by planning / rendering -----------------------------

    @property
    def n_states_actions(self) -> tuple[int, int]:
        return self.n_states, self.n_actions
