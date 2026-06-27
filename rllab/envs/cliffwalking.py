"""Cliff Walking (Sutton & Barto, Example 6.6).

A 4x12 grid. Start is bottom-left, goal is bottom-right. The whole bottom row
between them is a cliff.

    . . . . . . . . . . . .
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    S C C C C C C C C C C G

- Every step costs -1.
- Stepping into a cliff cell costs -100 and teleports the agent back to the
  start (the episode continues; it is NOT terminal).
- Reaching the goal ends the episode.

The classic result: Q-learning learns the optimal (risky) path right along the
cliff edge, while SARSA learns a safer path one row up, because on-policy
backups account for the exploratory steps that occasionally fall in.
"""

from __future__ import annotations

from rllab.envs.base import TabularEnv

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class CliffWalking(TabularEnv):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        step_reward: float = -1.0,
        cliff_reward: float = -100.0,
        max_steps: int = 1000,
        seed: int | None = None,
    ):
        super().__init__(seed)
        self.rows = rows
        self.cols = cols
        self.step_reward = step_reward
        self.cliff_reward = cliff_reward
        self.max_steps = max_steps

        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        # cliff = bottom row, columns 1..cols-2
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}

        self.n_states = rows * cols
        self.n_actions = 4
        self._pos = self.start
        self._t = 0

    def _idx(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def index_to_pos(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.cols)

    def is_cliff(self, pos: tuple[int, int]) -> bool:
        return pos in self.cliff

    def reset(self) -> int:
        self._pos = self.start
        self._t = 0
        return self._idx(self._pos)

    def step(self, action: int):
        self._t += 1
        dr, dc = _DELTAS[action]
        r, c = self._pos
        nr, nc = r + dr, c + dc

        # clamp to grid (edges act as walls)
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c
        self._pos = (nr, nc)

        if self._pos in self.cliff:
            # fall off: big penalty, reset to start, episode continues
            self._pos = self.start
            done = self._t >= self.max_steps
            return self._idx(self._pos), self.cliff_reward, done, {"fell": True}

        if self._pos == self.goal:
            return self._idx(self._pos), self.step_reward, True, {}

        done = self._t >= self.max_steps
        return self._idx(self._pos), self.step_reward, done, {"truncated": done}
