"""Windy Grid World (Sutton & Barto, Example 6.5).

A 7x10 grid. An upward "wind" pushes the agent north by a column-dependent
amount on every move, in addition to the chosen action. Start and goal sit on
the middle row.

Wind strength per column (default): [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

This is the "continuous-ish control" task discretized onto a grid: the wind
acts like an external drift the agent must compensate for, the way a controller
fights a disturbance. Every step costs -1 until the goal is reached.
"""

from __future__ import annotations

from rllab.envs.base import TabularEnv

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

_DEFAULT_WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)


class WindyGridWorld(TabularEnv):
    def __init__(
        self,
        rows: int = 7,
        cols: int = 10,
        wind: tuple[int, ...] = _DEFAULT_WIND,
        start: tuple[int, int] = (3, 0),
        goal: tuple[int, int] = (3, 7),
        step_reward: float = -1.0,
        max_steps: int = 1000,
        seed: int | None = None,
    ):
        super().__init__(seed)
        assert len(wind) == cols, "wind must have one entry per column"
        self.rows = rows
        self.cols = cols
        self.wind = wind
        self.start = start
        self.goal = goal
        self.step_reward = step_reward
        self.max_steps = max_steps

        self.n_states = rows * cols
        self.n_actions = 4
        self._pos = start
        self._t = 0

    def _idx(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def index_to_pos(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.cols)

    def _clamp(self, r: int, c: int) -> tuple[int, int]:
        r = min(max(r, 0), self.rows - 1)
        c = min(max(c, 0), self.cols - 1)
        return r, c

    def reset(self) -> int:
        self._pos = self.start
        self._t = 0
        return self._idx(self._pos)

    def step(self, action: int):
        self._t += 1
        r, c = self._pos
        dr, dc = _DELTAS[action]
        # wind for the *current* column pushes up (negative row)
        w = self.wind[c]
        nr, nc = self._clamp(r + dr - w, c + dc)
        self._pos = (nr, nc)

        if self._pos == self.goal:
            return self._idx(self._pos), self.step_reward, True, {}

        done = self._t >= self.max_steps
        return self._idx(self._pos), self.step_reward, done, {"truncated": done}
