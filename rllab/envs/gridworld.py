"""A small deterministic grid world.

Layout (default 4x4), start at top-left, goal at bottom-right:

    S . . .
    . # . .
    . . . .
    . . . G

The agent moves up/right/down/left. Hitting a wall (#) or the grid edge
leaves the agent in place. Every step costs -1; reaching the goal gives 0 and
ends the episode. The optimal policy is therefore the shortest path to G.
"""

from __future__ import annotations

from rllab.envs.base import TabularEnv

# action indices
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class GridWorld(TabularEnv):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        walls: tuple[tuple[int, int], ...] = ((1, 1),),
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__(seed)
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal if goal is not None else (rows - 1, cols - 1)
        self.walls = set(walls)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.n_states = rows * cols
        self.n_actions = 4
        self._pos = start
        self._t = 0

    # -- coordinate <-> index ------------------------------------------------

    def _idx(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def index_to_pos(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.cols)

    # -- API -----------------------------------------------------------------

    def reset(self) -> int:
        self._pos = self.start
        self._t = 0
        return self._idx(self._pos)

    def step(self, action: int):
        self._t += 1
        dr, dc = _DELTAS[action]
        r, c = self._pos
        nr, nc = r + dr, c + dc

        # blocked by edge or wall -> stay put
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            self._pos = (nr, nc)

        if self._pos == self.goal:
            return self._idx(self._pos), self.goal_reward, True, {}

        done = self._t >= self.max_steps
        return self._idx(self._pos), self.step_reward, done, {"truncated": done}
