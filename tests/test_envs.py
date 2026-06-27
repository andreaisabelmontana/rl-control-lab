"""Environment dynamics tests."""

import numpy as np

from rllab.envs import CliffWalking, GridWorld, WindyGridWorld
from rllab.envs.cliffwalking import DOWN, LEFT, RIGHT, UP


def test_gridworld_step_cost_and_goal():
    env = GridWorld(rows=4, cols=4, walls=())
    s = env.reset()
    assert s == 0  # start top-left
    # one step right: cost -1, not done
    ns, r, done, _ = env.step(RIGHT)
    assert ns == 1
    assert r == -1.0
    assert not done


def test_gridworld_wall_and_edge_block():
    env = GridWorld(rows=4, cols=4, walls=((1, 1),))
    env.reset()
    # moving up from start (row 0) hits the top edge -> stay in place
    ns, r, done, _ = env.step(UP)
    assert ns == 0
    # walk to (0,1) then DOWN would enter wall (1,1) -> blocked, stay at (0,1)
    env.reset()
    env.step(RIGHT)  # now at (0,1) = index 1
    ns, r, done, _ = env.step(DOWN)
    assert env.index_to_pos(ns) == (0, 1)


def test_gridworld_reaches_goal_terminal():
    env = GridWorld(rows=2, cols=2, walls=(), start=(0, 0), goal=(1, 1))
    env.reset()
    env.step(RIGHT)  # (0,1)
    ns, r, done, _ = env.step(DOWN)  # (1,1) goal
    assert env.index_to_pos(ns) == (1, 1)
    assert done
    assert r == 0.0  # goal reward


def test_cliff_reward_and_reset():
    env = CliffWalking()
    start = env.reset()
    assert env.index_to_pos(start) == (3, 0)
    # step RIGHT from start steps onto a cliff cell (3,1)
    ns, r, done, info = env.step(RIGHT)
    assert r == -100.0
    assert info.get("fell") is True
    # teleported back to start, episode continues
    assert env.index_to_pos(ns) == (3, 0)
    assert not done


def test_cliff_goal_is_terminal_and_not_cliff():
    env = CliffWalking()
    env.reset()
    # the goal cell must not be part of the cliff
    assert env.goal not in env.cliff
    # walk up out of the cliff row, across the top, then down to the goal
    actions = [UP] + [RIGHT] * 11 + [DOWN]
    done = False
    last_r = None
    for a in actions:
        _, last_r, done, _ = env.step(a)
        if done:
            break
    assert done
    assert last_r == -1.0  # reaching goal costs a normal step, not the cliff


def test_cliff_edges_act_as_walls():
    env = CliffWalking()
    s = env.reset()
    # at start (3,0): moving LEFT hits the wall, stay put, normal -1 cost
    ns, r, done, info = env.step(LEFT)
    assert env.index_to_pos(ns) == (3, 0)
    assert r == -1.0
    assert not info.get("fell")


def test_windy_drift_pushes_up():
    env = WindyGridWorld()
    env.reset()
    # move RIGHT three times to reach a windy column, then test the push
    for _ in range(3):
        env.step(RIGHT)
    r0, c0 = env.index_to_pos(env._idx(env._pos))
    # column 3 has wind 1: a RIGHT move drifts the row up by one
    ns, r, done, _ = env.step(RIGHT)
    nr, nc = env.index_to_pos(ns)
    assert nr == r0 - 1  # pushed north by the wind
    assert r == -1.0


def test_all_envs_state_action_counts():
    for env, ns, na in [
        (GridWorld(), 16, 4),
        (CliffWalking(), 48, 4),
        (WindyGridWorld(), 70, 4),
    ]:
        assert env.n_states == ns
        assert env.n_actions == na
        s = env.reset()
        assert 0 <= s < env.n_states
