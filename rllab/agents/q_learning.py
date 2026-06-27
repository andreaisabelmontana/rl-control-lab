"""Q-learning (Watkins, 1989) — off-policy TD control.

Update rule:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The target uses the greedy action at s' regardless of which action the
behaviour policy actually takes next, which is what makes it off-policy.
"""

from __future__ import annotations

from rllab.agents.base import TabularAgent


class QLearning(TabularAgent):
    on_policy = False

    def td_target(self, reward, next_state, done, next_action=None):
        if done:
            return reward
        return reward + self.gamma * self.Q[next_state].max()
