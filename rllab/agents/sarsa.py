"""SARSA — on-policy TD control.

Update rule (uses the action a' actually chosen by the policy at s'):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

Because the backup follows the exploratory behaviour policy, SARSA learns the
value of the policy it is actually running — leading to safer paths when
exploration is dangerous (e.g. the cliff).
"""

from __future__ import annotations

from rllab.agents.base import TabularAgent


class Sarsa(TabularAgent):
    on_policy = True

    def td_target(self, reward, next_state, done, next_action=None):
        if done:
            return reward
        assert next_action is not None, "SARSA needs the next action (on-policy)"
        return reward + self.gamma * self.Q[next_state, next_action]
