"""Expected SARSA — TD control with the expected next-state value.

Instead of bootstrapping from a single sampled next action (SARSA) or the max
(Q-learning), it takes the expectation over the epsilon-greedy policy:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * sum_a' pi(a'|s') Q(s',a') - Q(s,a) ]

with pi the epsilon-greedy policy: each greedy action gets
(1 - eps)/n_greedy + eps/n_actions, others get eps/n_actions. This removes the
variance from sampling a', usually giving smoother learning than SARSA.
"""

from __future__ import annotations

import numpy as np

from rllab.agents.base import TabularAgent


class ExpectedSarsa(TabularAgent):
    on_policy = True  # bootstraps from the behaviour policy's expectation

    def _policy_probs(self, state: int) -> np.ndarray:
        q = self.Q[state]
        probs = np.full(self.n_actions, self.epsilon / self.n_actions)
        best = np.flatnonzero(q == q.max())
        probs[best] += (1.0 - self.epsilon) / len(best)
        return probs

    def td_target(self, reward, next_state, done, next_action=None):
        if done:
            return reward
        probs = self._policy_probs(next_state)
        expected_q = float(np.dot(probs, self.Q[next_state]))
        return reward + self.gamma * expected_q
