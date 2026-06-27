"""REINFORCE — Monte-Carlo policy gradient (Williams, 1992).

A softmax policy over a linear preference. For tabular states we use one-hot
features, so the parameters theta have shape (n_states, n_actions) and the
preference for action a in state s is just theta[s, a]:

    pi(a|s) = softmax_a( theta[s, a] )

After each episode we compute the return from every step and nudge theta along
the score function, scaled by the (optionally baseline-corrected) return:

    G_t  = sum_{k>=t} gamma^{k-t} r_k
    grad log pi(a_t|s_t) = onehot(a_t) - pi(.|s_t)
    theta <- theta + alpha * (G_t - b) * grad log pi(a_t|s_t)
             + beta * grad H(pi(.|s_t))

The baseline b is the running mean of returns; it reduces gradient variance
without biasing the estimate. The optional entropy bonus (coefficient beta)
pushes the policy towards higher entropy, which keeps exploration alive.

A note on the gamma^t factor: the strictly-unbiased gradient weights step t by
gamma^t. In practice that starves the early states (including the start state)
of gradient and lets their softmax collapse onto a wall-bumping action before
the policy ever reaches the goal — a real failure mode under sparse rewards. So
the default here drops it (the common variant); pass discount_gradient=True for
the textbook-exact form. This is the only policy-based method in the lab: no
value table is learned, the policy parameters are optimised directly.
"""

from __future__ import annotations

import numpy as np


class Reinforce:
    on_policy = True

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        use_baseline: bool = True,
        entropy_beta: float = 0.0,
        discount_gradient: bool = False,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.entropy_beta = entropy_beta
        self.discount_gradient = discount_gradient
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros((n_states, n_actions), dtype=np.float64)
        self._baseline = 0.0
        self._baseline_count = 0

    # -- policy --------------------------------------------------------------

    def policy(self, state: int) -> np.ndarray:
        prefs = self.theta[state]
        prefs = prefs - prefs.max()  # numerical stability
        exp = np.exp(prefs)
        return exp / exp.sum()

    def act(self, state: int) -> int:
        probs = self.policy(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def greedy(self, state: int) -> int:
        return int(np.argmax(self.theta[state]))

    # -- learning ------------------------------------------------------------

    def update_episode(self, states, actions, rewards) -> None:
        """Apply the REINFORCE update for one completed episode."""
        T = len(rewards)
        # discounted returns G_t computed backwards
        returns = np.empty(T, dtype=np.float64)
        g = 0.0
        for t in reversed(range(T)):
            g = rewards[t] + self.gamma * g
            returns[t] = g

        for t in range(T):
            s, a = states[t], actions[t]
            g_t = returns[t]
            baseline = self._baseline if self.use_baseline else 0.0
            advantage = g_t - baseline

            probs = self.policy(s)
            grad_log = -probs
            grad_log[a] += 1.0  # onehot(a) - pi(.|s)

            # The gamma^t weighting is theoretically exact but in practice
            # starves early states of gradient (and lets the start state's
            # softmax collapse). The common, robust variant drops it.
            discount = (self.gamma ** t) if self.discount_gradient else 1.0
            step = self.alpha * discount * advantage * grad_log

            if self.entropy_beta > 0.0:
                # d/dtheta_i  H(pi) = -pi_i * (log pi_i - sum_j pi_j log pi_j)
                logp = np.log(probs + 1e-12)
                grad_entropy = -probs * (logp - float(np.dot(probs, logp)))
                step = step + self.entropy_beta * grad_entropy

            self.theta[s] += step

        if self.use_baseline:
            # running mean of episode returns G_0 as a simple baseline
            self._baseline_count += 1
            self._baseline += (returns[0] - self._baseline) / self._baseline_count
