# Results (real runs, mean of 5 seeds)

## CliffWalking (4x12)

| Algorithm | Greedy return | Greedy path | Gap to cliff |
|---|---:|---:|---:|
| Q-learning | -13.00 +/- 0.00 | 13 steps | 1 row(s) |
| SARSA | -17.00 +/- 0.00 | 17 steps | 2 row(s) |
| Expected-SARSA | -15.00 +/- 0.00 | 15 steps | 2 row(s) |

## GridWorld (5x5 for TD, 4x4 for REINFORCE)

| Algorithm | Greedy return |
|---|---:|
| Q-learning | -7.00 +/- 0.00 |
| SARSA | -7.00 +/- 0.00 |
| Expected-SARSA | -7.00 +/- 0.00 |
| REINFORCE | -5.00 +/- 0.00 |
