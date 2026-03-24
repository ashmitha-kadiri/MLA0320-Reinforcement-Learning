import numpy as np
import math
import random

# -----------------------------
# Environment Setup
# -----------------------------

contents = ["Movie", "Series", "Documentary", "Short Video"]
n_actions = len(contents)

# True user engagement probabilities (unknown to algorithm)
true_rewards = [0.5, 0.7, 0.4, 0.6]

steps = 1000

# Track selections and rewards
N = np.zeros(n_actions)   # number of times each content selected
Q = np.zeros(n_actions)   # estimated reward
total_reward = 0

# -----------------------------
# UCB Algorithm
# -----------------------------

for t in range(1, steps + 1):

    ucb_values = np.zeros(n_actions)

    for a in range(n_actions):
        if N[a] == 0:
            ucb_values[a] = float('inf')
        else:
            ucb_values[a] = Q[a] + math.sqrt((2 * math.log(t)) / N[a])

    action = np.argmax(ucb_values)

    # Simulated user engagement reward
    reward = 1 if random.random() < true_rewards[action] else 0

    N[action] += 1
    total_reward += reward

    # Update estimated reward
    Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])

# -----------------------------
# Results
# -----------------------------

print("Estimated Engagement Values:\n")
for i in range(n_actions):
    print(contents[i], ":", round(Q[i],3))

print("\nNumber of Recommendations:")
for i in range(n_actions):
    print(contents[i], ":", int(N[i]))

print("\nTotal User Engagement:", total_reward)
