import numpy as np
import random

# -----------------------------
# Environment Setup
# -----------------------------

# Content items (arms)
contents = ["Video", "Quiz", "Article", "Interactive"]

n_actions = len(contents)

# True engagement probabilities (unknown to agent)
true_rewards = [0.6, 0.4, 0.5, 0.7]

# Estimated values
Q = np.zeros(n_actions)

# Number of times each action selected
N = np.zeros(n_actions)

epsilon = 0.1
steps = 1000
total_reward = 0

# -----------------------------
# Epsilon-Greedy Simulation
# -----------------------------

for step in range(steps):

    # Exploration vs Exploitation
    if random.uniform(0,1) < epsilon:
        action = random.randint(0, n_actions-1)
    else:
        action = np.argmax(Q)

    # Simulated reward (user engagement)
    reward = 1 if random.random() < true_rewards[action] else 0

    total_reward += reward
    N[action] += 1

    # Update estimated value
    Q[action] = Q[action] + (1/N[action]) * (reward - Q[action])

# -----------------------------
# Results
# -----------------------------

print("Estimated Engagement Values:")
for i in range(n_actions):
    print(contents[i], ":", round(Q[i],3))

print("\nNumber of Recommendations:")
for i in range(n_actions):
    print(contents[i], ":", int(N[i]))

print("\nTotal Reward (Engagements):", total_reward)
