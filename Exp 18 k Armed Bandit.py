import numpy as np
import random
import math

# -----------------------------
# Environment (Marketing Campaigns)
# -----------------------------

campaigns = ["Email", "Social Media", "TV Ads", "Influencer", "Search Ads"]
k = len(campaigns)

# True conversion probabilities (unknown to algorithms)
true_prob = [0.12, 0.18, 0.10, 0.15, 0.20]

steps = 1000
epsilon = 0.1


# -----------------------------
# Epsilon-Greedy Algorithm
# -----------------------------

def epsilon_greedy():
    Q = np.zeros(k)
    N = np.zeros(k)
    total_reward = 0

    for t in range(steps):

        if random.random() < epsilon:
            action = random.randint(0, k-1)
        else:
            action = np.argmax(Q)

        reward = 1 if random.random() < true_prob[action] else 0

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        total_reward += reward

    return total_reward


# -----------------------------
# UCB Algorithm
# -----------------------------

def ucb():
    Q = np.zeros(k)
    N = np.zeros(k)
    total_reward = 0

    for t in range(1, steps+1):

        ucb_values = np.zeros(k)

        for a in range(k):
            if N[a] == 0:
                ucb_values[a] = float('inf')
            else:
                ucb_values[a] = Q[a] + math.sqrt((2 * math.log(t)) / N[a])

        action = np.argmax(ucb_values)

        reward = 1 if random.random() < true_prob[action] else 0

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        total_reward += reward

    return total_reward


# -----------------------------
# Thompson Sampling
# -----------------------------

def thompson_sampling():
    success = np.ones(k)
    failure = np.ones(k)
    total_reward = 0

    for t in range(steps):

        samples = [np.random.beta(success[i], failure[i]) for i in range(k)]
        action = np.argmax(samples)

        reward = 1 if random.random() < true_prob[action] else 0

        if reward == 1:
            success[action] += 1
        else:
            failure[action] += 1

        total_reward += reward

    return total_reward


# -----------------------------
# Run Algorithms
# -----------------------------

eg_reward = epsilon_greedy()
ucb_reward = ucb()
ts_reward = thompson_sampling()

print("Total Conversions:\n")
print("Epsilon-Greedy:", eg_reward)
print("UCB:", ucb_reward)
print("Thompson Sampling:", ts_reward)
