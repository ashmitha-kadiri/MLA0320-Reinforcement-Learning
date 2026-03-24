import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Hyperparameters
EPISODES = 500
TIME_STEPS = 50
ALPHA = 0.01
GAMMA = 0.99

# Market parameters
MU = 0.01
SIGMA = 0.05

# Policy parameter (mean allocation)
theta = 0.5
policy_std = 0.1

returns_history = []


def run_episode(theta):
    wealth = 1.0
    episode = []

    for t in range(TIME_STEPS):
        # Sample allocation from policy
        action = np.random.normal(theta, policy_std)
        action = np.clip(action, 0, 1)

        # Simulate market return
        market_return = np.random.normal(MU, SIGMA)

        # Update wealth
        new_wealth = wealth * (1 + action * market_return)

        reward = np.log(new_wealth / wealth)

        episode.append((action, reward))
        wealth = new_wealth

    return episode


for episode in range(EPISODES):
    episode_data = run_episode(theta)

    # Compute returns (Monte Carlo return)
    G = 0
    gradients = []

    for action, reward in reversed(episode_data):
        G = reward + GAMMA * G

        # Gradient of log policy (Gaussian)
        grad_log = (action - theta) / (policy_std ** 2)

        gradients.append(grad_log * G)

    # Update theta
    theta += ALPHA * np.sum(gradients)

    returns_history.append(G)


print("Optimized Investment Fraction (theta):", theta)

plt.plot(returns_history)
plt.title("Policy Gradient Training")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.show()
