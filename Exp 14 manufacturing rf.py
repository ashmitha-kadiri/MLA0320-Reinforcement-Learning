import numpy as np
import random

# -----------------------------
# Manufacturing Environment
# -----------------------------

class ManufacturingEnv:

    def __init__(self):
        # Machine settings (actions)
        self.actions = [0,1,2]   # Low, Medium, High
        self.state = random.randint(0,2)  # Process state

    def reset(self):
        self.state = random.randint(0,2)
        return self.state

    def step(self, action):

        # Reward based on product quality
        if self.state == action:
            reward = 10     # optimal quality
        else:
            reward = -2     # poor quality

        next_state = random.randint(0,2)
        done = False

        self.state = next_state
        return next_state, reward, done


# -----------------------------
# RL Agent (Q-Learning)
# -----------------------------

states = 3
actions = 3

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

env = ManufacturingEnv()

episodes = 1000

for episode in range(episodes):

    state = env.reset()

    for step in range(20):

        # Epsilon-Greedy Policy
        if random.uniform(0,1) < epsilon:
            action = random.randint(0, actions-1)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = env.step(action)

        # Value Function Update
        Q[state,action] = Q[state,action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state,action]
        )

        state = next_state


print("Learned Q-Table:\n")
print(Q)


# -----------------------------
# Optimal Policy
# -----------------------------

policy = np.argmax(Q, axis=1)

print("\nOptimal Machine Settings for Each State:")
print(policy)
