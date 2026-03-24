import numpy as np

# -----------------------------
# Gridworld Setup
# -----------------------------

grid_size = 4
gamma = 0.9

# Rewards grid
rewards = np.array([
    [0, 0, 0, 10],
    [0, -1, 0, -10],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Actions: Up, Down, Left, Right
actions = [(-1,0),(1,0),(0,-1),(0,1)]

# -----------------------------
# Policy (Random Policy)
# -----------------------------

def random_policy():
    return np.ones((grid_size, grid_size, len(actions))) / len(actions)


# -----------------------------
# Value Function Calculation
# -----------------------------

def policy_evaluation(policy, iterations=100):

    V = np.zeros((grid_size, grid_size))

    for _ in range(iterations):

        new_V = np.copy(V)

        for i in range(grid_size):
            for j in range(grid_size):

                value = 0

                for a, action in enumerate(actions):

                    ni = i + action[0]
                    nj = j + action[1]

                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        reward = rewards[ni][nj]
                        value += policy[i][j][a] * (reward + gamma * V[ni][nj])

                new_V[i][j] = value

        V = new_V

    return V


# -----------------------------
# Run Evaluation
# -----------------------------

policy = random_policy()
V = policy_evaluation(policy)

print("Value Function:\n")
print(np.round(V,2))
