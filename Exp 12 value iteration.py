import numpy as np

# Grid size
rows, cols = 5, 5

# Reward grid
rewards = np.full((rows, cols), -1)
goal = (4, 4)
rewards[goal] = 10

# Discount factor
gamma = 0.9

# Value function
V = np.zeros((rows, cols))

# Possible actions
actions = [(-1,0),(1,0),(0,-1),(0,1)]  # up, down, left, right

# Value Iteration
for _ in range(100):
    new_V = V.copy()
    for r in range(rows):
        for c in range(cols):
            values = []
            for a in actions:
                nr = min(max(r + a[0], 0), rows-1)
                nc = min(max(c + a[1], 0), cols-1)
                values.append(rewards[nr][nc] + gamma * V[nr][nc])
            new_V[r][c] = max(values)
    V = new_V

print("Optimal State Value Function:\n")
print(V)

# Deriving optimal policy path
state = (0,0)
path = [state]

while state != goal:
    r,c = state
    best_action = None
    best_value = -999

    for a in actions:
        nr = min(max(r + a[0], 0), rows-1)
        nc = min(max(c + a[1], 0), cols-1)

        value = rewards[nr][nc] + gamma * V[nr][nc]

        if value > best_value:
            best_value = value
            best_action = (nr,nc)

    state = best_action
    path.append(state)

print("\nOptimal Path from Start to Goal:")
print(path)
