import numpy as np
import matplotlib.pyplot as plt

# Grid size
ROWS, COLS = 5, 5
GAMMA = 0.9
THETA = 1e-4
STEP_COST = -1
DELIVERY_REWARD = 100

# Delivery points
delivery_points = [(4, 4), (0, 4)]

# Actions
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

# Check valid state
def is_valid(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS

# Get next state
def get_next_state(state, action):
    dr, dc = actions[action]
    next_state = (state[0] + dr, state[1] + dc)
    if is_valid(next_state):
        return next_state
    return state


# Policy Evaluation using Bellman Expectation Equation
def policy_evaluation(policy):
    V = np.zeros((ROWS, COLS))
    
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for r in range(ROWS):
            for c in range(COLS):
                
                if (r, c) in delivery_points:
                    new_V[r, c] = DELIVERY_REWARD
                    continue
                
                action_probs = policy[(r, c)]
                value = 0
                
                for action, prob in action_probs.items():
                    next_state = get_next_state((r, c), action)
                    reward = STEP_COST
                    value += prob * (reward + GAMMA * V[next_state])
                
                new_V[r, c] = value
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        
        V = new_V
        
        if delta < THETA:
            break
            
    return V


# -------------------------
# Define Different Policies
# -------------------------

# 1️⃣ Always Right
policy_right = {
    (r, c): {"R": 1.0} for r in range(ROWS) for c in range(COLS)
}

# 2️⃣ Always Down
policy_down = {
    (r, c): {"D": 1.0} for r in range(ROWS) for c in range(COLS)
}

# 3️⃣ Random Policy
policy_random = {
    (r, c): {a: 0.25 for a in actions} for r in range(ROWS) for c in range(COLS)
}

# Evaluate
V_right = policy_evaluation(policy_right)
V_down = policy_evaluation(policy_down)
V_random = policy_evaluation(policy_random)


# -------------------------
# Visualization
# -------------------------
def plot_value_function(V, title):
    plt.figure()
    plt.imshow(V, cmap='viridis')
    plt.colorbar(label="State Value")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()

plot_value_function(V_right, "Value Function - Always Right Policy")
plot_value_function(V_down, "Value Function - Always Down Policy")
plot_value_function(V_random, "Value Function - Random Policy")
