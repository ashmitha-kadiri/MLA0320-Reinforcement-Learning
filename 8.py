import numpy as np
import random

# Grid size (road network)
ROWS, COLS = 5, 5

# Start and destination
start = (0, 0)
destination = (4, 4)

# Rewards
STEP_COST = -1
DEST_REWARD = 100
RED_LIGHT_PENALTY = -20

# Traffic light toggle probability
def get_traffic_light():
    return random.choice(["GREEN", "RED"])


# Greedy movement toward destination
def greedy_action(state):
    x, y = state
    dx = destination[0] - x
    dy = destination[1] - y
    
    if dx > 0:
        return (x+1, y)
    elif dy > 0:
        return (x, y+1)
    return state


# Safe policy (obeys traffic light)
def safe_action(state, light):
    if light == "RED":
        return state  # wait
    return greedy_action(state)


# Simulation function
def simulate(policy_type, episodes=100):
    total_rewards = []
    
    for _ in range(episodes):
        state = start
        total_reward = 0
        steps = 0
        
        while state != destination and steps < 50:
            light = get_traffic_light()
            
            if policy_type == "greedy":
                next_state = greedy_action(state)
                if light == "RED":
                    total_reward += RED_LIGHT_PENALTY
            else:
                next_state = safe_action(state, light)
            
            if next_state != state:
                total_reward += STEP_COST
            
            state = next_state
            steps += 1
        
        if state == destination:
            total_reward += DEST_REWARD
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)


# Run comparison
greedy_score = simulate("greedy")
safe_score = simulate("safe")

print("Average Reward over 100 episodes:")
print("Greedy Policy:", greedy_score)
print("Safe Rule-Following Policy:", safe_score)
