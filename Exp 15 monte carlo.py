import numpy as np
import random

# -----------------------------
# Customer Churn Environment
# -----------------------------

states = ["Active", "AtRisk", "Churned"]

# Policy: action taken by company
actions = ["NoOffer", "Discount"]

# Transition probabilities
transition = {
    "Active": {"Active":0.7, "AtRisk":0.3},
    "AtRisk": {"Active":0.4, "Churned":0.6},
    "Churned": {"Churned":1.0}
}

# Rewards
rewards = {
    "Active": 10,
    "AtRisk": 2,
    "Churned": -10
}

# -----------------------------
# Monte Carlo Policy Evaluation
# -----------------------------

V = {s:0 for s in states}
returns = {s:[] for s in states}

episodes = 1000
gamma = 0.9

for episode in range(episodes):

    state = "Active"
    episode_states = []
    episode_rewards = []

    for t in range(10):

        episode_states.append(state)
        reward = rewards[state]
        episode_rewards.append(reward)

        next_state = random.choices(
            list(transition[state].keys()),
            list(transition[state].values())
        )[0]

        state = next_state

        if state == "Churned":
            episode_states.append(state)
            episode_rewards.append(rewards[state])
            break

    # Calculate returns
    G = 0
    for t in reversed(range(len(episode_states))):
        G = gamma * G + episode_rewards[t]
        s = episode_states[t]

        returns[s].append(G)
        V[s] = np.mean(returns[s])

# -----------------------------
# Results
# -----------------------------

print("State Value Estimates:\n")

for s in states:
    print(s, ":", round(V[s],2))
