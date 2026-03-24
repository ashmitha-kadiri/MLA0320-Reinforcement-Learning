import numpy as np

# -----------------------------
# Inventory Parameters
# -----------------------------

max_inventory = 5
max_order = 5

holding_cost = 1
ordering_cost = 2
stockout_cost = 5

gamma = 0.9

# Demand probabilities
demand_prob = {
    0:0.1,
    1:0.2,
    2:0.3,
    3:0.2,
    4:0.2
}

states = range(max_inventory + 1)
actions = range(max_order + 1)

V = np.zeros(max_inventory + 1)
policy = np.zeros(max_inventory + 1)

# -----------------------------
# Value Iteration using Bellman Equation
# -----------------------------

for iteration in range(100):

    new_V = np.copy(V)

    for s in states:

        action_values = []

        for a in actions:

            if s + a > max_inventory:
                continue

            expected_cost = 0

            for d, prob in demand_prob.items():

                next_inventory = max(0, s + a - d)

                holding = holding_cost * next_inventory
                stockout = stockout_cost * max(0, d - (s + a))
                order_cost = ordering_cost * a

                cost = order_cost + holding + stockout

                expected_cost += prob * (cost + gamma * V[next_inventory])

            action_values.append(expected_cost)

        best_value = min(action_values)
        new_V[s] = best_value
        policy[s] = np.argmin(action_values)

    V = new_V

# -----------------------------
# Results
# -----------------------------

print("Optimal Value Function:\n")
print(np.round(V,2))

print("\nOptimal Ordering Policy (units to order):\n")
for s in states:
    print("Inventory:", s, " -> Order:", int(policy[s]))
