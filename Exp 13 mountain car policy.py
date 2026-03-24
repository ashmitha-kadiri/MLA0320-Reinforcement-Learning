import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create environment
env = gym.make("MountainCar-v0")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Build Neural Network Policy
model = keras.Sequential([
    layers.Dense(24, activation="relu", input_shape=(state_size,)),
    layers.Dense(24, activation="relu"),
    layers.Dense(action_size, activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy")

# Training parameters
episodes = 500
gamma = 0.99

for episode in range(episodes):

    state = env.reset()[0]
    states, actions, rewards = [], [], []
    done = False

    while not done:

        state_input = state.reshape([1, state_size])
        action_prob = model.predict(state_input, verbose=0)[0]
        action = np.random.choice(action_size, p=action_prob)

        next_state, reward, done, _, _ = env.step(action)

        states.append(state)
        action_onehot = np.zeros(action_size)
        action_onehot[action] = 1
        actions.append(action_onehot)
        rewards.append(reward)

        state = next_state

    # Compute discounted rewards
    discounted_rewards = []
    cumulative = 0
    for r in reversed(rewards):
        cumulative = r + gamma * cumulative
        discounted_rewards.insert(0, cumulative)

    discounted_rewards = np.array(discounted_rewards)
    discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

    states = np.array(states)
    actions = np.array(actions)

    model.fit(states, actions * discounted_rewards[:, None], verbose=0)

    if episode % 50 == 0:
        print("Episode:", episode)

print("Training Complete")

# Demonstration
state = env.reset()[0]
done = False

while not done:
    env.render()
    state_input = state.reshape([1, state_size])
    action = np.argmax(model.predict(state_input, verbose=0)[0])
    state, _, done, _, _ = env.step(action)

env.close()
