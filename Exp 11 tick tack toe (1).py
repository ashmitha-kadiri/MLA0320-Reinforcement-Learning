import numpy as np
import random
import math

# -------------------------------
# Tic Tac Toe Environment
# -------------------------------

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9

    def reset(self):
        self.board = [' '] * 9
        return tuple(self.board)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def step(self, action, player):
        self.board[action] = player
        winner = self.check_winner()
        done = winner is not None or ' ' not in self.board

        if winner == 'X':
            reward = 1
        elif winner == 'O':
            reward = -1
        else:
            reward = 0

        return tuple(self.board), reward, done

    def check_winner(self):
        combos = [(0,1,2),(3,4,5),(6,7,8),
                  (0,3,6),(1,4,7),(2,5,8),
                  (0,4,8),(2,4,6)]

        for a,b,c in combos:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        return None


# -------------------------------
# Q Learning Agent
# -------------------------------

class QAgent:

    def __init__(self, strategy="epsilon", epsilon=0.1, tau=1.0, alpha=0.1, gamma=0.9):
        self.q = {}
        self.strategy = strategy
        self.epsilon = epsilon
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.steps = 0

    def get_q(self, state, action):
        return self.q.get((state, action), 0)

    def choose_action(self, state, actions):

        self.steps += 1

        if self.strategy == "epsilon":

            if random.random() < self.epsilon:
                return random.choice(actions)

            q_values = [self.get_q(state,a) for a in actions]
            return actions[np.argmax(q_values)]

        if self.strategy == "softmax":

            q_values = np.array([self.get_q(state,a) for a in actions])
            exp_q = np.exp(q_values/self.tau)
            probs = exp_q/np.sum(exp_q)

            return np.random.choice(actions,p=probs)

    def update(self,state,action,reward,next_state,next_actions):

        max_next = max([self.get_q(next_state,a) for a in next_actions], default=0)

        old = self.get_q(state,action)

        self.q[(state,action)] = old + self.alpha*(reward + self.gamma*max_next - old)


# -------------------------------
# Training Function
# -------------------------------

def train(strategy, episodes=5000):

    env = TicTacToe()
    agent = QAgent(strategy=strategy)

    wins = 0

    for ep in range(episodes):

        state = env.reset()
        done = False

        while not done:

            actions = env.available_actions()
            action = agent.choose_action(state, actions)

            next_state, reward, done = env.step(action,'X')

            next_actions = env.available_actions()

            agent.update(state,action,reward,next_state,next_actions)

            state = next_state

            if done and reward == 1:
                wins += 1

    return wins, agent.steps


# -------------------------------
# Run Comparison
# -------------------------------

episodes = 5000

eps_wins, eps_steps = train("epsilon", episodes)
soft_wins, soft_steps = train("softmax", episodes)

print("\nResults after training")

print("Epsilon Greedy")
print("Wins:", eps_wins)
print("Steps:", eps_steps)

print("\nSoftmax Exploration")
print("Wins:", soft_wins)
print("Steps:", soft_steps)

print("\nWin Rate Comparison")

print("Epsilon Greedy Win Rate:", eps_wins/episodes)
print("Softmax Win Rate:", soft_wins/episodes)
