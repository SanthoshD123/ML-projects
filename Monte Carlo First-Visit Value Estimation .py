import numpy as np
import random
from collections import defaultdict

# Define environment
grid_size = 4
actions = ['up', 'down', 'left', 'right']
terminal_states = [(0, 0), (3, 3)]

def step(state, action):
    i, j = state
    if state in terminal_states:
        return state, 0

    if action == 'up':
        i = max(i - 1, 0)
    elif action == 'down':
        i = min(i + 1, grid_size - 1)
    elif action == 'left':
        j = max(j - 1, 0)
    elif action == 'right':
        j = min(j + 1, grid_size - 1)

    new_state = (i, j)
    reward = 0 if new_state in terminal_states else -1
    return new_state, reward

# Generate an episode
def generate_episode(policy):
    state = random.choice([(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) not in terminal_states])
    episode = []
    while True:
        action = policy[state]
        next_state, reward = step(state, action)
        episode.append((state, action, reward))
        if next_state in terminal_states:
            break
        state = next_state
    return episode

# Initialize
V = defaultdict(float)           # State-value estimates
returns = defaultdict(list)      # For storing returns
policy = {}                      # Random policy

for i in range(grid_size):
    for j in range(grid_size):
        policy[(i, j)] = random.choice(actions)

# Monte Carlo First-Visit Prediction
num_episodes = 5000
gamma = 1.0  # Discount factor

for episode in range(num_episodes):
    ep = generate_episode(policy)
    G = 0
    visited = set()

    for t in reversed(range(len(ep))):
        state, action, reward = ep[t]
        G = gamma * G + reward
        if state not in visited:
            visited.add(state)
            returns[state].append(G)
            V[state] = np.mean(returns[state])

# Display estimated values
for i in range(grid_size):
    for j in range(grid_size):
        print(f"{V[(i,j)]:6.2f}", end=" ")
    print()
