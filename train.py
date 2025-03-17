import time
import logging

from agent import DQNAgent
from env import TrafficEnv

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Hyperparameters
STATE_SIZE = 13  # Size of the state vector (from TrafficEnv.get_state())
ACTION_SIZE = 4  # Number of actions (4 traffic light phases)
HIDDEN_SIZE = 64
LR = 1e-3
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
NUM_EPISODES = 1000
MAX_STEPS = 1000  # 1-hour simulation

# Initialize environment and agent
env = TrafficEnv()
agent = DQNAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LR, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, BUFFER_CAPACITY, BATCH_SIZE)

# Training loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    # Counters for exploration and exploitation
    explore_count = 0
    exploit_count = 0

    while not done:
        action, is_exploring = agent.act(state)  # Modify act() to return exploration status
        next_state, reward, done = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward
        step += 1

        # Update exploration/exploitation counters
        if is_exploring:
            explore_count += 1
        else:
            exploit_count += 1

        if step >= MAX_STEPS:
            break

    # Log episode results
    logging.info(f"Episode: {episode + 1}, Total Reward: {total_reward}, Explores: {explore_count}, Exploits: {exploit_count}, Epsilon: {agent.epsilon:.4f}")

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Explores: {explore_count}, Exploits: {exploit_count}, Epsilon: {agent.epsilon:.4f}")

    # Update target network every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.update_target_network()

# Close the environment
env.close()
