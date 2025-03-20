import time
import logging
import torch
from dqn_agent2 import DQNAgent
from env import TrafficEnv

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Hyperparameters
STATE_SIZE = 16  # current phase + number of cars + waiting time
ACTION_SIZE = 4
HIDDEN_SIZE = 64
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0  # Initial epsilon
EPSILON_MIN = 0.01   # Minimum epsilon
EPSILON_DECAY_STEPS = 1000  # Number of steps to decay epsilon linearly
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
NUM_EPISODES = 1000
MAX_STEPS = 1000
TARGET_UPDATE_FREQ = 10
EVAL_EPISODES = 20
SAVE_MODEL_EVERY = 50

# Initialize environment and agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = TrafficEnv()
agent = DQNAgent(
    STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LR, GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY_STEPS,
    BUFFER_CAPACITY, BATCH_SIZE, device=device
)

# Load previous model if available
try:
    agent.load_model()
    print("Loaded existing model.")
except FileNotFoundError:
    print("No saved model found, training from scratch.")

# Training loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    explore_count = 0
    exploit_count = 0

    while not done:
        action, is_exploring = agent.act(state)
        next_state, reward, done = env.step(action)
        print(next_state, action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward
        step += 1

        if is_exploring:
            explore_count += 1
        else:
            exploit_count += 1

        if step >= MAX_STEPS:
            break

    # Logging and printing episode results
    logging.info(f"Episode {episode + 1}: Reward={total_reward}, Explores={explore_count}, Exploits={exploit_count}, Epsilon={agent.epsilon:.4f}")
    print(f"Episode {episode + 1}: Reward={total_reward}, Explores={explore_count}, Exploits={exploit_count}, Epsilon={agent.epsilon:.4f}")

    # Update target network
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # Save model periodically
    if (episode + 1) % SAVE_MODEL_EVERY == 0:
        agent.save_model()
        print("Model saved.")

# Evaluation mode
def evaluate_agent():
    agent.epsilon = 0.0  # No exploration during evaluation
    total_rewards = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    print(f"Evaluation: Avg Reward: {sum(total_rewards) / len(total_rewards)}")

# Run evaluation
evaluate_agent()

# Close environment
env.close()