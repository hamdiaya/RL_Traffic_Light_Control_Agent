import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import TrafficEnv
from agent import DQNAgent

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize environment
env = TrafficEnv()

# Get state and action sizes from the environment
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

print(f"State Size: {STATE_SIZE}, Action Size: {ACTION_SIZE}")

# Hyperparameters
HIDDEN_SIZE = 64
LR = 1e-3
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
NUM_EPISODES = 1000
MAX_STEPS = 1000
TARGET_UPDATE_FREQ = 10
EVAL_EPISODES = 20
SAVE_MODEL_EVERY = 50

# Initialize agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE, LR, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, BUFFER_CAPACITY, BATCH_SIZE, device=device)

# Load previous model if available
try:
    agent.load_model()
    print("Loaded existing model.")
except FileNotFoundError:
    print("No saved model found, training from scratch.")

# Lists to store metrics
episode_rewards = []
explore_counts = []
exploit_counts = []
epsilon_values = []

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

    # Log metrics
    episode_rewards.append(total_reward)
    explore_counts.append(explore_count)
    exploit_counts.append(exploit_count)
    epsilon_values.append(agent.epsilon)

    logging.info(f"Episode {episode+1}: Reward={total_reward}, Explores={explore_count}, Exploits={exploit_count}, Epsilon={agent.epsilon:.4f}")
    print(f"Episode {episode+1}: Reward={total_reward}, Explores={explore_count}, Exploits={exploit_count}, Epsilon={agent.epsilon:.4f}")

    # Update target network
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # Save model
    if (episode + 1) % SAVE_MODEL_EVERY == 0:
        agent.save_model()
        print("Model saved.")

# Plot performance metrics
plt.figure(figsize=(12, 8))

# Plot total reward per episode
plt.subplot(2, 2, 1)
plt.plot(episode_rewards, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()

# Plot exploration vs. exploitation
plt.subplot(2, 2, 2)
plt.plot(explore_counts, label="Exploration")
plt.plot(exploit_counts, label="Exploitation")
plt.xlabel("Episode")
plt.ylabel("Count")
plt.title("Exploration vs. Exploitation")
plt.legend()

# Plot epsilon decay
plt.subplot(2, 2, 3)
plt.plot(epsilon_values, label="Epsilon")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Over Time")
plt.legend()

# Plot moving average of rewards (to smooth the curve)
window_size = 50
moving_avg = [np.mean(episode_rewards[max(0, i - window_size):i+1]) for i in range(len(episode_rewards))]
plt.subplot(2, 2, 4)
plt.plot(moving_avg, label="Moving Avg Reward")
plt.xlabel("Episode")
plt.ylabel("Moving Avg Reward")
plt.title(f"Moving Average Reward (Window Size={window_size})")
plt.legend()

plt.tight_layout()
plt.show()

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

evaluate_agent()

# Close environment
env.close()