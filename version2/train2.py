import time
import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from dqn_agent2 import DQNAgent
from env import TrafficEnv



# Hyperparameters
STATE_SIZE = 16  # current phase + number of cars + waiting time
ACTION_SIZE = 4
HIDDEN_SIZE = 64
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0  # Initial epsilon
EPSILON_MIN = 0.01   # Minimum epsilon
EPSILON_DECAY_STEPS = 2000000  # Decay epsilon over 2,000,000 steps
BUFFER_CAPACITY = 10000
BATCH_SIZE = 128
NUM_EPISODES = 2000
MAX_STEPS = 2000
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

# Initialize tracking variables
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

        # Track exploration vs. exploitation
        if is_exploring:
            explore_count += 1
        else:
            exploit_count += 1
        if step >= MAX_STEPS:
            break

    # Update tracking variables
    episode_rewards.append(total_reward)
    explore_counts.append(explore_count)
    exploit_counts.append(exploit_count)
    epsilon_values.append(agent.epsilon)

    # Log progress
    print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}, "
          f"Explore: {explore_count}, Exploit: {exploit_count}")

    # Update target network periodically
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # Save model periodically
    if episode % SAVE_MODEL_EVERY == 0:
        agent.save_model()

# Create a directory to save the plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot performance metrics
plt.figure(figsize=(16, 12))

# Plot total reward per episode
plt.subplot(3, 3, 1)
plt.plot(episode_rewards, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.savefig("plots/total_reward_per_episode.png")  # Save the plot

# Plot exploration vs. exploitation
plt.subplot(3, 3, 2)
plt.plot(explore_counts, label="Exploration")
plt.plot(exploit_counts, label="Exploitation")
plt.xlabel("Episode")
plt.ylabel("Count")
plt.title("Exploration vs. Exploitation")
plt.legend()
plt.savefig("plots/exploration_vs_exploitation.png")  # Save the plot

# Plot epsilon decay
plt.subplot(3, 3, 3)
plt.plot(epsilon_values, label="Epsilon")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Over Time")
plt.legend()
plt.savefig("plots/epsilon_decay.png")  # Save the plot

# Plot moving average of rewards
window_size = 50
moving_avg = [np.mean(episode_rewards[max(0, i - window_size):i+1]) for i in range(len(episode_rewards))]
plt.subplot(3, 3, 4)
plt.plot(moving_avg, label="Moving Avg Reward")
plt.xlabel("Episode")
plt.ylabel("Moving Avg Reward")
plt.title(f"Moving Average Reward (Window Size={window_size})")
plt.legend()
plt.savefig("plots/moving_avg_reward.png")  # Save the plot

# Plot cumulative reward
plt.subplot(3, 3, 5)
cumulative_rewards = np.cumsum(episode_rewards)
plt.plot(cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Episodes")
plt.legend()
plt.savefig("plots/cumulative_reward.png")  # Save the plot

# Plot exploration vs. exploitation ratio
plt.subplot(3, 3, 6)
explore_ratio = [explore_counts[i] / (explore_counts[i] + exploit_counts[i]) for i in range(len(explore_counts))]
plt.plot(explore_ratio, label="Exploration Ratio")
plt.xlabel("Episode")
plt.ylabel("Exploration Ratio")
plt.title("Exploration vs. Exploitation Ratio")
plt.legend()
plt.savefig("plots/exploration_ratio.png")  # Save the plot


# Plot reward distribution
plt.subplot(3, 3, 8)
plt.hist(episode_rewards, bins=50, label="Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution")
plt.legend()
plt.savefig("plots/reward_distribution.png")  # Save the plot


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

# Run evaluation
evaluate_agent()

# Close environment
env.close()