import time
import logging

from matplotlib import pyplot as plt
import numpy as np
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

# Initialize tracking variables with additional metrics
episode_rewards = []
explore_counts = []
exploit_counts = []
epsilon_values = []
episode_times = []
step_counts = []
avg_waiting_times = []
avg_queue_lengths = []

for episode in range(NUM_EPISODES):
    start_time = time.time()
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    explore_count = 0
    exploit_count = 0
    waiting_times = []
    queue_lengths = []

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

        # Collect performance metrics
        #waiting_times.append(env.get_waiting_time())  # Implement this method in your TrafficEnv
        #queue_lengths.append(env.get_queue_length())  # Implement this method in your TrafficEnv

        if step >= MAX_STEPS:
            break

    # Update tracking variables
    episode_time = time.time() - start_time
    episode_rewards.append(total_reward)
    explore_counts.append(explore_count)
    exploit_counts.append(exploit_count)
    epsilon_values.append(agent.epsilon)
    episode_times.append(episode_time)
    step_counts.append(step)
    avg_waiting_times.append(np.mean(waiting_times))
    avg_queue_lengths.append(np.mean(queue_lengths))

    # Log detailed metrics
    logging.info(
        f"Episode {episode + 1}/{NUM_EPISODES} | "
        f"Total Reward: {total_reward:.2f} | "
        f"Explore: {explore_count} | "
        f"Exploit: {exploit_count} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Avg Wait: {np.mean(waiting_times):.2f}s | "
        f"Avg Queue: {np.mean(queue_lengths):.2f} | "
        f"Time: {episode_time:.2f}s"
    )

    # Update target network periodically
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # Save model and metrics periodically
    if episode % SAVE_MODEL_EVERY == 0:
        agent.save_model()
        # Save metrics checkpoint
        np.savez("training_metrics.npz",
                 rewards=episode_rewards,
                 explore_counts=explore_counts,
                 exploit_counts=exploit_counts,
                 epsilon_values=epsilon_values,
                 waiting_times=avg_waiting_times,
                 queue_lengths=avg_queue_lengths)

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

# Run evaluation
evaluate_agent()

# Close environment
env.close()