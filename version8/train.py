import os
import time
import logging
import numpy as np
import torch
from matplotlib import pyplot as plt
from env import TrafficEnv
import traci
from ppo_agent import PPOAgent

# Hyperparameters
STATE_SIZE = 16
ACTION_SIZE = 12
HIDDEN_SIZE = 128
LR = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
BATCH_SIZE = 64
NUM_EPISODES = 2000  # Fewer episodes due to PPO’s efficiency
MAX_STEPS = 7200
EVAL_EPISODES = 20
SAVE_MODEL_EVERY = 50

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_ppo.log'),
            logging.StreamHandler()
        ]
    )

def initialize_environment():
    env = TrafficEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return env, device

def create_agent(device):
    return PPOAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        clip_epsilon=CLIP_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        device=device
    )

def load_or_initialize_model(agent):
    try:
        agent.load_model("ppo_model.pth")
        logging.info("Loaded existing PPO model.")
    except FileNotFoundError:
        logging.info("No saved PPO model found, training from scratch.")

def train_agent(env, agent):
    metrics = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'avg_queue_lengths': []
    }

    for episode in range(NUM_EPISODES):
        curriculum_progress = min((episode / (NUM_EPISODES * 0.66)) ** 2, 1.0)
        state = env.reset()
        env.current_difficulty = curriculum_progress

        total_reward = 0
        waiting_times = []
        queue_lengths = []
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            action, _ = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done, 0)  # Log_prob not used in training loop
            loss = agent.update()

            waiting_time = sum(traci.edge.getWaitingTime(e) for e in env.incoming_edges)
            queue_length = sum(traci.edge.getLastStepHaltingNumber(e) for e in env.incoming_edges)
            waiting_times.append(waiting_time)
            queue_lengths.append(queue_length)

            state = next_state
            total_reward += reward
            step += 1

        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
        avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0

        metrics['episode_rewards'].append(total_reward)
        metrics['avg_waiting_times'].append(avg_waiting_time)
        metrics['avg_queue_lengths'].append(avg_queue_length)

        logging.info(
            f"Episode {episode + 1}/{NUM_EPISODES}, "
            f"Difficulty: {curriculum_progress:.2f}, "
            f"Reward: {total_reward:.1f}, "
            f"Avg Waiting Time: {avg_waiting_time:.1f}s, "
            f"Avg Queue Length: {avg_queue_length:.1f} vehicles"
        )

        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            agent.save_model("ppo_model.pth")
            logging.info(f"PPO model saved at episode {episode + 1}")

    return metrics

def plot_metrics(metrics, agent):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(metrics['episode_rewards'], label='Total Reward')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(metrics['avg_waiting_times'], label='Avg Waiting Time', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time (s)")
    plt.title("Average Waiting Time per Episode")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(metrics['avg_queue_lengths'], label='Avg Queue Length', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Queue Length (vehicles)")
    plt.title("Average Queue Length per Episode")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/ppo_training_metrics.png")
    plt.close()

def evaluate_agent(env, agent):
    logging.info("Starting PPO evaluation...")
    total_rewards = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        env.current_difficulty = 1.0
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.act(state, eval_mode=True)
            state, reward, done = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / len(total_rewards)
    logging.info(f"PPO Evaluation Complete - Average Reward: {avg_reward:.1f}")

def main():
    setup_logging()
    env, device = initialize_environment()
    agent = create_agent(device)
    load_or_initialize_model(agent)
    try:
        metrics = train_agent(env, agent)
        plot_metrics(metrics, agent)
        evaluate_agent(env, agent)
    except Exception as e:
        logging.error(f"Training failed: {e}")
    finally:
        try:
            env.close()
        except Exception as e:
            logging.error(f"Error closing environment: {e}")
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    main()