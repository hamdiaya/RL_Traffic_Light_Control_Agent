import os
import time
import logging
import numpy as np
import torch
from matplotlib import pyplot as plt
from env import TrafficEnv
import traci
from version8.ppo_agent import PPOAgent

# PPO Hyperparameters
STATE_SIZE = 16
ACTION_SIZE = 12
HIDDEN_SIZE = 128
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
BATCH_SIZE = 2048
MINI_BATCH_SIZE = 256
UPDATE_EPOCHS = 10

NUM_EPISODES = 5000
MAX_STEPS = 7200
SAVE_MODEL_EVERY = 50
EVAL_EPISODES = 20


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def initialize_environment():
    env = TrafficEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return env, device

def create_agent(device):
    return PPOAgent(
        state_dim=STATE_SIZE,
        action_dim=ACTION_SIZE,
        hidden_dim=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPS,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        epochs=UPDATE_EPOCHS,
        minibatch_size=MINI_BATCH_SIZE,
        device=device
    )

def load_or_initialize_model(agent):
    try:
        agent.load_model("ppo_model.pth")
        logging.info("Loaded existing model.")
    except FileNotFoundError:
        logging.info("No saved model found, training from scratch.")

def train_agent(env, agent):
    metrics = {
        'episode_rewards': [],
        'losses': []
    }

    for episode in range(NUM_EPISODES):
        state = env.reset()
        env.current_difficulty = min((episode / (NUM_EPISODES * 0.66)) ** 2, 1.0)

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        ep_reward = 0
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state
            ep_reward += reward
            step += 1

        loss = agent.update(states, actions, rewards, dones, log_probs, values)

        metrics['episode_rewards'].append(ep_reward)
        metrics['losses'].append(loss)

        logging.info(
            f"Episode {episode + 1}/{NUM_EPISODES}, "
            f"Reward: {ep_reward:.2f}, "
            f"Loss: {loss:.4f}"
        )

        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            agent.save_model("ppo_model.pth")
            logging.info(f"Model saved at episode {episode + 1}")

    return metrics

def plot_metrics(metrics):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    window_size = 50
    moving_avg = [
        np.mean(metrics['episode_rewards'][max(0, i - window_size):i + 1])
        for i in range(len(metrics['episode_rewards']))
    ]
    plt.plot(moving_avg, label=f"Moving Avg (Window={window_size})")
    plt.title("Moving Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(metrics['losses'])
    plt.title("Loss Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.hist(metrics['episode_rewards'], bins=50)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("plots/ppo_training_metrics.png")
    plt.close()

def evaluate_agent(env, agent):
    logging.info("Starting evaluation...")
    total_rewards = []

    for _ in range(EVAL_EPISODES):
        state = env.reset()
        env.current_difficulty = 1.0
        total_reward = 0
        done = False
        while not done:
            action, _, _ = agent.select_action(state, eval_mode=True)
            state, reward, done = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    avg_reward = sum(total_rewards) / len(total_rewards)
    logging.info(f"Evaluation Complete - Average Reward: {avg_reward:.1f}")

def main():
    setup_logging()
    env, device = initialize_environment()
    agent = create_agent(device)
    load_or_initialize_model(agent)

    try:
        metrics = train_agent(env, agent)
        plot_metrics(metrics)
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
