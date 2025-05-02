import numpy as np
import torch
from env import TrafficEnv
from version8.ppo_agent import PPOAgent
import traci
import random
import os

# Configuration
MODEL_PATH = "ppo_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42

# Initialize environment
env = TrafficEnv()

# Initialize PPO agent with the SAME hyperparameters as used in training
agent = PPOAgent(
    state_dim=16,
    action_dim=12,
    hidden_dim=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    epochs=10,
    minibatch_size=256,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Load trained model
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"No model found at {MODEL_PATH}")

def run_episode(use_agent=True, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    state = env.reset()
    total_reward = 0
    step = 0
    done = False
    current_phase = 0
    phase_duration = 30

    while not done and step < MAX_STEPS:
        if use_agent:
            action, _, _ = agent.select_action(state, eval_mode=True)
        else:
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 12
            action = current_phase

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step += 1

    return total_reward

# Run tests
print("\n=== Testing Started ===")
agent_rewards = []
default_rewards = []

for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode
    agent_reward = run_episode(use_agent=True, seed=seed)
    default_reward = run_episode(use_agent=False, seed=seed)

    agent_rewards.append(agent_reward)
    default_rewards.append(default_reward)

    print(f"Episode {episode + 1}/{NUM_TEST_EPISODES}")
    print(f"Agent Reward: {agent_reward:.1f} | Default Reward: {default_reward:.1f}")

# Report results
avg_agent = np.mean(agent_rewards)
avg_default = np.mean(default_rewards)
improvement = (abs(avg_agent - avg_default) / abs(default_rewards[0] if default_rewards[0] != 0 else 1e-6)) * 100

print("\n=== Final Results ===")
print(f"Agent Average Reward: {avg_agent:.1f}")
print(f"Default Average Reward: {avg_default:.1f}")
print(f"Improvement: {improvement:.1f}%")

env.close()
