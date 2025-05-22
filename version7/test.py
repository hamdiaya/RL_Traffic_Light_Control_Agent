import numpy as np
import torch
import random
import os
from env import TrafficEnv
from dqn_agent import DQNAgent  # Updated import for the new agent
import traci

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42

# Initialize environment
env = TrafficEnv()

# Initialize agent with same config as during training
agent = DQNAgent(
    state_size=16,
    action_size=12,
    hidden_size=128,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=0.01,       # low to disable exploration
    epsilon_min=0.01,
    epsilon_decay=0.9995,     # won't matter for testing
    epsilon_decay_steps=900000,
    buffer_capacity=100000,
    batch_size=128,
    tau=0.005,
    update_every=4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_double=True,
    use_dueling=True,
    use_per=True
)

# Load trained model
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    agent.q_network.eval()
    print(f"✅ Successfully loaded model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"❌ No model found at {MODEL_PATH}")

# Disable exploration during testing
agent.epsilon = 0.0

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
    phase_duration = 30  # only used for default controller

    while not done and step < MAX_STEPS:
        if use_agent:
            action, _ = agent.act(state)
        else:
            # Default controller: naive round-robin through 12 phases
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 12
            action = current_phase

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step += 1

    return total_reward

# Run tests
print("\n🚦 === Testing Started ===")
agent_rewards = []
default_rewards = []
print(next(agent.q_network.parameters()).device)
for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode
    
    agent_reward = run_episode(use_agent=True, seed=seed)
    default_reward = run_episode(use_agent=False, seed=seed)
    
    agent_rewards.append(agent_reward)
    default_rewards.append(default_reward)

    print(f"🎮 Episode {episode+1}")
    print(f"Agent Reward: {agent_reward:.1f} | Default Reward: {default_reward:.1f}")

# Results
avg_agent = np.mean(agent_rewards)
avg_default = np.mean(default_rewards)
improvement = (avg_agent - avg_default) / abs(avg_default) * 100 if avg_default != 0 else float('inf')

print("\n📊 === Final Results ===")
print(f"Agent Average Reward:   {avg_agent:.1f}")
print(f"Default Average Reward: {avg_default:.1f}")
print(f"Improvement:            {improvement:.1f}%")

env.close()
