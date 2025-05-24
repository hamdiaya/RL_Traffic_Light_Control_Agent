import os
import random
import numpy as np
import torch
from env import TrafficEnv
from dqn_agent import DQNAgent
import traci

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42

# Environment and Device
env = TrafficEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent Config (must match training)
agent = DQNAgent(
    state_size=17,
    action_size=12,
    hidden_size=128,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=0.01,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    epsilon_decay_steps=1e6,
    buffer_capacity=200000,
    batch_size=256,
    tau=0.005,
    update_every=4,
    device=device,
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

# Force no exploration for evaluation
agent.epsilon = 0.0

def run_episode(use_agent=True, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    state = env.reset()
    total_reward = 0
    done = False
    step = 0

    current_phase = 0
    phase_duration = 10  # for round-robin

    while not done and step < MAX_STEPS:
        if False:
            action, _ = agent.act(state)
        else:
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 12
            action = current_phase

        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        step += 1

    return total_reward

# === Testing ===
print("\n🚦 === Testing Started ===")
agent_rewards = []
default_rewards = []

print(f"Running on device: {next(agent.q_network.parameters()).device}")

for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode

    # agent_reward = run_episode(use_agent=True, seed=seed)
    default_reward = run_episode(use_agent=False, seed=seed)

    # agent_rewards.append(agent_reward)
    default_rewards.append(default_reward)

    print(f"🎮 Episode {episode+1}")
    #print(f"Agent Reward: {agent_reward:.2f} | Default Reward: {default_reward:.2f}")

# Results
# avg_agent = np.mean(agent_rewards)
avg_default = np.mean(default_rewards)
# improvement = ((avg_agent - avg_default) / abs(avg_default)) * 100 if avg_default != 0 else float('inf')

# print("\n📊 === Final Results ===")
# print(f"Agent Average Reward:   {avg_agent:.2f}")
# print(f"Default Average Reward: {avg_default:.2f}")
# print(f"Improvement:            {improvement:.2f}%")

env.close()
