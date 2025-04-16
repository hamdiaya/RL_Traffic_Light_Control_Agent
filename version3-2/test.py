import numpy as np
import torch
from env import TrafficEnv
from dqn_agent2 import DQNAgent  # Keep original agent class
import traci
import random
import os

# --- Add Dueling DQN network definition here ---
class QNetwork(torch.nn.Module):
    """Dueling DQN architecture (overrides original)"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU()
        )
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        x = self.feature(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# --- Monkey-patch the agent to use our dueling network ---
DQNAgent.q_network = QNetwork  # Replace original Q-network
DQNAgent.target_network = QNetwork  # Replace target network

# --- Rest of your test script remains unchanged ---
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42

# Initialize environment
env = TrafficEnv()

# Agent initialization (now uses Dueling DQN)
agent = DQNAgent(
    state_size=16,
    action_size=4,
    hidden_size=64,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=0.01,
    epsilon_min=0.01,
    epsilon_decay_steps=2000000,
    buffer_capacity=10000,
    batch_size=64,
    tau=0.01,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Load model (must be trained with dueling architecture!)
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    agent.q_network.eval()
    print(f"Loaded Dueling DQN from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model {MODEL_PATH} not found")

# Disable exploration during testing
agent.epsilon = 0.00  # Minimal exploration (can set to 0 if you want pure exploitation)

def run_episode(use_agent=True, seed=None):
    """Run one test episode"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    state = env.reset()
    total_reward = 0
    step = 0
    done = False
    
    # Default controller variables
    current_phase = 0
    phase_duration = 30
    
    while not done and step < MAX_STEPS:
        if use_agent:
            action, _ = agent.act(state)
        else:
            # Default controller: cycle through phases
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 4
            action = current_phase
            
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
    
    return total_reward

# Testing
print("\n=== Testing Started ===")
agent_rewards = []
default_rewards = []

for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode
    
    # Test agent
    agent_reward = run_episode(use_agent=True, seed=seed)
    agent_rewards.append(agent_reward)
    
    # Test default controller
    default_reward = run_episode(use_agent=False, seed=seed)
    default_rewards.append(default_reward)
    
    print(f"Episode {episode+1}/10")
    print(f"Agent Reward: {agent_reward:.1f} | Default Reward: {default_reward:.1f}")

# Results
avg_agent = np.mean(agent_rewards)
avg_default = np.mean(default_rewards)
improvement = (abs(avg_default - avg_agent) / abs(avg_default)) * 100

print("\n=== Final Results ===")
print(f"Agent Average Reward: {avg_agent:.1f}")
print(f"Default Average Reward: {avg_default:.1f}")
print(f"Improvement: {improvement:.1f}%")

env.close()