import numpy as np
import torch
from env import TrafficEnv
from dqn_agent2 import DQNAgent
import traci
import random
import os
import time
from sumolib import checkBinary

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42
VISUALIZE = True  # Set to False for faster testing
DELAY = 100  # ms delay between steps

class VisualTrafficEnv(TrafficEnv):
    def __init__(self):
        sumo_cmd = [
            checkBinary('sumo-gui' if VISUALIZE else 'sumo'),
            "-c", "sumo_files/sumo_config.sumocfg",
            "--step-length", "1",
            "--no-warnings"
        ]
        if VISUALIZE:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            
        traci.start(sumo_cmd)
        super().__init__()  # Properly initialize parent class

def run_episode(use_agent=True, seed=None, episode_num=0):
    """Run one test episode with optional visualization"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    env = VisualTrafficEnv()
    state = env.reset()
    total_reward = 0
    step = 0
    done = False
    
    current_phase = 0
    phase_duration = 30
    
    print(f"\nRunning Episode {episode_num} ({'Agent' if use_agent else 'Default'})")
    
    while not done and step < MAX_STEPS:
        if use_agent:
            action, _ = agent.act(state)
        else:
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 4
            action = current_phase
            
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
        
        if VISUALIZE:
            traci.gui.settle()
            time.sleep(DELAY/1000)
    
    env.close()
    return total_reward

# Initialize agent
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

# Load model
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    agent.q_network.eval()
    print(f"Model loaded from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Testing
print("\n=== Testing Started ===")
agent_rewards = []
default_rewards = []

for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode
    
    # Visualize first episode only
    visualize = VISUALIZE and (episode == 0)
    
    # Test agent
    agent_reward = run_episode(use_agent=True, seed=seed, episode_num=episode+1)
    agent_rewards.append(agent_reward)
    
    # Test default controller
    default_reward = run_episode(use_agent=False, seed=seed, episode_num=episode+1)
    default_rewards.append(default_reward)
    
    print(f"Episode {episode+1}/{NUM_TEST_EPISODES}")
    print(f"Agent Reward: {agent_reward:.1f} | Default Reward: {default_reward:.1f}")

# Results
avg_agent = np.mean(agent_rewards)
avg_default = np.mean(default_rewards)
improvement = (abs(avg_default - avg_agent) / abs(avg_default)) * 100

print("\n=== Final Results ===")
print(f"Agent Average Reward: {avg_agent:.1f}")
print(f"Default Average Reward: {avg_default:.1f}")
print(f"Improvement: {improvement:.1f}%")