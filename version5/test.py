import numpy as np
import torch
from env import TrafficEnv
from dqn_agent2 import DQNAgent
import traci
import random
import os
import csv 
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


CSV_LOG_PATH = "plots/test_results.csv"

# Create/open the CSV file and write headers
with open(CSV_LOG_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Agent_Reward", "Default_Reward"])

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42

# Initialize environment
env = TrafficEnv()

# Initialize agent with EXACT parameters from dqn_agent2.py
agent = DQNAgent(
    state_size=16,                # Must match your env's state size
    action_size=4,                # Must match your env's action size
    hidden_size=64,               # Default in your agent
    lr=1e-3,                      # Learning rate
    gamma=0.99,                   # Discount factor
    epsilon_start=0.01,           # Starting epsilon (low for testing)
    epsilon_min=0.01,             # Minimum epsilon
    epsilon_decay_steps=2000000,  # Large value to prevent decay during testing
    buffer_capacity=10000,        # From your agent
    batch_size=64,                # From your agent
    tau=0.01,                     # Target network update factor
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Load the trained model
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    agent.q_network.eval()  # Set to evaluation mode
    print(f"Successfully loaded model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"No model found at {MODEL_PATH}")

# Disable exploration during testing
agent.epsilon = 0.0 # Minimal exploration (can set to 0 if you want pure exploitation)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_q_values(q_logs, save_as="q_animation.gif"):
    fig, ax = plt.subplots()
    # Define action names corresponding to the numbers
    action_map = {0: 'Red', 1: 'Green', 2: 'Yellow', 3: 'No Change'}
    
    actions = ['Red', 'Green', 'Yellow', 'No Change']
    bar_container = ax.bar(actions, [0]*4, color='grey')
    ax.set_ylim(-5, 5)  # Adjust according to your Q-value range
    ax.set_ylabel("Q-value")
    ax.set_title("Q-values Over Time (Chosen in Red)")

    def update(frame):
        step, q_values, chosen = q_logs[frame]
        for i, bar in enumerate(bar_container):
            bar.set_height(q_values[i])
            bar.set_color('red' if i == chosen else 'grey')
        ax.set_title(f"Step {step} | Chosen Action: {action_map[chosen]}")
        return bar_container

    anim = FuncAnimation(fig, update, frames=len(q_logs), blit=False, interval=50)
    anim.save(save_as, writer='pillow')
    plt.close("all")


def run_episode(use_agent=True, seed=None ,collect_q=False):
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
    
    actions_taken = []  # to plot actions
    q_logs = []  # Stores (step, q_values, chosen_action)

    while not done and step < MAX_STEPS:
        if use_agent:
            
            action, _, q_values = agent.act(state, return_q=True)

            if collect_q:
                q_logs.append((step, q_values, action))
        else:
            # Default controller: cycle through phases
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 4
            action = current_phase
            
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        step += 1
        actions_taken.append(action)
    if collect_q:
        return total_reward, actions_taken, q_logs
    else:
         return total_reward, actions_taken, []


# Testing
print("\n=== Testing Started ===")
agent_rewards = []
default_rewards = []

for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode
    
    # Test agent
    agent_reward,agent_actions ,q_logs = run_episode(use_agent=True, seed=seed,collect_q=True)
    agent_rewards.append(agent_reward)
    animate_q_values(q_logs, save_as="plots/agent_q_animation.gif")
    # Test default controller
    default_reward, default_actions, _ = run_episode(use_agent=False, seed=seed)
    default_rewards.append(default_reward)
    
    #Append to CSV log
    with open(CSV_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode + 1, agent_reward, default_reward])

    print(f"Episode {episode+1}/10")
    print(f"Agent Reward: {agent_reward:.1f} | Default Reward: {default_reward:.1f}")





# Results
avg_agent = np.mean(agent_rewards)
avg_default = np.mean(default_rewards)
improvement = (abs(avg_default - avg_agent) / abs(avg_default)) * 100
#improvements = ((avg_agent - avg_default) / abs(avg_default) if avg_default != 0 else 0) * 100  #it's relative to the average, not just the first episode


print("\n=== Final Results ===")
print(f"Agent Average Reward: {avg_agent:.1f}")
print(f"Default Average Reward: {avg_default:.1f}")
print(f"Improvement: {improvement:.1f}%")
# print(f"Improvements: {improvements:.1f}%")

action_map = {1: 'Red', 2: 'Green', 3: 'Yellow', 4: 'No Change'}

# Plot actions for the last episode
plt.figure(figsize=(12, 4))
plt.plot(agent_actions, drawstyle='steps-post', label="Agent Actions")
plt.xlabel("Step")
plt.ylabel("Action")
plt.title("Agent Actions Over Time (Episode {})".format(NUM_TEST_EPISODES))
plt.yticks([1, 2, 3, 4], [action_map[1], action_map[2], action_map[3], action_map[4]])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/agent_actions_episode_{}.png".format(NUM_TEST_EPISODES))
plt.show()

# Action 1 → Red phase

# Action 2 → Green phase

# Action 3 → Yellow phase

# Action 4 → No Change (Neutral)

env.close()