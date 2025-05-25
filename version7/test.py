import numpy as np
import torch
import pandas as pd
from env import TrafficEnv
from dqn_agent import DQNAgent
import traci
import random
import os

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 50
MAX_STEPS = 7200
FIXED_SEED = 42
MAX_WAITING = 3600  # From TrafficEnv.get_state for denormalization
MAX_QUEUE = 200     # From TrafficEnv.get_state for queue denormalization
MAX_THROUGHPUT = 60 # From TrafficEnv.calculate_reward

# Initialize environment
env = TrafficEnv()

# Initialize agent with parameters matching training script
agent = DQNAgent(
    state_size=16,
    action_size=12,  # 4 phases x 3 durations
    hidden_size=128,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=0.01,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    buffer_capacity=100000,
    batch_size=128,
    tau=0.005,
    update_every=4,
    epsilon_decay_steps=900000,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_double=True,
    use_dueling=True,
    use_per=True
)

# Load the trained model
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    agent.q_network.eval()
    print(f"Successfully loaded model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"No model found at {MODEL_PATH}")

# Disable exploration during testing
agent.epsilon = 0.0

def run_episode(use_agent=True, seed=None):
    """Run one test episode, returning reward, avg waiting time, avg queue length, and throughput"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    state = env.reset()
    env.current_difficulty = 1.0  # Full difficulty for testing
    total_reward = 0
    wait_total_list = []
    queue_total_list = []
 
    step = 0
    done = False
    
    # Default controller variables
    current_phase = 0
    phase_duration = 30
    
    while not done and step < MAX_STEPS:
        if use_agent:
            action, _ = agent.act(state, eval_mode=True)
        else:
            # Default controller: round-robin with fixed 30s duration
            if step % phase_duration == 0:
                current_phase = (current_phase + 1) % 4
            action = current_phase  # Maps to GREEN_PHASES[current_phase]
            
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Extract metrics from current state
        wait_times = state[8:12] * MAX_WAITING
        queue_lengths = state[12:16] * MAX_QUEUE
        wait_total_list.append(np.mean(wait_times))
        queue_total_list.append(np.mean(queue_lengths))
        
      
        
        state = next_state
        step += 1
    
    return (
        total_reward,
        np.mean(wait_total_list) if wait_total_list else 0,
        np.mean(queue_total_list) if queue_total_list else 0,
       
    )

# Testing
print("\n=== Testing Started ===")
episode_metrics = []

for episode in range(NUM_TEST_EPISODES):
    seed = FIXED_SEED + episode
    
    # Test agent
    agent_reward, agent_wait, agent_queue = run_episode(use_agent=True, seed=seed)
    
    # Test default controller
    default_reward, default_wait, default_queue = run_episode(use_agent=False, seed=seed)
    
    episode_metrics.append({
        'episode': episode + 1,
        'agent_reward': agent_reward,
        'default_reward': default_reward,
        'agent_wait_total': agent_wait,
        'default_wait_total': default_wait,
        'agent_queue_total': agent_queue,
        'default_queue_total': default_queue,

    })
    
    print(f"Episode {episode+1}/{NUM_TEST_EPISODES}")
    print(f"Agent Reward: {agent_reward:.1f} | Default Reward: {default_reward:.1f}")
    print(f"Agent Avg Waiting Time (All Lanes): {agent_wait:.1f} seconds | Default Avg Waiting Time (All Lanes): {default_wait:.1f} seconds")
    print(f"Agent Avg Queue Length (All Lanes): {agent_queue:.1f} vehicles | Default Avg Queue Length (All Lanes): {default_queue:.1f} vehicles")
    
# Aggregate results
avg_metrics = {
    'agent_reward': np.mean([m['agent_reward'] for m in episode_metrics]),
    'default_reward': np.mean([m['default_reward'] for m in episode_metrics]),
    'agent_wait_total': np.mean([m['agent_wait_total'] for m in episode_metrics]),
    'default_wait_total': np.mean([m['default_wait_total'] for m in episode_metrics]),
    'agent_queue_total': np.mean([m['agent_queue_total'] for m in episode_metrics]),
    'default_queue_total': np.mean([m['default_queue_total'] for m in episode_metrics]),

} 

# Calculate improvements
improvement_reward = ((avg_metrics['agent_reward'] - avg_metrics['default_reward']) / abs(avg_metrics['default_reward'])) * 100 if avg_metrics['default_reward'] != 0 else 0
improvement_wait = ((avg_metrics['default_wait_total'] - avg_metrics['agent_wait_total']) / avg_metrics['default_wait_total']) * 100 if avg_metrics['default_wait_total'] != 0 else 0
improvement_queue = ((avg_metrics['default_queue_total'] - avg_metrics['agent_queue_total']) / avg_metrics['default_queue_total']) * 100 if avg_metrics['default_queue_total'] != 0 else 0

print("\n=== Final Results ===")
print(f"Agent Average Reward: {avg_metrics['agent_reward']:.1f}")
print(f"Default Average Reward: {avg_metrics['default_reward']:.1f}")
print(f"Reward Improvement: {improvement_reward:.1f}%")
print(f"Agent Average Waiting Time (All Lanes): {avg_metrics['agent_wait_total']:.1f} seconds")
print(f"Default Average Waiting Time (All Lanes): {avg_metrics['default_wait_total']:.1f} seconds")
print(f"Waiting Time Improvement: {improvement_wait:.1f}%")
print(f"Agent Average Queue Length (All Lanes): {avg_metrics['agent_queue_total']:.1f} vehicles")
print(f"Default Average Queue Length (All Lanes): {avg_metrics['default_queue_total']:.1f} vehicles")
print(f"Queue Length Improvement: {improvement_queue:.1f}%")


# Save to CSV
df = pd.DataFrame(episode_metrics)
df.to_csv('test_results.csv', index=False)
print("\nResults saved to 'test_results.csv'")

env.close()