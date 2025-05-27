import numpy as np
import torch
import pandas as pd
from env import TrafficEnv
from dqn_agent import DQNAgent
import traci
import random
import os
import logging
import time
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_test.log'),
        logging.StreamHandler()
    ]
)

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 50
MAX_STEPS = 7200
FIXED_SEED = 42
MAX_WAITING = 600  # From env.py MAX_WAIT=600
MAX_QUEUE = 20     # From env.py MAX_Q=50
OUTPUT_CSV = "agent_test_results.csv"
PLOT_PATH = "agent_metrics.png"

def initialize_environment():
    """Initialize SUMO environment"""
    try:
        time.sleep(1)
        env = TrafficEnv()
        logging.info("Environment initialized successfully")
        return env
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        raise

def initialize_agent():
    """Initialize DQN agent with parameters matching train.py"""
    try:
        agent = DQNAgent(
            state_size=17,
            action_size=32,
            hidden_size=128,
            lr=0.001,
            gamma=0.99,
            epsilon_start=0.01,
            epsilon_min=0.01,
            epsilon_decay=0.9995,
            buffer_capacity=200000,
            batch_size=256,
            tau=0.005,
            update_every=4,
            epsilon_decay_steps=2000000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            use_double=True,
            use_dueling=True,
            use_per=True
        )
        logging.info("Agent initialized successfully")
        return agent
    except Exception as e:
        logging.error(f"Failed to initialize agent: {e}")
        raise

def load_model(agent):
    """Load trained model"""
    if os.path.exists(MODEL_PATH):
        try:
            agent.load_model(MODEL_PATH)
            agent.q_network.eval()
            agent.epsilon = 0.0
            logging.info(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    else:
        logging.error(f"No model found at {MODEL_PATH}")
        raise FileNotFoundError(f"No model found at {MODEL_PATH}")

def run_episode(env, agent, seed=None):
    """Run one test episode for the agent, returning metrics from state"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    try:
        state = env.reset()
        if not isinstance(state, (list, np.ndarray)) or len(state) != 17:
            raise ValueError(f"Invalid state: expected length 17, got {type(state)} of length {len(state)}")
        state = np.array(state)
        
        total_reward = 0
        wait_total_list = []
        queue_total_list = []
        step = 0
        done = False
        
        while not done and step < MAX_STEPS:
            action, _ = agent.act(state, eval_mode=True)
            
            next_state, reward, done = env.step(action)
            if not isinstance(next_state, (list, np.ndarray)) or len(next_state) != 17:
                raise ValueError(f"Invalid next_state: expected length 17, got {type(next_state)} of length {len(next_state)}")
            next_state = np.array(next_state)
            
            total_reward += reward
            
            queue_lengths = next_state[[0, 3, 6, 9]] * MAX_QUEUE
            wait_times = next_state[[1, 4, 7, 10]] * MAX_WAITING
            
            wait_total_list.append(np.mean(wait_times))
            queue_total_list.append(np.mean(queue_lengths))
            
            state = next_state
            step += 1
        
        return (
            total_reward,
            np.mean(wait_total_list) if wait_total_list else 0,
            np.mean(queue_total_list) if queue_total_list else 0
        )
    except Exception as e:
        logging.error(f"Episode failed: {e}")
        raise

def plot_metrics(metrics):
    """Plot agent metrics"""
    episodes = [m['episode'] for m in metrics]
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(episodes, [m['reward'] for m in metrics], label='Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Reward')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(episodes, [m['wait_total'] for m in metrics], label='Wait Total', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Waiting Time (s)')
    plt.title('Agent Waiting Time')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(episodes, [m['queue_total'] for m in metrics], label='Queue Total', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Queue Length (vehicles)')
    plt.title('Agent Queue Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    logging.info(f"Plots saved to {PLOT_PATH}")

def main():
    """Main testing pipeline for agent only"""
    env = None
    try:
        env = initialize_environment()
        agent = initialize_agent()
        load_model(agent)
        
        logging.info("=== Agent Testing Started ===")
        episode_metrics = []
        
        for episode in range(NUM_TEST_EPISODES):
            seed = FIXED_SEED + episode * 5
            
            reward, wait_total, queue_total = run_episode(env, agent, seed=seed)
            
            episode_metrics.append({
                'episode': episode + 1,
                'reward': reward,
                'wait_total': wait_total,
                'queue_total': queue_total
            })
            
            logging.info(f"Episode {episode+1}/{NUM_TEST_EPISODES}")
            logging.info(f"Agent Reward: {reward:.1f}")
            logging.info(f"Agent Avg Waiting Time: {wait_total:.1f} seconds")
            logging.info(f"Agent Avg Queue Length: {queue_total:.1f} vehicles")
        
        # Aggregate results
        avg_metrics = {
            'reward': np.mean([m['reward'] for m in episode_metrics]),
            'wait_total': np.mean([m['wait_total'] for m in episode_metrics]),
            'queue_total': np.mean([m['queue_total'] for m in episode_metrics])
        }
        
        logging.info("\n=== Final Agent Results ===")
        logging.info(f"Average Reward: {avg_metrics['reward']:.1f}")
        logging.info(f"Average Waiting Time: {avg_metrics['wait_total']:.1f} seconds")
        logging.info(f"Average Queue Length: {avg_metrics['queue_total']:.1f} vehicles")
        
        # Save to CSV
        try:
            df = pd.DataFrame(episode_metrics)
            df.to_csv(OUTPUT_CSV, index=False)
            logging.info(f"Results saved to {OUTPUT_CSV}")
        except Exception as e:
            logging.error(f"Failed to save results to CSV: {e}")
            raise
        
        # Plot metrics
        plot_metrics(episode_metrics)
    
    except Exception as e:
        logging.error(f"Testing failed: {e}")
        raise
    
    finally:
        if env is not None:
            try:
                env.close()
                logging.info("Environment closed successfully")
            except Exception as e:
                logging.error(f"Error closing environment: {e}")

if __name__ == "__main__":
    main()