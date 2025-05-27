import os
import time
import logging
import numpy as np
import torch
import traci
import pandas as pd
from matplotlib import pyplot as plt
from env import TrafficEnv
from dqn_agent import DQNAgent
from plot import plot_metrics  

# if 'SUMO_HOME' not in os.environ:
#     os.environ['SUMO_HOME'] = "C:/path/to/your/sumo"  # Replace with actual SUMO path
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)

# Hyperparameters 
STATE_SIZE = 17
ACTION_SIZE = 32
HIDDEN_SIZE = 128
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
EPSILON_DECAY_STEPS = 2e6
BUFFER_CAPACITY = 200000
BATCH_SIZE = 256
NUM_EPISODES = 10000
MAX_STEPS = 7200
TARGET_UPDATE_FREQ = 4
TAU = 0.005
EVAL_EPISODES = 20
SAVE_MODEL_EVERY = 50
LOG_FILE = "training.log"

#########################################################################################
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)  # Create the folder if it doesn't exist
    log_path = os.path.join(log_dir, LOG_FILE )

    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
#########################################################################################

def initialize_environment():
    """Initialize SUMO environment"""
    env = TrafficEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return env, device

def create_agent(device):
    """Initialize DQN agent"""
    return DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        tau=TAU,
        update_every=TARGET_UPDATE_FREQ,
        device=device,
        use_double=True,
        use_dueling=True,
        use_per=True
    )

def load_or_initialize_model(agent):
    """Load existing model or start fresh"""
    try:
        agent.load_model("dqn_model.pth")
        logging.info("Loaded existing model.")
    except FileNotFoundError:
        logging.info("No saved model found, training from scratch.")

def train_agent(env, agent):
    """Main training loop"""
    # Tracking metrics
    metrics = {
        'episode_rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'actions': [],
        'explore_counts': [],
        'exploit_counts': [],
        'epsilon_values': [],
        'curriculum_difficulties': [],
        'vehicle_counts': [],
        'losses': [],
        'states': [],  # Store state samples (queue lengths)
        'gradient_norms': []
    }

    for episode in range(NUM_EPISODES):
        # Curriculum learning progress
        curriculum_progress = min((episode / (NUM_EPISODES * 0.66)) ** 2, 1.0)
        state = env.reset()
        env.current_difficulty = curriculum_progress

        # Episode tracking
        total_reward = 0
        total_waiting_time = 0
        total_queue_length = 0
        episode_actions = []
        done = False
        step = 0
        explore_count = 0
        exploit_count = 0
        episode_loss = 0
        update_count = 0
        episode_states = []

        while not done and step < MAX_STEPS:
            # Agent takes action
            action, is_exploring = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
                update_count += 1

                # Track gradient norms
                total_norm = 0
                for p in agent.q_network.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                agent.gradient_norms.append(total_norm)

            # Collect traffic metrics
            waiting_time = sum(traci.edge.getWaitingTime(e) for e in env.incoming_edges)
            queue_length = sum(traci.edge.getLastStepHaltingNumber(e) for e in env.incoming_edges)
            
            # Update state and counters
            total_reward += reward
            total_waiting_time += waiting_time
            total_queue_length += queue_length
            episode_actions.append(action)
            episode_states.append(state[4:8])  # Queue lengths (indices 4-7 in state)
            state = next_state
            step += 1

            if is_exploring:
                explore_count += 1
            else:
                exploit_count += 1

        # Calculate averages
        avg_loss = episode_loss / max(1, update_count)
        avg_waiting_time = total_waiting_time / max(1, step)
        avg_queue_length = total_queue_length / max(1, step)
        
        # Store metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['waiting_times'].append(avg_waiting_time)
        metrics['queue_lengths'].append(avg_queue_length)
        metrics['actions'].extend(episode_actions)
        metrics['explore_counts'].append(explore_count)
        metrics['exploit_counts'].append(exploit_count)
        metrics['epsilon_values'].append(agent.epsilon)
        metrics['curriculum_difficulties'].append(curriculum_progress)
        metrics['vehicle_counts'].append(sum(traci.edge.getLastStepVehicleNumber(e) for e in env.incoming_edges))
        metrics['losses'].append(avg_loss)
        metrics['states'].extend(episode_states)
        metrics['gradient_norms'].extend(agent.gradient_norms[-update_count:])

        # Log progress
        logging.info(
            f"Episode {episode + 1}/{NUM_EPISODES}, "
            f"Difficulty: {curriculum_progress:.2f}, "
            f"Reward: {total_reward:.1f}, "
            f"Waiting Time: {avg_waiting_time:.1f}s, "
            f"Queue Length: {avg_queue_length:.1f}, "
            f"Loss: {avg_loss:.4f}, "
            f"Epsilon: {agent.epsilon:.3f}, "
            f"Explore: {explore_count}, "
            f"Exploit: {exploit_count}"
        )

        # Save model periodically
        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            agent.save_model("dqn_model.pth")
            logging.info(f"Model saved at episode {episode + 1}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'episode': np.arange(1, len(metrics['episode_rewards']) + 1),
        'episode_rewards': metrics['episode_rewards'],
        'waiting_times': metrics['waiting_times'],
        'queue_lengths': metrics['queue_lengths'],
        'explore_counts': metrics['explore_counts'],
        'exploit_counts': metrics['exploit_counts'],
        'epsilon_values': metrics['epsilon_values'],
        'curriculum_difficulties': metrics['curriculum_difficulties'],
        'vehicle_counts': metrics['vehicle_counts'],
        'losses': metrics['losses']
    })
    metrics_df.to_csv("metrics.csv", index=False)
    
    # Save actions and states separately due to different lengths
    pd.DataFrame({'actions': metrics['actions']}).to_csv("actions.csv", index=False)
    pd.DataFrame(metrics['states'], columns=['queue_ns', 'queue_ew', 'queue_sn', 'queue_ws']).to_csv("states.csv", index=False)
    pd.DataFrame({'gradient_norms': metrics['gradient_norms']}).to_csv("gradient_norms.csv", index=False)
    
    return metrics

def plot_metrics_wrapper(metrics, agent):
    """Wrapper to call plot_metrics from plot.py"""
    # Ensure gradient_norms is included
    metrics['gradient_norms'] = agent.gradient_norms
    plot_metrics(metrics)

def evaluate_agent(env, agent):
    """Evaluate trained agent"""
    logging.info("Starting evaluation...")
    agent.epsilon = 0.0  # Disable exploration
    total_rewards = []
    
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        env.current_difficulty = 1.0  # Full difficulty
        total_reward = 0
        done = False
        
        while not done:
            action, _ = agent.act(state, eval_mode=True)
            state, reward, done = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    logging.info(f"Evaluation Complete - Average Reward: {avg_reward:.1f}")

def main():
    """Main training pipeline"""
    setup_logging()
    env, device = initialize_environment()
    agent = create_agent(device)
    load_or_initialize_model(agent)
    
    try:
        metrics = train_agent(env, agent)
        plot_metrics_wrapper(metrics, agent)  # Call plotting function
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