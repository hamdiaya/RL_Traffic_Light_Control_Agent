import os
import logging
import numpy as np
import torch
import traci
import pandas as pd
from envtest import TrafficEnv
from dqn_agent import DQNAgent

# Hyperparameters (aligned with training setup)
STATE_SIZE = 17
ACTION_SIZE = 12
HIDDEN_SIZE = 128
EVAL_EPISODES = 10  # Number of episodes for evaluation
MAX_STEPS = 7200    # Maximum steps per episode
LOG_FILE = "evaluation.log"

def setup_logging():
    """Configure logging for evaluation"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
    log_path = os.path.join(log_dir, LOG_FILE)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def initialize_environment():
    """Initialize SUMO environment for testing"""
    env = TrafficEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return env, device

def create_agent(device):
    """Initialize DQN agent with same parameters as training"""
    return DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=device  # Minimal parameters for testing
    )

def load_model(agent):
    """Load the trained model"""
    try:
        agent.load_model("dqn_model.pth")
        logging.info("Loaded trained model successfully.")
    except FileNotFoundError:
        logging.error("No saved model found ('dqn_model.pth'). Please train the agent first.")
        raise

def test_agent(env, agent):
    """Main testing loop, mirroring the training process"""
    # Metrics to track (similar to training)
    metrics = {
        'episode_rewards': [],
        'waiting_times': [],
        'queue_lengths': [],
        'actions': [],
        'vehicle_counts': [],
        'states': []  # Store state samples (queue lengths)
    }

    # Set agent to evaluation mode (no exploration)
    agent.epsilon = 0.1

    for episode in range(EVAL_EPISODES):
        state = env.reset()
        env.current_difficulty = 1.0  # Full difficulty, as in final training stages

        # Episode tracking
        total_reward = 0
        total_waiting_time = 0
        total_queue_length = 0
        episode_actions = []
        done = False
        step = 0
        episode_states = []

        while not done and step < MAX_STEPS:
            # Agent selects action using learned policy (no exploration)
            action, _ = agent.act(state, eval_mode=True)
            next_state, reward, done = env.step(action)

            # Collect traffic metrics (as in training)
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

        # Calculate averages for the episode
        avg_waiting_time = total_waiting_time / max(1, step)
        avg_queue_length = total_queue_length / max(1, step)

        # Store metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['waiting_times'].append(avg_waiting_time)
        metrics['queue_lengths'].append(avg_queue_length)
        metrics['actions'].extend(episode_actions)
        metrics['vehicle_counts'].append(sum(traci.edge.getLastStepVehicleNumber(e) for e in env.incoming_edges))
        metrics['states'].extend(episode_states)

        # Log episode results
        logging.info(
            f"Test Episode {episode + 1}/{EVAL_EPISODES}, "
            f"Reward: {total_reward:.1f}, "
            f"Waiting Time: {avg_waiting_time:.1f}s, "
            f"Queue Length: {avg_queue_length:.1f}"
        )

    # Calculate overall averages
    avg_reward = sum(metrics['episode_rewards']) / EVAL_EPISODES
    avg_waiting_time = sum(metrics['waiting_times']) / EVAL_EPISODES
    avg_queue_length = sum(metrics['queue_lengths']) / EVAL_EPISODES

    # Log final evaluation results
    logging.info(
        f"Evaluation Complete - Average Reward: {avg_reward:.1f}, "
        f"Average Waiting Time: {avg_waiting_time:.1f}s, "
        f"Average Queue Length: {avg_queue_length:.1f}"
    )

    # Save metrics to CSV files
    metrics_df = pd.DataFrame({
        'episode': np.arange(1, EVAL_EPISODES + 1),
        'episode_rewards': metrics['episode_rewards'],
        'waiting_times': metrics['waiting_times'],
        'queue_lengths': metrics['queue_lengths'],
        'vehicle_counts': metrics['vehicle_counts']
    })
    metrics_df.to_csv("evaluation_metrics.csv", index=False)

    # Save actions and states separately
    pd.DataFrame({'actions': metrics['actions']}).to_csv("evaluation_actions.csv", index=False)
    pd.DataFrame(metrics['states'], columns=['queue_ns', 'queue_ew', 'queue_sn', 'queue_ws']).to_csv("evaluation_states.csv", index=False)

    return metrics

def main():
    """Main testing pipeline"""
    setup_logging()
    env, device = initialize_environment()
    agent = create_agent(device)
    load_model(agent)

    try:
        metrics = test_agent(env, agent)
    except Exception as e:
        logging.error(f"Testing failed: {e}")
    finally:
        try:
            env.close()
        except Exception as e:
            logging.error(f"Error closing environment: {e}")

        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    main()