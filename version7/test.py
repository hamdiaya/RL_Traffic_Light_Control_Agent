import os
import logging
import numpy as np
import torch
import traci
import matplotlib.pyplot as plt
from env import TrafficEnv
from dqn_agent2 import DQNAgent
import csv

# Configuration
STATE_SIZE = 16
ACTION_SIZE = 12
HIDDEN_SIZE = 128
TEST_EPISODES = 20
MAX_STEPS = 7200
MODEL_PATH = "dqn_model.pth"
LOG_FILE = "test.log"
PLOT_PATH = "plots/test_metrics.png"
CSV_LOG_PATH = "logs/test_results.csv"

#########################################################################################
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)  # Create the folder if it doesn't exist
    log_path = os.path.join(log_dir, LOG_FILE)
    """Configure logging for test results"""
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

def load_agent(device):
    """Initialize and load trained DQN agent"""
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=1e-3,  # Learning rate (unused during testing)
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_min=0.0,
        epsilon_decay=0.9995,
        epsilon_decay_steps=2e6,
        buffer_capacity=200000,
        batch_size=256,
        tau=0.005,
        update_every=4,
        device=device,
        use_double=True,
        use_dueling=True,
        use_per=True
    )
    try:
        agent.load_model(MODEL_PATH)
        logging.info(f"Loaded trained model from {MODEL_PATH}")
    except FileNotFoundError:
        logging.error(f"Model file {MODEL_PATH} not found")
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
    return agent

def test_agent(env, agent):
    """Run test episodes and collect metrics"""
    metrics = {
        'total_rewards': [],
        'avg_waiting_times': [],
        'avg_queue_lengths': []
    }

    # Ensure plot directory exists
    os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)

    # Open CSV file and write headers
    with open(CSV_LOG_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Agent_Reward", "Avg_Waiting_Time", "Avg_Queue_Length"])

        for episode in range(TEST_EPISODES):
            state = env.reset()
            env.current_difficulty = 1.0  # Maximum difficulty for testing
            total_reward = 0
            waiting_times = []
            queue_lengths = []
            done = False
            step = 0

            while not done and step < MAX_STEPS:
                action, _ = agent.act(state, eval_mode=True)
                next_state, reward, done = env.step(action)

                total_reward += reward
                waiting_time = sum(traci.edge.getWaitingTime(e) for e in env.incoming_edges)
                queue_length = sum(traci.edge.getLastStepHaltingNumber(e) for e in env.incoming_edges)
                waiting_times.append(waiting_time)
                queue_lengths.append(queue_length)

                state = next_state
                step += 1

            avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
            avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0

            # Store metrics
            metrics['total_rewards'].append(total_reward)
            metrics['avg_waiting_times'].append(avg_waiting_time)
            metrics['avg_queue_lengths'].append(avg_queue_length)

            # Log results to file
            writer.writerow([episode + 1, total_reward, avg_waiting_time, avg_queue_length])

            # Log to console/file
            logging.info(
                f"Test Episode {episode + 1}/{TEST_EPISODES}, "
                f"Total Reward: {total_reward:.1f}, "
                f"Avg Waiting Time: {avg_waiting_time:.1f}s, "
                f"Avg Queue Length: {avg_queue_length:.1f} vehicles"
            )

    return metrics


def plot_metrics(metrics):
    """Plot test metrics"""
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(15, 10))

    # Total Reward
    plt.subplot(3, 1, 1)
    plt.plot(metrics['total_rewards'], label='Total Reward')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Test Episode")
    plt.grid(True)
    plt.legend()

    # Average Waiting Time
    plt.subplot(3, 1, 2)
    plt.plot(metrics['avg_waiting_times'], label='Avg Waiting Time', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time (s)")
    plt.title("Average Waiting Time per Test Episode")
    plt.grid(True)
    plt.legend()

    # Average Queue Length
    plt.subplot(3, 1, 3)
    plt.plot(metrics['avg_queue_lengths'], label='Avg Queue Length', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Queue Length (vehicles)")
    plt.title("Average Queue Length per Test Episode")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    logging.info(f"Test metrics plot saved to {PLOT_PATH}")

def main():
    """Main testing pipeline"""
    setup_logging()
    logging.info("Starting testing phase...")

    try:
        env, device = initialize_environment()
        agent = load_agent(device)
        metrics = test_agent(env, agent)
        plot_metrics(metrics)

        # Log summary statistics
        avg_reward = np.mean(metrics['total_rewards'])
        std_reward = np.std(metrics['total_rewards'])
        avg_waiting = np.mean(metrics['avg_waiting_times'])
        avg_queue = np.mean(metrics['avg_queue_lengths'])
        logging.info(
            f"Test Summary: "
            f"Avg Reward: {avg_reward:.1f} (±{std_reward:.1f}), "
            f"Avg Waiting Time: {avg_waiting:.1f}s, "
            f"Avg Queue Length: {avg_queue:.1f} vehicles"
        )

    except Exception as e:
        logging.error(f"Testing failed: {e}")
        raise
    finally:
        try:
            env.close()
        except Exception as e:
            logging.error(f"Error closing environment: {e}")

        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    main()