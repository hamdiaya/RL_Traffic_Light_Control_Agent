import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set Seaborn style for better visuals
sns.set(style="darkgrid")

def plot_metrics(metrics, save_dir="plots"):
    """
    Generate and save plots for DRL training metrics.
    
    Args:
        metrics (dict): Dictionary containing lists of metrics (e.g., episode_rewards, waiting_times)
        save_dir (str): Directory to save plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episodes = np.arange(1, len(metrics['episode_rewards']) + 1)
    
    # 1. Reward Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['episode_rewards'], label="Total Reward", color="#4CAF50")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_over_time.png"))
    plt.close()

    # 2. Moving Average Reward
    window_size = 50
    moving_avg = [
        np.mean(metrics['episode_rewards'][max(0, i-window_size):i+1]) 
        for i in range(len(metrics['episode_rewards']))
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, moving_avg, label=f"Moving Avg (Window={window_size})", color="#2196F3")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Moving Average Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "moving_avg_reward.png"))
    plt.close()

    # 3. Waiting Time Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['waiting_times'], label="Avg Waiting Time (s)", color="#FF9800")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time (s)")
    plt.title("Average Waiting Time per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "waiting_time_over_time.png"))
    plt.close()

    # 4. Queue Length Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['queue_lengths'], label="Queue Length", color="#9C27B0")
    plt.xlabel("Episode")
    plt.ylabel("Queue Length")
    plt.title("Queue Length per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "queue_length_over_time.png"))
    plt.close()

    # 5. Chosen Actions Distribution
    action_labels = [
        f"Phase {p} ({d}s)" 
        for p in [0, 2, 4, 6] 
        for d in [20, 30, 50]
    ]
    action_counts = pd.Series(metrics['actions']).value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values, palette="viridis")
    plt.xticks(ticks=range(len(action_counts)), labels=[action_labels[i] for i in action_counts.index], rotation=45)
    plt.xlabel("Actions (Phase + Duration)")
    plt.ylabel("Frequency")
    plt.title("Chosen Actions Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "action_distribution.png"))
    plt.close()

    # 6. Exploration vs. Exploitation
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['explore_counts'], label="Exploration", color="#4CAF50")
    plt.plot(episodes, metrics['exploit_counts'], label="Exploitation", color="#F44336")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("Exploration vs. Exploitation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "exploration_exploitation.png"))
    plt.close()

    # 7. Epsilon Decay
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['epsilon_values'], label="Epsilon", color="#2196F3")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "epsilon_decay.png"))
    plt.close()

    # 8. Training Loss :Monitors the DQN loss to ensure stable learning.
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['losses'], label="Training Loss", color="#F44336")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    # # 9. Reward Distribution
    # plt.figure(figsize=(10, 6))
    # plt.hist(metrics['episode_rewards'], bins=50, color="#4CAF50", alpha=0.7)
    # plt.xlabel("Reward")
    # plt.ylabel("Frequency")
    # plt.title("Reward Distribution")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "reward_distribution.png"))
    # plt.close()

    # # 10. State Distribution (Queue Lengths NS vs. EW)
    # plt.figure(figsize=(10, 6))
    # states = np.array(metrics['states'])
    # plt.scatter(states[:, 0], states[:, 1], alpha=0.5, color="#2196F3")
    # plt.xlabel("Queue Length NS")
    # plt.ylabel("Queue Length EW")
    # plt.title("State Distribution (Queue Lengths)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "state_distribution.png"))
    # plt.close()

    # 11. Curriculum Difficulty and Vehicle Counts: Tracks curriculum progression and vehicle counts to correlate with difficulty.
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, metrics['curriculum_difficulties'], label="Curriculum Difficulty", color="#9C27B0")
    norm_vehicle_counts = np.array(metrics['vehicle_counts']) / max(1, max(metrics['vehicle_counts']))
    plt.plot(episodes, norm_vehicle_counts, label="Normalized Vehicle Count", color="#FF9800")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Curriculum Difficulty and Vehicle Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curriculum_vehicle_counts.png"))
    plt.close()

    # 12. Gradient Norms:Monitors gradient norms to diagnose training stability.
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['gradient_norms'], label="Gradient Norm", color="#F44336")
    plt.xlabel("Update Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms During Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient_norms.png"))
    plt.close()

if __name__ == "__main__":
    # Example usage with sample data
    sample_metrics = {
        'episode_rewards': np.random.normal(50, 10, 1000),
        'waiting_times': np.random.normal(20, 5, 1000),
        'queue_lengths': np.random.normal(10, 3, 1000),
        'actions': np.random.randint(0, 12, 10000),
        'explore_counts': np.random.randint(0, 1000, 1000),
        'exploit_counts': np.random.randint(0, 1000, 1000),
        'epsilon_values': np.exp(-0.005 * np.arange(1000)),
        'losses': np.random.normal(0.1, 0.02, 1000),
        'states': np.random.rand(1000, 2) * 10,
        'curriculum_difficulties': np.linspace(0, 1, 1000),
        'vehicle_counts': np.random.randint(0, 200, 1000),
        'gradient_norms': np.random.normal(1, 0.2, 1000)
    }
    plot_metrics(sample_metrics)

    #Load Metrics for Plotting:
    
    # metrics = pd.read_csv("metrics.csv").to_dict(orient="lists")
    # metrics['actions'] = pd.read_csv("actions.csv")['actions'].values
    # metrics['states'] = pd.read_csv("states.csv")[['queue_ns', 'queue_ew', 'queue_sn', 'queue_ws']].values
    # metrics['gradient_norms'] = pd.read_csv("gradient_norms.csv")['gradient_norms'].values
    # plot_metrics(metrics)