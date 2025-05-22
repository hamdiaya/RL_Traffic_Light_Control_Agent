import os
import time
import logging
import numpy as np
import torch
from matplotlib import pyplot as plt
from env import TrafficEnv
import traci
from dqn_agent import DQNAgent

# Hyperparameters
STATE_SIZE = 16
ACTION_SIZE = 12
HIDDEN_SIZE = 128
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
EPSILON_DECAY_STEPS = 900000
BUFFER_CAPACITY = 100000
BATCH_SIZE = 128
NUM_EPISODES = 3000
MAX_STEPS = 7200
TARGET_UPDATE_FREQ = 4
TAU = 0.005
EVAL_EPISODES = 20
SAVE_MODEL_EVERY = 50

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

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
        'explore_counts': [],
        'exploit_counts': [],
        'epsilon_values': [],
        'curriculum_difficulties': [],
        'vehicle_counts': [],
        'losses': []
    }

    for episode in range(NUM_EPISODES):
        # Curriculum learning progress
        curriculum_progress = min((episode / (NUM_EPISODES * 0.66)) ** 2, 1.0)
        state = env.reset()
        env.current_difficulty = curriculum_progress

        # Episode tracking
        total_reward = 0
        done = False
        step = 0
        explore_count = 0
        exploit_count = 0
        episode_loss = 0
        update_count = 0

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
                for p in agent.q_network.parameters():  # Fixed: Using q_network instead of qnetwork_local
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                agent.gradient_norms.append(total_norm)

            # Update state and counters
            state = next_state
            total_reward += reward
            step += 1

            if is_exploring:
                explore_count += 1
            else:
                exploit_count += 1

        # Calculate average loss
        avg_loss = episode_loss / max(1, update_count)
        
        # Store metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['explore_counts'].append(explore_count)
        metrics['exploit_counts'].append(exploit_count)
        metrics['epsilon_values'].append(agent.epsilon)
        metrics['curriculum_difficulties'].append(curriculum_progress)
        metrics['vehicle_counts'].append(sum(traci.edge.getLastStepVehicleNumber(e) for e in env.incoming_edges))
        metrics['losses'].append(avg_loss)

        # Log progress
        logging.info(
            f"Episode {episode + 1}/{NUM_EPISODES}, "
            f"Difficulty: {curriculum_progress:.2f}, "
            f"Reward: {total_reward:.1f}, "
            f"Loss: {avg_loss:.4f}, "
            f"Epsilon: {agent.epsilon:.3f}, "
            f"Explore: {explore_count}, "
            f"Exploit: {exploit_count} ,"
            f"action: {action}"
        )

        # Save model periodically
        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            agent.save_model("dqn_model.pth")
            logging.info(f"Model saved at episode {episode + 1}")

    return metrics

def plot_metrics(metrics, agent):
    """Visualize training metrics"""
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(20, 20))
    
    # Plot configurations
    plots = [
        ('episode_rewards', "Total Reward", "Total Reward per Episode"),
        ('explore_counts', "Exploration", "Exploration vs. Exploitation", 'exploit_counts'),
        ('epsilon_values', "Epsilon", "Epsilon Decay Over Time"),
        ('losses', "Training Loss", "Training Loss Over Episodes"),
        ('curriculum_difficulties', "Curriculum Difficulty", "Curriculum Progression", 'vehicle_counts')
    ]
    
    for i, plot in enumerate(plots, 1):
        plt.subplot(3, 3, i)
        
        if len(plot) == 3:  # Single plot
            key, ylabel, title = plot
            plt.plot(metrics[key])
            plt.ylabel(ylabel)
        else:  # Dual plot
            key1, label1, title, key2 = plot
            plt.plot(metrics[key1], label=label1)
            if key2 == 'vehicle_counts':
                norm_counts = np.array(metrics[key2]) / max(1, max(metrics[key2]))
                plt.plot(norm_counts, label="Normalized Vehicle Count")
            else:
                plt.plot(metrics[key2], label=label1.replace("Exploration", "Exploitation"))
            plt.legend()
        
        plt.xlabel("Episode")
        plt.title(title)

    # Additional plots
    plt.subplot(3, 3, 6)
    explore_ratio = [
        e/(e+x) for e, x in zip(metrics['explore_counts'], metrics['exploit_counts'])
    ]
    plt.plot(explore_ratio, label="Exploration Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.title("Exploration Ratio Over Time")
    plt.legend()

    plt.subplot(3, 3, 7)
    window_size = 50
    moving_avg = [
        np.mean(metrics['episode_rewards'][max(0, i-window_size):i+1]) 
        for i in range(len(metrics['episode_rewards']))
    ]
    plt.plot(moving_avg, label=f"Moving Avg (Window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Moving Average Reward")
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.hist(metrics['episode_rewards'], bins=50)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")

    plt.subplot(3, 3, 9)
    plt.plot(agent.gradient_norms)
    plt.xlabel("Update Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms During Training")

    plt.tight_layout()
    plt.savefig("plots/training_metrics.png")
    plt.close()

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
        plot_metrics(metrics, agent)
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