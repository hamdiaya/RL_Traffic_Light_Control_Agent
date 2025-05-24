import numpy as np
import torch
import random
import os
import pandas as pd
import traci
import logging
from env import TrafficEnv
from dqn_agent import DQNAgent
from matplotlib import pyplot as plt

# Configuration
MODEL_PATH = "dqn_model.pth"
NUM_TEST_EPISODES = 10
MAX_STEPS = 7200
FIXED_SEED = 42
LOG_FILE = "test.log"
STATE_SIZE = 17
ACTION_SIZE = 32  # 4 phases * 8 durations
HIDDEN_SIZE = 128
FIXED_PHASE_DURATION = 30
GREEN_PHASES = [0, 2, 4, 6]
YELLOW_PHASES = [1, 3, 5, 7]
TL_ID = "TL"

def setup_logging():
    """Configure logging"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILE)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
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
        lr=1e-3,
        gamma=0.99,
        epsilon_start=0.0,
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

def run_episode(env, agent, use_agent=True, seed=None):
    """Run a single episode for RL agent or fixed-time controller"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    state = env.reset()
    total_reward = 0
    total_waiting_time = 0
    total_queue_length = 0
    total_throughput = 0
    step = 0
    done = False
    current_phase_idx = 0
    env.prev_exited = 0

    while not done and step < MAX_STEPS:
        try:
            if use_agent:
                # RL Agent: selects from 32 actions (4 phases * 8 durations)
                action, _ = agent.act(state, eval_mode=True)
                next_state, reward, done = env.step(action)
                phase_idx = action // len(env.phase_durations)
                duration_idx = action % len(env.phase_durations)
                logging.debug(f"RL Action: {action}, Phase: {GREEN_PHASES[phase_idx]}, "
                             f"Duration: {env.phase_durations[duration_idx]}")
            else:
                # Fixed-time controller: cycle through green phases with fixed duration
                if step % FIXED_PHASE_DURATION == 0:
                    current_phase_idx = (current_phase_idx + 1) % len(GREEN_PHASES)
                current_action = GREEN_PHASES[current_phase_idx]
                
                # Yellow transition
                if env.last_action is not None and env.last_action != current_action:
                    yellow_phase = YELLOW_PHASES[GREEN_PHASES.index(env.last_action)]
                    traci.trafficlight.setPhase(TL_ID, yellow_phase)
                    for _ in range(5):
                        traci.simulationStep()
                        env.simulation_time += 1
                        if random.random() < 0.5:
                            env.generate_random_traffic()
                
                # Green phase
                traci.trafficlight.setPhase(TL_ID, current_action)
                for _ in range(FIXED_PHASE_DURATION):
                    traci.simulationStep()
                    env.simulation_time += 1
                    if random.random() < 0.5:
                        env.generate_random_traffic()
                
                env.last_action = current_action
                env.current_action = current_action
                env.current_duration = FIXED_PHASE_DURATION
                next_state = env.get_state()
                reward = env.calculate_reward()
                done = env.simulation_time >= 7200

            # Collect metrics
            waiting_time = sum(traci.edge.getWaitingTime(e) for e in env.incoming_edges)
            queue_length = sum(traci.edge.getLastStepHaltingNumber(e) for e in env.incoming_edges)
            current_exited = traci.simulation.getArrivedNumber()
            throughput = current_exited - env.prev_exited
            env.prev_exited = current_exited

            total_reward += reward
            total_waiting_time += waiting_time
            total_queue_length += queue_length
            total_throughput += throughput
            state = next_state
            step += 1

            logging.debug(f"Step {step}: Waiting Time: {waiting_time:.1f}s, Queue Length: {queue_length:.1f}, "
                         f"Throughput: {throughput}, Reward: {reward:.2f}")
        except traci.FatalTraCIError as e:
            logging.error(f"TraCI error at step {step}: {e}")
            done = True
            break

    avg_waiting_time = total_waiting_time / max(1, step)
    avg_queue_length = total_queue_length / max(1, step)

    return {
        'reward': total_reward,
        'waiting_time': avg_waiting_time,
        'queue_length': avg_queue_length,
        'throughput': total_throughput
    }

def plot_comparison(metrics_rl, metrics_default):
    """Create bar charts comparing RL and fixed-time metrics"""
    metrics = ['reward', 'waiting_time', 'queue_length', 'throughput']
    labels = ['Total Reward', 'Average Waiting Time (s)', 'Average Queue Length', 'Total Throughput']
    
    for metric, label in zip(metrics, labels):
        rl_mean = np.mean([ep[metric] for ep in metrics_rl])
        default_mean = np.mean([ep[metric] for ep in metrics_default])
        rl_std = np.std([ep[metric] for ep in metrics_rl])
        default_std = np.std([ep[metric] for ep in metrics_default])

        fig, ax = plt.subplots()
        ax.bar([0, 1], [rl_mean, default_mean], yerr=[rl_std, default_std], 
               tick_label=['RL Agent', 'Fixed-Time'], color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison (Difficulty 0.5)')
        plt.savefig(f'test_{metric}_comparison.png')
        plt.close()

def main():
    setup_logging()
    logging.info("🚦 === Testing Started (Difficulty 0.5) ===")

    env, device = initialize_environment()
    agent = create_agent(device)

    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
        agent.q_network.eval()
        logging.info(f"✅ Successfully loaded model from {MODEL_PATH}")
    else:
        logging.error(f"❌ No model found at {MODEL_PATH}")
        env.close()
        return

    agent.epsilon = 0.0
    logging.info(f"Device: {next(agent.q_network.parameters()).device}")

    metrics_rl = []
    metrics_default = []

    for episode in range(NUM_TEST_EPISODES):
        seed = FIXED_SEED + episode

        rl_result = run_episode(env, agent, use_agent=True, seed=seed)
        default_result = run_episode(env, None, use_agent=False, seed=seed)

        metrics_rl.append(rl_result)
        metrics_default.append(default_result)

        logging.info(
            f"Episode {episode+1}/{NUM_TEST_EPISODES}\n"
            f"RL Agent - Reward: {rl_result['reward']:.1f}, "
            f"Waiting Time: {rl_result['waiting_time']:.1f}s, "
            f"Queue Length: {rl_result['queue_length']:.1f}, "
            f"Throughput: {rl_result['throughput']}\n"
            f"Fixed-Time - Reward: {default_result['reward']:.1f}, "
            f"Waiting Time: {default_result['waiting_time']:.1f}s, "
            f"Queue Length: {default_result['queue_length']:.1f}, "
            f"Throughput: {default_result['throughput']}"
        )

    pd.DataFrame(metrics_rl).to_csv("test_rl_metrics_difficulty_0.5.csv", index=False)
    pd.DataFrame(metrics_default).to_csv("test_default_metrics_difficulty_0.5.csv", index=False)

    plot_comparison(metrics_rl, metrics_default)

    logging.info("\n📊 === Final Results (Difficulty 0.5) ===")
    rl_avg_reward = np.mean([ep['reward'] for ep in metrics_rl])
    rl_avg_waiting = np.mean([ep['waiting_time'] for ep in metrics_rl])
    rl_avg_queue = np.mean([ep['queue_length'] for ep in metrics_rl])
    rl_avg_throughput = np.mean([ep['throughput'] for ep in metrics_rl])

    default_avg_reward = np.mean([ep['reward'] for ep in metrics_default])
    default_avg_waiting = np.mean([ep['waiting_time'] for ep in metrics_default])
    default_avg_queue = np.mean([ep['queue_length'] for ep in metrics_default])
    default_avg_throughput = np.mean([ep['throughput'] for ep in metrics_default])

    reward_improvement = ((rl_avg_reward - default_avg_reward) / abs(default_avg_reward) * 100
                         if default_avg_reward != 0 else float('inf'))
    waiting_improvement = ((default_avg_waiting - rl_avg_waiting) / abs(default_avg_waiting) * 100
                          if default_avg_waiting != 0 else float('inf'))
    queue_improvement = ((default_avg_queue - rl_avg_queue) / abs(default_avg_queue) * 100
                         if default_avg_queue != 0 else float('inf'))
    throughput_improvement = ((rl_avg_throughput - default_avg_throughput) / abs(default_avg_throughput) * 100
                             if default_avg_throughput != 0 else float('inf'))

    logging.info(f"RL Agent - Avg Reward: {rl_avg_reward:.1f}, "
                 f"Avg Waiting Time: {rl_avg_waiting:.1f}s, "
                 f"Avg Queue Length: {rl_avg_queue:.1f}, "
                 f"Avg Throughput: {rl_avg_throughput:.1f}")
    logging.info(f"Fixed-Time - Avg Reward: {default_avg_reward:.1f}, "
                 f"Avg Waiting Time: {default_avg_waiting:.1f}s, "
                 f"Avg Queue Length: {default_avg_queue:.1f}, "
                 f"Avg Throughput: {default_avg_throughput:.1f}")
    logging.info(f"Improvements - Reward: {reward_improvement:.1f}%, "
                 f"Waiting Time: {waiting_improvement:.1f}%, "
                 f"Queue Length: {queue_improvement:.1f}%, "
                 f"Throughput: {throughput_improvement:.1f}%")

    env.close()

if __name__ == "__main__":
    main()