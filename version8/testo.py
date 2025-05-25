import os
import logging
import numpy as np
import pandas as pd
import torch
import traci
from env import TrafficEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

# Constants
STATE_SIZE = 17
ACTION_SIZE = 32
HIDDEN_SIZE = 128
EVAL_EPISODES = 3
MAX_STEPS = 200  # Shorten for easier observation
MODEL_PATH = "dqn_model.pth"

# Thresholds for perfection
MAX_ALLOWED_AVG_QUEUE = 5
MAX_ALLOWED_WAIT_TIME = 10

def create_agent(device):
    return DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration
        epsilon_min=0.0,
        epsilon_decay=1.0,
        epsilon_decay_steps=1,
        buffer_capacity=1,
        batch_size=1,
        tau=0.005,
        update_every=4,
        device=device,
        use_double=True,
        use_dueling=True,
        use_per=True
    )

def print_state_info(state, step):
    """Nicely prints state information for debugging"""
    print(f"\n🔹 Step {step}")
    print(f"Raw State: {state}")
    try:
        print(f"  ▪ Phase one-hot: {state[0:4]}")
        print(f"  ▪ Queue lengths: N={state[4]:.2f}, E={state[5]:.2f}, S={state[6]:.2f}, W={state[7]:.2f}")
        print(f"  ▪ Wait times:    N={state[8]:.2f}, E={state[9]:.2f}, S={state[10]:.2f}, W={state[11]:.2f}")
        print(f"  ▪ Vehicle counts N={state[12]:.2f}, E={state[13]:.2f}, S={state[14]:.2f}, W={state[15]:.2f}")
        print(f"  ▪ Time of day: {state[16]:.2f}")
    except IndexError:
        print("⚠ State format mismatch!")

def evaluate_agent():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficEnv()
    agent = create_agent(device)

    try:
        agent.load_model(MODEL_PATH)
        logging.info("✅ Model loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        return

    total_rewards = []
    avg_queues = []
    avg_waiting_times = []

    for ep in range(EVAL_EPISODES):
        state = env.reset()
        env.current_difficulty = 1.0
        done = False
        episode_reward = 0
        total_queue = 0
        total_waiting = 0
        step = 0

        print(f"\n\n=== 🚦 Evaluation Episode {ep + 1} ===")

        while not done and step < MAX_STEPS:
            print_state_info(state, step)

            action, _ = agent.act(state, eval_mode=True)
            print(f"  ▪ Chosen Action: {action}")

            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward

            queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in env.incoming_edges)
            wait_time = sum(traci.edge.getWaitingTime(e) for e in env.incoming_edges)
            total_queue += queue
            total_waiting += wait_time
            step += 1

        avg_queue = total_queue / max(1, step)
        avg_wait = total_waiting / max(1, step)
        total_rewards.append(episode_reward)
        avg_queues.append(avg_queue)
        avg_waiting_times.append(avg_wait)

        logging.info(f"✅ Episode {ep+1} done — Reward: {episode_reward:.1f}, Avg Queue: {avg_queue:.2f}, Avg Wait: {avg_wait:.2f}s")

    # Summary
    mean_reward = np.mean(total_rewards)
    mean_queue = np.mean(avg_queues)
    mean_wait = np.mean(avg_waiting_times)

    print("\n=== 📊 Evaluation Summary ===")
    print(f"Average Reward:        {mean_reward:.2f}")
    print(f"Average Queue Length:  {mean_queue:.2f}")
    print(f"Average Waiting Time:  {mean_wait:.2f} seconds")

    passed = mean_queue <= MAX_ALLOWED_AVG_QUEUE and mean_wait <= MAX_ALLOWED_WAIT_TIME
    if passed:
        print("✅ Agent PASSED — nearly perfect control!")
    else:
        print("❌ Agent FAILED — improve policy or adjust threshold.")

    # Optional: plot
    plt.figure(figsize=(10, 4))
    plt.plot(avg_queues, label='Avg Queue')
    plt.plot(avg_waiting_times, label='Avg Wait Time')
    plt.title('Evaluation Metrics per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Cleanup
    try:
        env.close()
    except:
        pass

    if traci.isLoaded():
        traci.close()

if __name__ == "__main__":
    evaluate_agent()
