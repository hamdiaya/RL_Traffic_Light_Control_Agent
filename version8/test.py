import os
import logging
import torch
from env import TrafficEnv
from dqn_agent import DQNAgent

# Constants from your env
MAX_Q = 50
MAX_WAIT = 3600
MAX_VEH = 10

STATE_SIZE = 17  # Adjust if needed
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
TAU = 0.005
TARGET_UPDATE_FREQ = 4
MODEL_PATH = "dqn_model.pth"  # Your trained model path

# GREEN_PHASES from your env, adjust to your actual list
GREEN_PHASES = ['phase1', 'phase2', 'phase3', 'phase4']  # example, replace with actual

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_environment():
    env = TrafficEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return env, device

def create_agent(device):
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

def load_model(agent):
    if os.path.exists(MODEL_PATH):
        agent.load_model(MODEL_PATH)
        logging.info(f"Model loaded from {MODEL_PATH}")
    else:
        logging.error(f"No saved model found at {MODEL_PATH}")
        exit(1)

def decode_phase(one_hot):
    # one_hot is a list of length 4 (last 4 bits)
    idx = one_hot.index(1) if 1 in one_hot else -1
    if idx == -1:
        return "Unknown"
    return GREEN_PHASES[idx]

def test_agent(env, agent, num_episodes=3):
    agent.epsilon = 0.0  # No exploration
    for ep in range(num_episodes):
        state = env.reset()
        env.current_difficulty = 1.0
        done = False
        step = 0
        total_reward = 0

        sum_waiting_time = 0
        sum_queue_length = 0
        count_steps = 0

        print(f"\n=== Episode {ep+1} ===")
        while not done:
            action, _ = agent.act(state, eval_mode=True)
            next_state, reward, done = env.step(action)
            total_reward += reward

            # --- Decode state ---
            N = len(GREEN_PHASES)
            norm_queues = [state[i*3] for i in range(N)]
            norm_waits = [state[i*3+1] for i in range(N)]
            norm_vehs = [state[i*3+2] for i in range(N)]
            norm_dur = state[3*N]
            one_hot = state[3*N+1:3*N+5]

            current_phase = decode_phase(one_hot)

            # --- Current phase index ---
            phase_idx = one_hot.index(1) if 1 in one_hot else None
            if phase_idx is not None:
                cur_queue = norm_queues[phase_idx] * MAX_Q
                cur_wait = norm_waits[phase_idx] * MAX_WAIT
                cur_veh = norm_vehs[phase_idx] * MAX_VEH
            else:
                cur_queue = cur_wait = cur_veh = 0

            print(f"Step {step}:")
            print(f"  State vector: {state}")
            print(f"  Current phase: {current_phase}")
            print(f"  Queue length (current phase): {cur_queue:.2f}")
            print(f"  Waiting time (current phase): {cur_wait:.2f}")
            print(f"  Vehicle count (current phase): {cur_veh:.2f}")
            print(f"  Normalized phase duration: {norm_dur:.3f}")
            print(f"  Reward: {reward:.3f}\n")

            # --- Track total queue/waiting over ALL phases ---
            total_queue_all_phases = sum([q * MAX_Q for q in norm_queues])
            total_wait_all_phases = sum([w * MAX_WAIT for w in norm_waits])

            sum_queue_length += total_queue_all_phases
            sum_waiting_time += total_wait_all_phases
            count_steps += 1

            state = next_state
            step += 1

        avg_queue = sum_queue_length / count_steps
        avg_wait = sum_waiting_time / count_steps

        print(f"=== Episode {ep+1} Summary ===")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Avg queue length (all phases): {avg_queue:.2f}")
        print(f"  Avg waiting time (all phases): {avg_wait:.2f}")

def main():
    setup_logging()
    env, device = initialize_environment()
    agent = create_agent(device)
    load_model(agent)

    try:
        test_agent(env, agent, num_episodes=3)
    except Exception as e:
        logging.error(f"Test failed: {e}")
    finally:
        try:
            env.close()
        except Exception as e:
            logging.error(f"Error closing env: {e}")
        import traci
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    main()
