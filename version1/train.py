import traci
import numpy as np
from version1.dqn_agent import DQNAgent

# SUMO environment functions
def get_state():
    # Traffic light state (0 = red, 1 = green)
    current_phase = traci.trafficlight.getPhase("intersection")
    current_color = 1 if current_phase == 0 else 0  # Assuming phase 0 is green

    # Number of cars
    num_cars = traci.edge.getLastStepVehicleNumber("road1") + traci.edge.getLastStepVehicleNumber("road2")

    # Average waiting time
    waiting_time_road1 = traci.edge.getWaitingTime("road1")
    waiting_time_road2 = traci.edge.getWaitingTime("road2")
    avg_waiting_time = (waiting_time_road1 + waiting_time_road2) / 2

    return [current_color, num_cars, avg_waiting_time]

def perform_action(action):
    if action == 0:  # Switch road 1 to green, road 2 to red
        traci.trafficlight.setPhase("intersection", 0)
    elif action == 1:  # Switch road 1 to red, road 2 to green
        traci.trafficlight.setPhase("intersection", 2)
    elif action == 2:  # Keep current state
        pass

def calculate_reward():
    waiting_time_road1 = traci.edge.getWaitingTime("road1")
    waiting_time_road2 = traci.edge.getWaitingTime("road2")
    avg_waiting_time = (waiting_time_road1 + waiting_time_road2) / 2
    return -avg_waiting_time  # Negative waiting time as reward

# Training function
def run_simulation():
    traci.start(["sumo-gui", "-c", "your_config.sumocfg"])  # Start SUMO
    agent = DQNAgent(state_dim=3, action_dim=3)  # Initialize agent

    for episode in range(1000):  # Number of episodes
        traci.simulationStep()  # Advance simulation by one step
        state = get_state()  # Get current state
        action = agent.act(state)  # Choose action
        perform_action(action)  # Perform action
        next_state = get_state()  # Get next state
        reward = calculate_reward()  # Calculate reward
        done = traci.simulation.getTime() >= 1000  # End episode after 1000 steps
        agent.remember(state, action, reward, next_state, done)
        agent.replay(32)  # Train on a batch of 32 experiences

        # Track total reward for the episode
        if episode == 0:
            total_reward = reward
        else:
            total_reward += reward

        # Store the total reward for the episode
        if done:
            agent.rewards.append(total_reward)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    traci.close()

    # Plot the results after training
    agent.plot_results()

if __name__ == "__main__":
    run_simulation()