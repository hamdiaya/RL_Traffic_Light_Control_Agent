import os
import random

import gym
import traci
from sumolib import checkBinary
import numpy as np

# SUMO configuration
sumo_config = "sumo_files/sumo_config.sumocfg"
sumo_binary = checkBinary("sumo")  # Use "sumo-gui" for visualization

# Define valid vehicle routes (unchanged)
routes = {
    "route_1": ["E1_in", "E2_out"],  # West to East (Straight)
    "route_2": ["E1_in", "E4_out"],  # West to North (Right Turn)
    "route_3": ["E1_in", "E3_out"],  # West to South (Left Turn),
    "route_4": ["E3_in", "E1_out"],  # South to West (Left Turn)
    "route_5": ["E3_in", "E4_out"],  # South to North (Straight)
    "route_6": ["E3_in", "E2_out"],  # South to East (Right Turn),
    "route_7": ["E4_in", "E3_out"],  # North to South (Straight)
    "route_8": ["E4_in", "E1_out"],  # North to West (Right Turn)
    "route_9": ["E4_in", "E2_out"],  # North to East (Left Turn),
    "route_10": ["E2_in", "E1_out"],  # East to West (Straight)
    "route_11": ["E2_in", "E3_out"],  # East to South (Right Turn)
    "route_12": ["E2_in", "E4_out"],  # East to North (Left Turn)
}

GREEN_PHASES = [0, 2, 4, 6]
YELLOW_PHASES = [1, 3, 5, 7]

class TrafficEnv:
    def __init__(self):
        """Initialize SUMO environment."""
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1", "--no-warnings"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        
        # Define state size (use fixed 16 based on actual state representation)
        self.state_size = 16

        # Define action size (only green phases are selectable)
        self.action_size = 4  # Since GREEN_PHASES = [0, 2, 4, 6]

        # Define observation space (ensure proper normalization of state values)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # Define action space (4 discrete green phases)
        self.action_space = gym.spaces.Discrete(self.action_size)

    def generate_random_traffic(self):
        """Generate random vehicles dynamically based on the lane setup."""
        traffic_profiles = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.2,
        }

        profile = random.choices(list(traffic_profiles.keys()), weights=list(traffic_profiles.values()), k=1)[0]
        vehicle_probability = {"low": 0.2, "medium": 0.5, "high": 0.8}[profile]

        if random.random() < vehicle_probability:
            route_name, edges = random.choice(list(routes.items()))
            depart_lane = random.choice(["0", "1"])  # Choose one of the two lanes
            depart_speed = random.uniform(5, 11.1)

            route_id = f"{route_name}_{self.simulation_time}"
            traci.route.add(route_id, edges)

            vehicle_id = f"veh_{self.simulation_time}"
            traci.vehicle.add(vehicle_id, route_id, typeID="car", departLane=depart_lane, departSpeed=str(depart_speed))

    def get_state(self):
        """Retrieve the current state and normalize values."""
        state = []

        # 1. Current Traffic Light Phase (One-Hot Encoding)
        current_phase = traci.trafficlight.getPhase("n0")
        phase_encoding = [1 if current_phase == p else 0 for p in GREEN_PHASES]
        state.extend(phase_encoding)

        # 2. Number of Vehicles Waiting at Each Lane
        for edge in ["E1_in", "E2_in", "E3_in", "E4_in"]:
            state.append(traci.edge.getLastStepVehicleNumber(edge))

        # 3. Waiting Time of Vehicles
        for edge in ["E1_in", "E2_in", "E3_in", "E4_in"]:
            state.append(traci.edge.getWaitingTime(edge))

        # 4. Queue Length at Each Lane
        for edge in ["E1_in", "E2_in", "E3_in", "E4_in"]:
            state.append(traci.edge.getLastStepHaltingNumber(edge))

        assert len(state) == 16, f"State size mismatch! Expected 16, got {len(state)}"
        return np.array(state, dtype=np.float32)

    def calculate_reward(self):
        """Calculate the reward based on traffic conditions."""
        incoming_edges = ["E1_in", "E2_in", "E3_in", "E4_in"]
        outgoing_edges = ["E1_out", "E2_out", "E3_out", "E4_out"]

        # Penalize waiting time & queue length, reward throughput
        total_waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in incoming_edges)
        total_queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in incoming_edges)
        throughput = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in outgoing_edges)

        # Reward function
        reward = (
            -0.1 * total_waiting_time  # Penalize high waiting time
            -0.05 * total_queue_length  # Penalize large queues
            + 0.2 * throughput          # Reward vehicle throughput
            
        )

        return reward

    def step(self, action):
        """Advance the simulation with correct traffic light control."""
        self.current_action = GREEN_PHASES[action]  # Map action to correct green phase

        if self.last_action is not None and self.last_action != self.current_action:
            # Introduce a yellow transition before switching to new green phase
            yellow_phase = YELLOW_PHASES[GREEN_PHASES.index(self.last_action)]  # Correct mapping
            traci.trafficlight.setPhase("n0", yellow_phase)
            for _ in range(7):  # Yellow phase lasts 5 seconds
                traci.simulationStep()
                self.simulation_time += 1

        # Set new green phase
        traci.trafficlight.setPhase("n0", self.current_action)
        for _ in range(5):  # Green phase lasts 5 steps per action
            traci.simulationStep()
            self.simulation_time += 1
            if random.random() < 0.5:  # Keep dynamic traffic generation
                self.generate_random_traffic()

        self.last_action = self.current_action  # Update phase history
        next_state = self.get_state()
        reward = self.calculate_reward()
        done = self.simulation_time >= 3600  # Stop after 1-hour simulation

        return next_state, reward, done

    def reset(self):
        """Reset the environment for a new episode."""
        traci.close()
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None

        traci.trafficlight.setPhase("n0", GREEN_PHASES[0])

        return self.get_state()

    def close(self):
        """Close SUMO connection."""
        traci.close()

# Test Run
if __name__ == "__main__":
    env = TrafficEnv()
    try:
        for _ in range(3600):  # 1-hour simulation
            action = env.action_space.sample()  # Random action for testing
            next_state, reward, done = env.step(action)
            #print(f"Reward: {reward}")
            print(env.get_state())
            if done:
                break
    finally:
        env.close()
