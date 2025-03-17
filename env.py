import os
import random
import traci
from sumolib import checkBinary
import numpy as np

# SUMO configuration
sumo_config = "sumo_files/sumo_config.sumocfg"
sumo_binary = checkBinary("sumo")  # Use "sumo-gui" for visualization

# Define valid vehicle routes (updated for new lanes)
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

class TrafficEnv:
    def __init__(self):
        """Initialize SUMO environment."""
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1","--no-warnings"])
        self.simulation_time = 0
        self.last_action = None  # Track the last action for phase switching penalty
        self.current_action = None  # Track the current action for phase switching penalty
        self.action_space = [0, 1, 2, 3]  # 4 actions corresponding to the 4 traffic light phases
        self.state_space = self.get_state().shape  # Define state space

    def generate_random_traffic(self):
        """Generate random vehicles dynamically based on new lane setup."""
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
            depart_speed = random.uniform(5, 13.9)

            route_id = f"{route_name}_{self.simulation_time}"
            traci.route.add(route_id, edges)

            vehicle_id = f"veh_{self.simulation_time}"
            traci.vehicle.add(vehicle_id, route_id, typeID="car", departLane=depart_lane, departSpeed=str(depart_speed))

    def get_state(self):
        """Retrieve the current state of the environment."""
        state = []

        # 1. Current Traffic Light Phase
        current_phase = traci.trafficlight.getPhase("n0")
        state.append(current_phase)

        # 2. Number of Vehicles Waiting at Each Lane
        for edge in ["E1_in", "E2_in", "E3_in", "E4_in"]:
            state.append(traci.edge.getLastStepVehicleNumber(edge))

        # 3. Waiting Time of Vehicles at Each Lane
        for edge in ["E1_in", "E2_in", "E3_in", "E4_in"]:
            state.append(traci.edge.getWaitingTime(edge))

        # 4. Queue Length at Each Lane
        for edge in ["E1_in", "E2_in", "E3_in", "E4_in"]:
            state.append(traci.edge.getLastStepHaltingNumber(edge))

        return np.array(state)

    def calculate_reward(self):
        """Calculate the reward based on traffic conditions."""
        # Define the edges to monitor
        incoming_edges = ["E1_in", "E2_in", "E3_in", "E4_in"]
        outgoing_edges = ["E1_out", "E2_out", "E3_out", "E4_out"]

        # 1. Total Waiting Time (Penalty)
        total_waiting_time = sum(traci.edge.getWaitingTime(edge) for edge in incoming_edges)

        # 2. Total Queue Length (Penalty)
        total_queue_length = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in incoming_edges)

        # 3. Throughput (Reward)
        throughput = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in outgoing_edges)

        # 4. Emergency Vehicle Priority (Reward)
        emergency_vehicles = 0
        for vehicle_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(vehicle_id) == "emergency":
                emergency_vehicles += 1

        # 5. Phase Switching Penalty (Penalty)
        phase_switching_penalty = 1 if self.last_action != self.current_action else 0

        # 6. Fairness Penalty (Penalty)
        current_phase_duration = traci.trafficlight.getPhaseDuration("n0")
        target_duration = 30  # Target phase duration
        fairness_penalty = abs(current_phase_duration - target_duration)

        # Combine components with weights
        weights = {
            "waiting_time": -0.1,  # Penalize waiting time
            "queue_length": -0.05,  # Penalize queue length
            "throughput": 0.2,      # Reward throughput
            "emergency_vehicles": 1.0,  # Reward emergency vehicles
            "phase_switching": -0.2,  # Penalize frequent phase switching
            "fairness": -0.1  # Penalize unfair green time distribution
        }

        reward = (
            weights["waiting_time"] * total_waiting_time +
            weights["queue_length"] * total_queue_length +
            weights["throughput"] * throughput +
            weights["emergency_vehicles"] * emergency_vehicles +
            weights["phase_switching"] * phase_switching_penalty +
            weights["fairness"] * fairness_penalty
        )

        return reward

    def step(self, action):
        """Advance the simulation by one step."""
        self.current_action = action
        traci.trafficlight.setPhase("n0", action)  # Set the traffic light phase based on the action
        traci.simulationStep()
        self.simulation_time += 1
        if random.random() < 0.5:
            self.generate_random_traffic()

        next_state = self.get_state()
        reward = self.calculate_reward()
        done = self.simulation_time >= 3600  # End episode after 1 hour

        return next_state, reward, done

    def reset(self):
        """Reset the environment to start a new episode."""
        traci.close()
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        return self.get_state()

    def close(self):
        """Close SUMO connection."""
        traci.close()

# Run simulation
if __name__ == "__main__":
    env = TrafficEnv()
    try:
        for _ in range(3600):  # 1-hour simulation
            action = random.choice(env.action_space)  # Random action for testing
            next_state, reward, done = env.step(action)
            print(f"Reward: {reward}")
            if done:
                break
    finally:
        env.close()