import os
import random
import traci
from sumolib import checkBinary

# SUMO configuration
sumo_config = "sumo_files/sumo_config.sumocfg"
sumo_binary = checkBinary("sumo-gui")  # Use "sumo-gui" for visualization

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
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
        self.simulation_time = 0
        self.last_action = None  # Track the last action for phase switching penalty
        self.current_action = None  # Track the current action for phase switching penalty

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

    def get_vehicle_direction(self, veh_id):
        """Determine vehicle movement direction based on new lane assignments."""
        route = traci.vehicle.getRoute(veh_id)
        if len(route) < 2:
            return "unknown"

        current_edge = traci.vehicle.getRoadID(veh_id)
        next_edge = route[1]

        # Define turns for each incoming lane
        direction_map = {
            "E1_in": {"E2_out": "straight", "E3_out": "left", "E4_out": "right"},
            "E3_in": {"E1_out": "left", "E2_out": "right", "E4_out": "straight"},
            "E4_in": {"E1_out": "right", "E2_out": "left", "E3_out": "straight"},
            "E2_in": {"E1_out": "straight", "E3_out": "right", "E4_out": "left"},
        }

        return direction_map.get(current_edge, {}).get(next_edge, "unknown")

    def get_simulation_info(self):
        """Retrieve SUMO state data."""
        observation = {
            "traffic_light_state": traci.trafficlight.getPhase("n0"),
            "vehicles": [],
        }

        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            position = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            route = traci.vehicle.getRoute(veh_id)
            next_edge = route[1] if len(route) > 1 else "Destination"
            direction = self.get_vehicle_direction(veh_id)

            observation["vehicles"].append({
                "id": veh_id,
                "position": position,
                "speed": speed,
                "waiting_time": waiting_time,
                "next_edge": next_edge,
                "direction": direction,
            })

        return observation

    def calculate_reward(self):
        """Calculate the reward based on traffic conditions."""
        # Define the edges to monitor (update these based on your SUMO network)
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
        # Get the current phase duration (a single float value)
        current_phase_duration = traci.trafficlight.getPhaseDuration("n0")
        # For fairness, compare the current phase duration to a target duration (e.g., 30 seconds)
        target_duration = 30  # Adjust this based on your traffic light configuration
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

    def step(self):
        """Advance the simulation."""
        traci.simulationStep()
        self.simulation_time += 1
        if random.random() < 0.5:
            self.generate_random_traffic()

        observation = self.get_simulation_info()
        reward = self.calculate_reward()

        return observation, reward

    def close(self):
        """Close SUMO connection."""
        traci.close()

# Run simulation
if __name__ == "__main__":
    env = TrafficEnv()
    try:
        for _ in range(3600):  # 1-hour simulation
            observation, reward = env.step()
            print(f"Reward: {reward}")
            
    finally:
        env.close()