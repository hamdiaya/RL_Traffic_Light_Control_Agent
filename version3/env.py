import os
import random
import gym
import traci
from sumolib import checkBinary
import numpy as np

# SUMO configuration
sumo_config = "sumo_files/sumo_config.sumocfg"  # Ensure this points to new net file
sumo_binary = checkBinary("sumo") 
TL_ID = "TL"  # Updated traffic light ID

# Updated routes matching the new network edges
routes = {
    # East approaches
    "E2TL_EW_straight": ["E2TL", "TL2W"],
    "E2TL_NS_right": ["E2TL", "TL2N"],
    "E2TL_SW_left": ["E2TL", "TL2S"],
    
    # North approaches
    "N2TL_NS_straight": ["N2TL", "TL2S"],
    "N2TL_WE_right": ["N2TL", "TL2W"],
    "N2TL_ES_left": ["N2TL", "TL2E"],
    
    # West approaches
    "W2TL_WE_straight": ["W2TL", "TL2E"],
    "W2TL_EN_right": ["W2TL", "TL2N"],
    "W2TL_WS_left": ["W2TL", "TL2S"],
    
    # South approaches
    "S2TL_SN_straight": ["S2TL", "TL2N"],
    "S2TL_NE_right": ["S2TL", "TL2E"],
    "S2TL_EW_left": ["S2TL", "TL2W"]
}

GREEN_PHASES = [0, 2, 4, 6]  # Same phase indices as original
YELLOW_PHASES = [1, 3, 5, 7]

class TrafficEnv:
    def __init__(self):
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1", "--no-warnings"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0  # New tracking variable
        self.prev_queues = 0   # New tracking variable
        
        # Spaces remain same structure
        self.state_size = 16
        self.action_size = 4
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_size)

    def generate_random_traffic(self):
        """Generate vehicles with proper lane selection for new edges"""
        traffic_profiles = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.2,
        }

        profile = random.choices(list(traffic_profiles.keys()), 
                              weights=list(traffic_profiles.values()), k=1)[0]
        vehicle_probability = {"low": 0.2, "medium": 0.5, "high": 0.8}[profile]

        if random.random() < vehicle_probability:
            route_name, edges = random.choice(list(routes.items()))
            
            # Dynamic lane selection based on actual edge lanes
            edge_data = traci.edge.getLaneNumber(edges[0])
            depart_lane = str(random.randint(0, edge_data - 1))  # Now handles 1-3 lanes
            
            depart_speed = random.uniform(5, 11.1)  # Matches network speed limits

            route_id = f"{route_name}_{self.simulation_time}"
            traci.route.add(route_id, edges)

            vehicle_id = f"veh_{self.simulation_time}"
            traci.vehicle.add(
                vehicle_id, route_id, 
                typeID="car", 
                departLane=depart_lane,
                departSpeed=str(depart_speed))
                
    def get_state(self):
        """State observation with new edge IDs and normalization"""
        state = []
        
        # Phase one-hot encoding
        current_phase = traci.trafficlight.getPhase(TL_ID)
        phase_encoding = [1 if current_phase == p else 0 for p in GREEN_PHASES]
        state.extend(phase_encoding)

        # Normalized metrics for new incoming edges
        incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        
        # Vehicle count (normalized by max 40 vehicles)
        state.extend([traci.edge.getLastStepVehicleNumber(e)/40 for e in incoming_edges])
        
        # Waiting time (normalized by 2 minutes)
        state.extend([traci.edge.getWaitingTime(e)/120 for e in incoming_edges])
        
        # Queue length (normalized by 20 vehicles)
        state.extend([traci.edge.getLastStepHaltingNumber(e)/20 for e in incoming_edges])

        return np.array(state, dtype=np.float32)

    def calculate_reward(self):
        """Improved reward function with delta calculations"""
        incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        outgoing_edges = ["TL2W", "TL2E", "TL2N", "TL2S"]

        # Current metrics
        current_waiting = sum(traci.edge.getWaitingTime(e) for e in incoming_edges)
        current_queues = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)
        throughput = sum(traci.edge.getLastStepVehicleNumber(e) for e in outgoing_edges)

        # Calculate deltas from previous step
        delta_waiting = self.prev_waiting - current_waiting  # Positive if improved
        delta_queues = self.prev_queues - current_queues    # Positive if improved

        # Store current values for next calculation
        self.prev_waiting = current_waiting
        self.prev_queues = current_queues

        # Balanced reward components
        reward = (
            + 0.4 * delta_waiting        # Reward for reducing waiting
            + 0.3 * delta_queues         # Reward for reducing queues
            + 0.2 * throughput           # Reward for throughput
            - 0.1 * current_waiting/100  # Small penalty for absolute waiting
        )

        return round(reward, 2)

    def step(self, action):
        """Updated phase control with TL_ID"""
        self.current_action = GREEN_PHASES[action]

        # Yellow phase transition logic
        if self.last_action is not None and self.last_action != self.current_action:
            yellow_phase = YELLOW_PHASES[GREEN_PHASES.index(self.last_action)]
            traci.trafficlight.setPhase(TL_ID, yellow_phase)
            for _ in range(10):  # Maintain 7-step yellow duration
                traci.simulationStep()
                self.simulation_time += 1
                self.generate_random_traffic()

        # Set green phase
        traci.trafficlight.setPhase(TL_ID, self.current_action)
        for _ in range(60):  # 5-step green phase
            traci.simulationStep()
            self.simulation_time += 1
            if random.random() < 0.5:
                self.generate_random_traffic()

        self.last_action = self.current_action
        return self.get_state(), self.calculate_reward(), self.simulation_time >= 3600

    def reset(self):
        traci.close()
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0  # Reset tracking variables
        self.prev_queues = 0   # Reset tracking variables
        traci.trafficlight.setPhase(TL_ID, GREEN_PHASES[0])
        return self.get_state()

    def close(self):
        traci.close()

# Test run remains similar
if __name__ == "__main__":
    env = TrafficEnv()
    try:
        state = env.reset()
        while True:
            action = env.action_space.sample()
            next_state, reward, done = env.step(action)
            print(f"State: {state} | Reward: {reward}")
            state = next_state
            if done:
                break
    finally:
        env.close()