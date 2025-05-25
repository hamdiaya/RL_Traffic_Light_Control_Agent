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
        traci.start([sumo_binary, "-gui", sumo_config, "--step-length", "1", "--no-warnings"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0  # New tracking variable
        self.prev_queues = 0   # New tracking variable
        self.prev_exited = 0
        # Spaces remain same structure
        self.state_size = 16
        self.action_size = 4
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_size)

        # Define incoming and outgoing edges as class-level attributes
        self.incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        self.outgoing_edges = ["TL2W", "TL2E", "TL2N", "TL2S"]

        self.episode_count = 0  # Track curriculum progression
        self.current_difficulty = 0.0
        
        # Curriculum parameters
        self.base_traffic = {
            'low': {'prob': 0.8, 'spawn': 0.1, 'max_lanes': 1},
            'medium': {'prob': 0.2, 'spawn': 0.3, 'max_lanes': 2},
            'high': {'prob': 0.0, 'spawn': 0.5, 'max_lanes': 3}
        }
        self.target_traffic = {
            'low': {'prob': 0.1, 'spawn': 0.3, 'max_lanes': 1},
            'medium': {'prob': 0.3, 'spawn': 0.6, 'max_lanes': 3},
            'high': {'prob': 0.6, 'spawn': 0.9, 'max_lanes': 4}
        }

        


    def generate_random_traffic(self):
                # Curriculum-based vehicle generation
        traffic_profile = {}
        for key in self.base_traffic:
            traffic_profile[key] = {
                'prob': self._lerp(
                    self.base_traffic[key]['prob'], 
                    self.target_traffic[key]['prob']
                ),
                'spawn': self._lerp(
                    self.base_traffic[key]['spawn'],
                    self.target_traffic[key]['spawn']
                ),
                'max_lanes': min(  # Ensure we don't exceed physical lanes
                    int(self._lerp(
                        self.base_traffic[key]['max_lanes'],
                        self.target_traffic[key]['max_lanes']
                    )),
                    traci.edge.getLaneNumber("E2TL")  # Physical lane limit
                )
            }
        
        # Normalize probabilities and select profile
        total_prob = sum(v['prob'] for v in traffic_profile.values())
        profile = random.choices(
            list(traffic_profile.keys()),
            weights=[v['prob'] / total_prob for v in traffic_profile.values()],
            k=1
        )[0]
        config = traffic_profile[profile]
        
        # Generate vehicle if spawn check passes
        if random.random() < config['spawn']:
            route_name, edges = random.choice(list(routes.items()))
            
            # Ensure we don't request non-existent lanes
            available_lanes = min(config['max_lanes'], traci.edge.getLaneNumber(edges[0]))
            depart_lane = str(random.randint(0, available_lanes - 1))
            
            depart_speed = random.uniform(5, 12.8)  # Network speed limits
        
            route_id = f"{route_name}_{self.simulation_time}"
            traci.route.add(route_id, edges)
        
            vehicle_id = f"veh_{self.simulation_time}"
            traci.vehicle.add(
                vehicle_id, route_id, 
                typeID="car", 
                departLane=depart_lane,
                departSpeed=str(depart_speed)
        )
        
    def _lerp(self, start, end):
            """Linear interpolation between start and end values"""
            return start + (end - start) * self.current_difficulty
                
    def get_state(self):
      """State observation aligned with reward normalization"""
      state = []
  
      # Phase one-hot encoding (helps learning cause-effect)
      current_phase = traci.trafficlight.getPhase(TL_ID)
      phase_encoding = [1 if current_phase == p else 0 for p in GREEN_PHASES]
      state.extend(phase_encoding)
  
      # Incoming traffic metrics
      incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
  
      # Normalization constants (match reward)
      MAX_VEHICLES = 200      # total across edges
      MAX_WAITING = 3600      # 1 hour of waiting time total
      MAX_QUEUE = 200         # Max 50 vehicles per edge × 4
  
      # Vehicle count per edge
      state.extend([
          traci.edge.getLastStepVehicleNumber(e) / MAX_VEHICLES
          for e in incoming_edges
      ])
  
      # Waiting time per edge
      state.extend([
          traci.edge.getWaitingTime(e) / MAX_WAITING
          for e in incoming_edges
      ])
  
      # Queue length per edge
      state.extend([
          traci.edge.getLastStepHaltingNumber(e) / MAX_QUEUE
          for e in incoming_edges
      ])
  
      return np.array(state, dtype=np.float32)

    
    def calculate_reward(self):
        """Optimized reward function for SUMO traffic light control"""
        incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        outgoing_edges = ["TL2W", "TL2E", "TL2N", "TL2S"]
        
        # 1. Core Traffic Metrics
        current_waiting = sum(traci.edge.getWaitingTime(e) for e in incoming_edges)
        current_queues = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)
        
        # 2. Throughput Calculation (improved)
        current_exited = traci.simulation.getArrivedNumber()
        throughput = current_exited - self.prev_exited 
        self.prev_exited = current_exited
        
        
        # 4. Improvement Deltas
        delta_waiting = self.prev_waiting - current_waiting  # Positive if improved
        delta_queues = self.prev_queues - current_queues
        
        # Store for next step
        self.prev_waiting = current_waiting
        self.prev_queues = current_queues
        
        # 5. Normalization Factors (tuned for your 4-lane intersection)
        MAX_WAITING = 3600    # 1 hour max waiting (reasonable upper bound)
        MAX_QUEUE = 200       # 4 lanes × 50 vehicles/lane
        MAX_THROUGHPUT = 60   # ~1 vehicle/second max throughput
        
        
        # 6. Reward Components
        reward = (
            + 1.5 * (delta_waiting / MAX_WAITING)          # Waiting time improvement
            + 1.0 * (delta_queues / MAX_QUEUE)             # Queue improvement
            + 0.8 * (throughput / MAX_THROUGHPUT)          # Throughput bonus
           # - 0.5 * (current_waiting / MAX_WAITING)        # Absolute waiting penalty
           # - 0.4 * (current_queues / MAX_QUEUE)           # Absolute queue penalty
            
                                  
        )
        
       
        
        return round(reward, 2)
                                              

    def step(self, action):
        """Updated phase control with TL_ID"""
        self.current_action = GREEN_PHASES[action]

        # Yellow phase transition logic
        if self.last_action is not None and self.last_action != self.current_action:
            yellow_phase = YELLOW_PHASES[GREEN_PHASES.index(self.last_action)]
            traci.trafficlight.setPhase(TL_ID, yellow_phase)
            for _ in range(5):  # Maintain 7-step yellow duration
                traci.simulationStep()
                self.simulation_time += 1
                self.generate_random_traffic()

        # Set green phase
        traci.trafficlight.setPhase(TL_ID, self.current_action)
        for _ in range(30):  # 5-step green phase
            traci.simulationStep()
            self.simulation_time += 1
            if random.random() < 0.5:
                self.generate_random_traffic()

        self.last_action = self.current_action
        return self.get_state(), self.calculate_reward(), self.simulation_time >= 7200

    def reset(self):
        traci.close()
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0  # Reset tracking variables
        self.prev_queues = 0   # Reset tracking variables
        traci.trafficlight.setPhase(TL_ID, GREEN_PHASES[0])

        self.episode_count += 1
        # Update difficulty (full difficulty at episode 2000)
        self.current_difficulty = min(self.episode_count / 2000, 1.0)
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