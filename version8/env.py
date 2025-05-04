# env.py (optimized for PPO with dynamic reward function)
import os
import random
import gym
import traci
from sumolib import checkBinary
import numpy as np

# SUMO configuration
sumo_config = "sumo_files/sumo_config.sumocfg"
sumo_binary = checkBinary("sumo")
TL_ID = "TL"

# Routes mapping
routes = {
    # East approaches
    "E2TL_EW_straight": ["E2TL", "TL2W"],
    "E2TL_NS_right":    ["E2TL", "TL2N"],
    "E2TL_SW_left":     ["E2TL", "TL2S"],
    # North approaches
    "N2TL_NS_straight": ["N2TL", "TL2S"],
    "N2TL_WE_right":    ["N2TL", "TL2W"],
    "N2TL_ES_left":     ["N2TL", "TL2E"],
    # West approaches
    "W2TL_WE_straight": ["W2TL", "TL2E"],
    "W2TL_EN_right":    ["W2TL", "TL2N"],
    "W2TL_WS_left":     ["W2TL", "TL2S"],
    # South approaches
    "S2TL_SN_straight": ["S2TL", "TL2N"],
    "S2TL_NE_right":    ["S2TL", "TL2E"],
    "S2TL_EW_left":     ["S2TL", "TL2W"]
}

# Phase indices
GREEN_PHASES  = [0, 2, 4, 6]
YELLOW_PHASES = [1, 3, 5, 7]

class TrafficEnv:
    def __init__(self):
        # Start SUMO
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1", "--no-warnings"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0
        self.prev_queues = 0
        self.prev_exited = 0

        # State & action spaces
        self.state_size  = 16
        self.phase_durations = [20, 30, 50]
        self.action_size = len(GREEN_PHASES) * len(self.phase_durations)  # 4 phases x 3 durations = 12
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_size)

        # Edge definitions
        self.incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        self.outgoing_edges = ["TL2W", "TL2E", "TL2N", "TL2S"]

        # Curriculum learning parameters
        self.episode_count = 0
        self.current_difficulty = 0
        self.base_traffic = {
            'low':    {'prob':0.8, 'spawn':0.1, 'max_lanes':1},
            'medium': {'prob':0.2, 'spawn':0.3, 'max_lanes':2},
            'high':   {'prob':0.0, 'spawn':0.5, 'max_lanes':3}
        }
        self.target_traffic = {
            'low':    {'prob':0.1, 'spawn':0.3, 'max_lanes':1},
            'medium': {'prob':0.3, 'spawn':0.6, 'max_lanes':3},
            'high':   {'prob':0.6, 'spawn':0.9, 'max_lanes':4}
        }

    def _lerp(self, start, end):
        return start + (end - start) * self.current_difficulty

    def generate_random_traffic(self):
        # Curriculum-based profile blending
        traffic_profile = {}
        for key in self.base_traffic:
            traffic_profile[key] = {
                'prob': self._lerp(self.base_traffic[key]['prob'], self.target_traffic[key]['prob']),
                'spawn': self._lerp(self.base_traffic[key]['spawn'], self.target_traffic[key]['spawn']),
                'max_lanes': min(
                    int(self._lerp(self.base_traffic[key]['max_lanes'], self.target_traffic[key]['max_lanes'])),
                    traci.edge.getLaneNumber("E2TL")
                )
            }
        # Normalize & choose profile
        total = sum(v['prob'] for v in traffic_profile.values())
        profile = random.choices(list(traffic_profile), weights=[v['prob']/total for v in traffic_profile.values()], k=1)[0]
        config = traffic_profile[profile]

        if random.random() < config['spawn']:
            route_name, edges = random.choice(list(routes.items()))
            available_lanes = min(config['max_lanes'], traci.edge.getLaneNumber(edges[0]))
            depart_lane  = str(random.randint(0, max(available_lanes-1, 0)))
            depart_speed = random.uniform(5, 12.8)
            rid = f"{route_name}_{self.simulation_time}"
            traci.route.add(rid, edges)
            vid = f"veh_{self.simulation_time}"
            traci.vehicle.add(vid, rid, typeID="car", departLane=depart_lane, departSpeed=str(depart_speed))

    def get_state(self):
        state = []
        # Phase one-hot
        current_phase = traci.trafficlight.getPhase(TL_ID)
        state += [1 if current_phase==p else 0 for p in GREEN_PHASES]
        # Metrics normalization
        MAX_VEH=200; MAX_WAIT=3600; MAX_Q=200
        # Vehicle counts
        state += [traci.edge.getLastStepVehicleNumber(e)/MAX_VEH for e in self.incoming_edges]
        # Waiting times
        state += [traci.edge.getWaitingTime(e)/MAX_WAIT for e in self.incoming_edges]
        # Queue lengths
        state += [traci.edge.getLastStepHaltingNumber(e)/MAX_Q for e in self.incoming_edges]
        return np.array(state, dtype=np.float32)

    def calculate_reward(self):
        cur_wait = sum(traci.edge.getWaitingTime(e) for e in self.incoming_edges)
        cur_q = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.incoming_edges)
        cur_ex = traci.simulation.getArrivedNumber()
        throughput = cur_ex - self.prev_exited

        # Deltas
        delta_wait = self.prev_waiting - cur_wait
        delta_q = self.prev_queues - cur_q

        # Dynamic normalization based on observed max values
        max_wait = max(self.prev_waiting, cur_wait, 1)
        max_q = max(self.prev_queues, cur_q, 1)
        max_t = max(throughput, 1)

        # Balanced weights and congestion penalty
        reward = (1.0 * (delta_wait / max_wait) +
                  1.0 * (delta_q / max_q) +
                  0.5 * (throughput / max_t) -
                  0.2 * (cur_q / max_q if cur_q > 20 else 0))  # Penalize large queues

        # Save for next
        self.prev_waiting = cur_wait
        self.prev_queues = cur_q
        self.prev_exited = cur_ex

        return round(reward, 2)

    def step(self, action):
        # Decode action -> phase + duration
        phase_idx    = action // len(self.phase_durations)
        duration_idx = action %  len(self.phase_durations)
        self.current_action = GREEN_PHASES[phase_idx]
        selected_duration  = self.phase_durations[duration_idx]
        # Yellow transition
        if self.last_action is not None and self.last_action!=self.current_action:
            y = YELLOW_PHASES[GREEN_PHASES.index(self.last_action)]
            traci.trafficlight.setPhase(TL_ID, y)
            for _ in range(5): traci.simulationStep(); self.simulation_time+=1; self.generate_random_traffic()
        # Green phase
        traci.trafficlight.setPhase(TL_ID, self.current_action)
        for _ in range(selected_duration):
            traci.simulationStep(); self.simulation_time+=1
            if random.random()<0.5: self.generate_random_traffic()
        self.last_action = self.current_action
        return self.get_state(), self.calculate_reward(), self.simulation_time>=7200

    def reset(self):
        traci.close()
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0
        self.prev_queues = 0
        traci.trafficlight.setPhase(TL_ID, GREEN_PHASES[0])
        self.episode_count += 1
        self.current_difficulty = min(self.episode_count/2000, 1.0)
        return self.get_state()

    def close(self):
        traci.close()