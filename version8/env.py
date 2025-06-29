# env.py (updated for dynamic phase durations)
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

import logging
import sys



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

# Lanes definitions
GREEN_PHASE_DEFS = {
    4: {"name": "EW_straight_right", "edges": ["E2TL", "W2TL"], "lanes": [0, 1, 2], "outgoing": ["TL2W", "TL2E", "TL2N", "TL2S"]},
    0: {"name": "NS_straight_right", "edges": ["N2TL", "S2TL"], "lanes": [0, 1, 2], "outgoing": ["TL2S", "TL2N", "TL2E", "TL2W"]},
    6: {"name": "EW_left", "edges": ["E2TL", "W2TL"], "lanes": [3], "outgoing": ["TL2S", "TL2N"]},
    2: {"name": "NS_left", "edges": ["N2TL", "S2TL"], "lanes": [3], "outgoing": ["TL2E", "TL2W"]}
}


class TrafficEnv:
    def __init__(self):
        # Start SUMO
        print("Attempting to start SUMO...")
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1", "--no-warnings"])
        print("SUMO started successfully.")
              
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        
        self.prev_waiting_active = 0
        self.prev_queue_active = 0
        self.prev_waiting_inactive = 0
        self.prev_queue_inactive = 0
        self.prev_exited = 0
        
        # State & action spaces
        self.state_size = 17
        # Define discrete duration options 
        self.phase_durations = [20, 30, 40, 50, 60, 70, 80, 90]
        self.action_size = len(GREEN_PHASES) * len(self.phase_durations)  
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_size)

        # Edge definitions
        self.incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        self.outgoing_edges = ["TL2W", "TL2E", "TL2N", "TL2S"]
        
        
        self.GREEN_PHASE_DEFS = GREEN_PHASE_DEFS


        # Curriculum learning parameters
        self.episode_count = 0
        self.current_difficulty = 0.0
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
        current_phase = traci.trafficlight.getPhase(TL_ID)
        one_hot = [1 if i == GREEN_PHASES.index(current_phase) else 0 for i in range(len(GREEN_PHASES))]
    
        MAX_Q = 50
        MAX_WAIT = 600
        MAX_VEH = 10
        MAX_DUR = max(self.phase_durations)
    
        state = []
    
        # Get metrics for all GREEN_PHASES
        for phase in GREEN_PHASES:
            if phase not in self.GREEN_PHASE_DEFS:
                state.extend([0.0, 0.0, 0.0])
                continue
    
            phase_info = self.GREEN_PHASE_DEFS[phase]
            controlled_lanes = []
    
            for edge in phase_info['edges']:
                for lane_index in phase_info['lanes']:
                    controlled_lanes.append(f"{edge}_{lane_index}")
    
            queue_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in controlled_lanes]
            waiting_times = [traci.lane.getWaitingTime(l) for l in controlled_lanes]
            vehicle_counts = [traci.lane.getLastStepVehicleNumber(l) for l in controlled_lanes]
    
            total_queue = sum(queue_lengths)
            total_wait = sum(waiting_times)
            total_vehicles = sum(vehicle_counts)
    
            norm_queue = total_queue / MAX_Q
            norm_wait = total_wait / MAX_WAIT
            norm_veh = total_vehicles / MAX_VEH
    
            state.extend([
                round(norm_queue, 3),
                round(norm_wait, 3),
                round(norm_veh, 3),
            ])
    
        # Add normalized duration of current phase
        norm_dur = (self.current_duration / MAX_DUR) if hasattr(self, "current_duration") else 0.0
        state.append(round(norm_dur, 3))
        state.extend(one_hot)
        return state


    def calculate_reward(self):

        current_phase = traci.trafficlight.getPhase(TL_ID)
        if current_phase not in self.GREEN_PHASE_DEFS:
            return 0  # Safety check
    
        phase_info = self.GREEN_PHASE_DEFS[current_phase]
    
        # Active lanes (controlled by current green phase)
        active_lanes = [f"{edge}_{lane_index}"
                        for edge in phase_info['edges']
                        for lane_index in phase_info['lanes']]
    
        # All lanes defined in all phases
        all_lanes = set()
        for info in self.GREEN_PHASE_DEFS.values():
            for edge in info['edges']:
                for lane_index in info['lanes']:
                    all_lanes.add(f"{edge}_{lane_index}")
    
        inactive_lanes = list(all_lanes - set(active_lanes))
    
        # Get metrics
        cur_wait_active = sum(traci.lane.getWaitingTime(l) for l in active_lanes)
        cur_queue_active = sum(traci.lane.getLastStepHaltingNumber(l) for l in active_lanes)
    
        cur_wait_inactive = sum(traci.lane.getWaitingTime(l) for l in inactive_lanes)
        cur_queue_inactive = sum(traci.lane.getLastStepHaltingNumber(l) for l in inactive_lanes)
    
        cur_exited = traci.simulation.getArrivedNumber()
        throughput = cur_exited - self.prev_exited
    
        # Calculate deltas
        delta_wait_active = self.prev_waiting_active - cur_wait_active
        delta_queue_active = self.prev_queue_active - cur_queue_active
        delta_wait_inactive = self.prev_waiting_inactive - cur_wait_inactive
        delta_queue_inactive = self.prev_queue_inactive - cur_queue_inactive
    
        # Update history
        self.prev_waiting_active = cur_wait_active
        self.prev_queue_active = cur_queue_active
        self.prev_waiting_inactive = cur_wait_inactive
        self.prev_queue_inactive = cur_queue_inactive
        self.prev_exited = cur_exited
    
        # Normalization constants
        MAX_WAIT = 600
        MAX_Q = 50
        MAX_T = 10
    
        # Compute reward
        reward = (
            +1.5 * (delta_wait_active / MAX_WAIT)
            +1.0 * (delta_queue_active / MAX_Q)
            +0.8 * (throughput / MAX_T)
            -1.0 * (delta_wait_inactive / MAX_WAIT)
            -0.5 * (delta_queue_inactive / MAX_Q)
        )
    
        return round(reward, 2)

    
    def step(self, action):
        # Decode action -> phase + duration
        phase_idx = action // len(self.phase_durations)
        duration_idx = action % len(self.phase_durations)
       
        self.current_action = GREEN_PHASES[phase_idx]
        selected_duration = self.phase_durations[duration_idx]
        self.current_duration = selected_duration  # ✅ Store for state
    
        # Yellow transition
        if self.last_action is not None and self.last_action != self.current_action:
            y = YELLOW_PHASES[GREEN_PHASES.index(self.last_action)]
            traci.trafficlight.setPhase(TL_ID, y)
            for _ in range(5):
                traci.simulationStep()
                self.simulation_time += 1
                self.generate_random_traffic()
    
        # Green phase
        traci.trafficlight.setPhase(TL_ID, self.current_action)
        for _ in range(selected_duration):
            traci.simulationStep()
            self.simulation_time += 1
            if random.random() < 0.5:
                self.generate_random_traffic()
    
        self.last_action = self.current_action
    
        # Get state and reward
        state = self.get_state()
        reward = self.calculate_reward()
    
        
        return state, reward, self.simulation_time >= 7200
    

    def reset(self):
        traci.close()
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1", "--no-warnings"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = GREEN_PHASES[0]
        self.prev_waiting_active = 0
        self.prev_queue_active = 0
        self.prev_waiting_inactive = 0
        self.prev_queue_inactive = 0
        self.prev_exited = 0
        self.current_duration = 0 
        traci.trafficlight.setPhase(TL_ID, GREEN_PHASES[0])
        self.episode_count += 1
        self.current_difficulty = 0.5
  
        return self.get_state()

    def close(self):
        if traci.isLoaded():
           traci.close()


    
