import traci
import os
import sys
import random
import numpy as np
import pandas as pd
from sumolib import checkBinary

# SUMO configuration
sumo_config = "../sumo_files/sumo_config.sumocfg"
sumo_binary = checkBinary("sumo")
TL_ID = "TL"

# Routes mapping
routes = {
    "E2TL_EW_straight": ["E2TL", "TL2W"],
    "E2TL_NS_right": ["E2TL", "TL2N"],
    "E2TL_SW_left": ["E2TL", "TL2S"],
    "N2TL_NS_straight": ["N2TL", "TL2S"],
    "N2TL_WE_right": ["N2TL", "TL2W"],
    "N2TL_ES_left": ["N2TL", "TL2E"],
    "W2TL_WE_straight": ["W2TL", "TL2E"],
    "W2TL_EN_right": ["W2TL", "TL2N"],
    "W2TL_WS_left": ["W2TL", "TL2S"],
    "S2TL_SN_straight": ["S2TL", "TL2N"],
    "S2TL_NE_right": ["S2TL", "TL2E"],
    "S2TL_EW_left": ["S2TL", "TL2W"]
}

# Phase definitions
GREEN_PHASES = [0, 2, 4, 6]
YELLOW_PHASES = [1, 3, 5, 7]
GREEN_PHASE_DEFS = {
    4: {"name": "EW_straight_right", "edges": ["E2TL", "W2TL"], "lanes": [0, 1, 2], "outgoing": ["TL2W", "TL2E", "TL2N", "TL2S"]},
    0: {"name": "NS_straight_right", "edges": ["N2TL", "S2TL"], "lanes": [0, 1, 2], "outgoing": ["TL2S", "TL2N", "TL2E", "TL2W"]},
    6: {"name": "EW_left", "edges": ["E2TL", "W2TL"], "lanes": [3], "outgoing": ["TL2S", "TL2N"]},
    2: {"name": "NS_left", "edges": ["N2TL", "S2TL"], "lanes": [3], "outgoing": ["TL2E", "TL2W"]}
}

class EnvTest:
    def __init__(self):
        self.sumo_cfg = sumo_config
        self.GREEN_PHASE_DEFS = GREEN_PHASE_DEFS
        self.prev_waiting_active = 0
        self.prev_queue_active = 0
        self.prev_waiting_inactive = 0
        self.prev_queue_inactive = 0
        self.prev_exited = 0
        self.current_phase_index = 0
        self.phases = GREEN_PHASES
        self.last_phase = None
        self.simulation_time = 0
        self.current_difficulty = 1.0  # Fixed at max difficulty for testing
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

    def _lerp(self, start, end):
        return start + (end - start) * self.current_difficulty

    def generate_random_traffic(self):
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
        total = sum(v['prob'] for v in traffic_profile.values())
        profile = random.choices(list(traffic_profile), weights=[v['prob']/total for v in traffic_profile.values()], k=1)[0]
        config = traffic_profile[profile]

        if random.random() < config['spawn']:
            route_name, edges = random.choice(list(routes.items()))
            available_lanes = min(config['max_lanes'], traci.edge.getLaneNumber(edges[0]))
            depart_lane = str(random.randint(0, max(available_lanes-1, 0)))
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
        MAX_DUR = 90  # Max duration from RL env
        state = []

        for phase in GREEN_PHASES:
            if phase not in self.GREEN_PHASE_DEFS:
                state.extend([0.0, 0.0, 0.0])
                continue
            phase_info = self.GREEN_PHASE_DEFS[phase]
            controlled_lanes = [f"{edge}_{lane_index}" for edge in phase_info['edges'] for lane_index in phase_info['lanes']]
            queue_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in controlled_lanes]
            waiting_times = [traci.lane.getWaitingTime(l) for l in controlled_lanes]
            vehicle_counts = [traci.lane.getLastStepVehicleNumber(l) for l in controlled_lanes]
            total_queue = sum(queue_lengths)
            total_wait = sum(waiting_times)
            total_vehicles = sum(vehicle_counts)
            norm_queue = total_queue / MAX_Q
            norm_wait = total_wait / MAX_WAIT
            norm_veh = total_vehicles / MAX_VEH
            state.extend([round(norm_queue, 3), round(norm_wait, 3), round(norm_veh, 3)])

        norm_dur = 30 / MAX_DUR  # Fixed 30-second duration
        state.append(round(norm_dur, 3))
        state.extend(one_hot)
        return state

    def calculate_reward(self):
        current_phase = traci.trafficlight.getPhase(TL_ID)
        if current_phase not in self.GREEN_PHASE_DEFS:
            return 0
        phase_info = self.GREEN_PHASE_DEFS[current_phase]
        active_lanes = [f"{edge}_{lane_index}" for edge in phase_info['edges'] for lane_index in phase_info['lanes']]
        all_lanes = set(f"{edge}_{lane_index}" for info in self.GREEN_PHASE_DEFS.values() for edge in info['edges'] for lane_index in info['lanes'])
        inactive_lanes = list(all_lanes - set(active_lanes))

        cur_wait_active = sum(traci.lane.getWaitingTime(l) for l in active_lanes)
        cur_queue_active = sum(traci.lane.getLastStepHaltingNumber(l) for l in active_lanes)
        cur_wait_inactive = sum(traci.lane.getWaitingTime(l) for l in inactive_lanes)
        cur_queue_inactive = sum(traci.lane.getLastStepHaltingNumber(l) for l in inactive_lanes)
        cur_exited = traci.simulation.getArrivedNumber()
        throughput = cur_exited - self.prev_exited

        delta_wait_active = self.prev_waiting_active - cur_wait_active
        delta_queue_active = self.prev_queue_active - cur_queue_active
        delta_wait_inactive = self.prev_waiting_inactive - cur_wait_inactive
        delta_queue_inactive = self.prev_queue_inactive - cur_queue_inactive

        self.prev_waiting_active = cur_wait_active
        self.prev_queue_active = cur_queue_active
        self.prev_waiting_inactive = cur_wait_inactive
        self.prev_queue_inactive = cur_queue_inactive
        self.prev_exited = cur_exited

        MAX_WAIT = 600
        MAX_Q = 50
        MAX_T = 10

        reward = (
            +1.5 * (delta_wait_active / MAX_WAIT)
            +1.0 * (delta_queue_active / MAX_Q)
            +0.8 * (throughput / MAX_T)
            -1.0 * (delta_wait_inactive / MAX_WAIT)
            -0.5 * (delta_queue_inactive / MAX_Q)
        )
        return round(reward, 2)

    def step(self):
        phase = self.phases[self.current_phase_index]
        # Yellow phase transition
        if self.last_phase is not None and self.last_phase != phase:
            yellow_phase = YELLOW_PHASES[GREEN_PHASES.index(self.last_phase)]
            traci.trafficlight.setPhase(TL_ID, yellow_phase)
            for _ in range(5):
                traci.simulationStep()
                self.simulation_time += 1
                if random.random() < 0.5:
                    self.generate_random_traffic()

        # Green phase
        traci.trafficlight.setPhase(TL_ID, phase)
        for _ in range(30):  # Fixed 30-second duration
            traci.simulationStep()
            self.simulation_time += 1
            if random.random() < 0.5:
                self.generate_random_traffic()

        self.last_phase = phase
        self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)

        state = self.get_state()
        reward = self.calculate_reward()

        # Calculate metrics for info dictionary
        all_lanes = set(f"{edge}_{lane_index}" for info in self.GREEN_PHASE_DEFS.values() for edge in info['edges'] for lane_index in info['lanes'])
        wait_total = sum(traci.lane.getWaitingTime(l) for l in all_lanes)
        queue_total = sum(traci.lane.getLastStepHaltingNumber(l) for l in all_lanes)
        cur_exited = traci.simulation.getArrivedNumber()
        throughput = traci.simulation.getArrivedNumber()

        done = self.simulation_time >= 7200

        return state, reward, done, {
            'wait_total': wait_total,
            'queue_total': queue_total,
            'throughput': throughput
        }

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start([sumo_binary, "-c", self.sumo_cfg, "--step-length", "1", "--no-warnings"])
        self.prev_waiting_active = 0
        self.prev_queue_active = 0
        self.prev_waiting_inactive = 0
        self.prev_queue_inactive = 0
        self.prev_exited = 0
        self.current_phase_index = 0
        self.simulation_time = 0
        self.last_phase = None
        traci.trafficlight.setPhase(TL_ID, GREEN_PHASES[0])
        return self.get_state()

    def close(self):
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    env = EnvTest()
    num_episodes = 50
    episode_metrics = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        wait_total_list = []
        queue_total_list = []
        throughput_list = []

        while True:
            state, reward, done, info = env.step()
            total_reward += reward
            wait_total_list.append(info['wait_total'])
            queue_total_list.append(info['queue_total'])
            throughput_list.append(info['throughput'])
            if done:
                break

        episode_metrics.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'avg_wait_total': np.mean(wait_total_list),
            'avg_queue_total': np.mean(queue_total_list),
            'throughput': sum(throughput_list)
        })

        print(f"Episode {episode + 1}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg Waiting Time (All Lanes): {np.mean(wait_total_list):.2f} seconds")
        print(f"  Avg Queue Length (All Lanes): {np.mean(queue_total_list):.2f} vehicles")
        print(f"  Throughput: {sum(throughput_list)} vehicles")

    # Aggregate metrics
    avg_metrics = {
        'total_reward': np.mean([m['total_reward'] for m in episode_metrics]),
        'avg_wait_total': np.mean([m['avg_wait_total'] for m in episode_metrics]),
        'avg_queue_total': np.mean([m['avg_queue_total'] for m in episode_metrics]),
        'throughput': np.mean([m['throughput'] for m in episode_metrics])
    }

    print("\nAverage over 50 episodes:")
    print(f"  Avg Total Reward: {avg_metrics['total_reward']:.2f}")
    print(f"  Avg Waiting Time (All Lanes): {avg_metrics['avg_wait_total']:.2f} seconds")
    print(f"  Avg Queue Length (All Lanes): {avg_metrics['avg_queue_total']:.2f} vehicles")
    print(f"  Avg Throughput: {avg_metrics['throughput']:.2f} vehicles")

    # Save to CSV
    df = pd.DataFrame(episode_metrics)
    df.to_csv('env_test_results.csv', index=False)
    print("\nResults saved to 'env_test_results.csv'")

    env.close()