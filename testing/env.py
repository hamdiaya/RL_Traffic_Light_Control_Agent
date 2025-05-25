import os
import random
import gym
import traci
from sumolib import checkBinary
import numpy as np
import csv

# SUMO configuration
sumo_config = "sumo_files/sumo_config.sumocfg"  # Ensure this points to your SUMO config
sumo_binary = checkBinary("sumo") 
TL_ID = "TL"  # Traffic light ID in your net

# Routes dictionary (unchanged)
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

GREEN_PHASES = [0, 2, 4, 6]
YELLOW_PHASES = [1, 3, 5, 7]

class TrafficEnv:
    def __init__(self):
        traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1", "--no-warnings"])
        self.simulation_time = 0
        self.last_action = None
        self.current_action = None
        self.prev_waiting = 0
        self.prev_queues = 0
        self.prev_exited = 0
        
        self.state_size = 16
        self.action_size = 4
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_size)

        self.incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
        self.outgoing_edges = ["TL2W", "TL2E", "TL2N", "TL2S"]

        self.episode_count = 0
        self.current_difficulty = 0.5

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
        total_prob = sum(v['prob'] for v in traffic_profile.values())
        profile = random.choices(
            list(traffic_profile.keys()),
            weights=[v['prob'] / total_prob for v in traffic_profile.values()],
            k=1
        )[0]
        config = traffic_profile[profile]
        if random.random() < config['spawn']:
            route_name, edges = random.choice(list(routes.items()))
            available_lanes = min(config['max_lanes'], traci.edge.getLaneNumber(edges[0]))
            depart_lane = str(random.randint(0, available_lanes - 1))
            depart_speed = random.uniform(5, 12.8)
            route_id = f"{route_name}_{self.simulation_time}"
            traci.route.add(route_id, edges)
            vehicle_id = f"veh_{self.simulation_time}"
            traci.vehicle.add(vehicle_id, route_id, typeID="car", departLane=depart_lane, departSpeed=str(depart_speed))

    def _lerp(self, start, end):
        return start + (end - start) * self.current_difficulty

    def get_state(self):
        state = []
        current_phase = traci.trafficlight.getPhase(TL_ID)
        phase_encoding = [1 if current_phase == p else 0 for p in GREEN_PHASES]
        state.extend(phase_encoding)
        MAX_VEHICLES = 200
        MAX_WAITING = 3600
        MAX_QUEUE = 200
        state.extend([traci.edge.getLastStepVehicleNumber(e) / MAX_VEHICLES for e in self.incoming_edges])
        state.extend([traci.edge.getWaitingTime(e) / MAX_WAITING for e in self.incoming_edges])
        state.extend([traci.edge.getLastStepHaltingNumber(e) / MAX_QUEUE for e in self.incoming_edges])
        return np.array(state, dtype=np.float32)

    def calculate_reward(self):
        current_waiting = sum(traci.edge.getWaitingTime(e) for e in self.incoming_edges)
        current_queues = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.incoming_edges)
        current_exited = traci.simulation.getArrivedNumber()
        throughput = current_exited - self.prev_exited
        self.prev_exited = current_exited
        delta_waiting = self.prev_waiting - current_waiting
        delta_queues = self.prev_queues - current_queues
        self.prev_waiting = current_waiting
        self.prev_queues = current_queues

        MAX_WAITING = 3600
        MAX_QUEUE = 200
        MAX_THROUGHPUT = 60

        reward = (
            1.5 * (delta_waiting / MAX_WAITING) +
            1.0 * (delta_queues / MAX_QUEUE) +
            0.8 * (throughput / MAX_THROUGHPUT)
        )
        return round(reward, 2)

    def step(self, action):
        self.current_action = GREEN_PHASES[action]
        if self.last_action is not None and self.last_action != self.current_action:
            yellow_phase = YELLOW_PHASES[GREEN_PHASES.index(self.last_action)]
            traci.trafficlight.setPhase(TL_ID, yellow_phase)
            for _ in range(5):
                traci.simulationStep()
                self.simulation_time += 1
                self.generate_random_traffic()
        traci.trafficlight.setPhase(TL_ID, self.current_action)
        for _ in range(30):
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
        self.prev_waiting = 0
        self.prev_queues = 0
        self.prev_exited = 0
        traci.trafficlight.setPhase(TL_ID, GREEN_PHASES[0])
        self.episode_count += 1
        self.current_difficulty = 0.5
        return self.get_state()

    def close(self):
        traci.close()

class DefaultTrafficLightTester:
    def __init__(self, episodes=20, csv_file="default_results.csv"):
        self.episodes = episodes
        self.csv_file = csv_file

    def run(self):
        fieldnames = ['episode', 'step', 'phase', 'reward', 'waiting_time', 'queue_length', 'throughput']
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ep in range(1, self.episodes + 1):
                traci.start([sumo_binary, "-c", sumo_config, "--step-length", "1"])
                print(f"Starting episode {ep} with default traffic light controller...")
                step = 0
                prev_waiting = 0
                prev_queues = 0
                prev_exited = 0
                done = False

                while not done:
                    current_phase = traci.trafficlight.getPhase(TL_ID)
                    # Step simulation one step at a time
                    traci.simulationStep()
                    step += 1

                    # Collect metrics
                    incoming_edges = ["E2TL", "N2TL", "S2TL", "W2TL"]
                    current_waiting = sum(traci.edge.getWaitingTime(e) for e in incoming_edges)
                    current_queues = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming_edges)
                    current_exited = traci.simulation.getArrivedNumber()
                    throughput = current_exited - prev_exited

                    # Reward-like metric (same as env)
                    delta_waiting = prev_waiting - current_waiting
                    delta_queues = prev_queues - current_queues
                    MAX_WAITING = 3600
                    MAX_QUEUE = 200
                    MAX_THROUGHPUT = 60
                    reward = (
                        1.5 * (delta_waiting / MAX_WAITING) +
                        1.0 * (delta_queues / MAX_QUEUE) +
                        0.8 * (throughput / MAX_THROUGHPUT)
                    )
                    reward = round(reward, 2)

                    # Save previous metrics for next step
                    prev_waiting = current_waiting
                    prev_queues = current_queues
                    prev_exited = current_exited

                    writer.writerow({
                        'episode': ep,
                        'step': step,
                        'phase': current_phase,
                        'reward': reward,
                        'waiting_time': current_waiting,
                        'queue_length': current_queues,
                        'throughput': throughput
                    })

                    # Define done condition (e.g. 7200 steps)
                    if step >= 7200:
                        done = True

                traci.close()
                print(f"Episode {ep} done and data saved.")

if __name__ == "__main__":
    # Run default traffic light controller for 20 episodes and save results
    tester = DefaultTrafficLightTester(episodes=20)
    tester.run()
