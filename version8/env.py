# env.py
import numpy as np

class TrafficEnv:
    def __init__(self):
        self.state_size = 16  # Adjust based on your actual observation shape
        self.action_size = 4  # Number of discrete traffic light actions

        self.max_steps = 7200
        self.current_step = 0

        # SUMO setup
        self._init_sumo()

    def _init_sumo(self):
        import traci
        import sumolib
        import os

        # Use SUMO_BINARY env variable or default path
        self.sumo_binary = os.environ.get("SUMO_BINARY", "sumo")
        self.sumo_config = "your_network.sumocfg"  # Replace with your file
        self.sumo_cmd = [self.sumo_binary, "-c", self.sumo_config, "--start"]

        traci.start(self.sumo_cmd)
        self.traci = traci

    def reset(self):
        self.traci.close()
        self._init_sumo()
        self.current_step = 0

        return self._get_observation()

    def step(self, action):
        self._apply_action(action)

        reward = self._compute_reward()
        self.traci.simulationStep()
        self.current_step += 1

        done = self.current_step >= self.max_steps
        obs = self._get_observation()

        return obs, reward, done

    def _apply_action(self, action):
        tls_id = self.traci.trafficlight.getIDList()[0]
        phases = self.traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
        phase_index = min(action, len(phases) - 1)
        self.traci.trafficlight.setPhase(tls_id, phase_index)

    def _get_observation(self):
        # Example: vehicle count, speed, waiting time on 4 approaches
        obs = []
        lanes = self.traci.lane.getIDList()
        for lane in lanes:
            obs.append(self.traci.lane.getLastStepVehicleNumber(lane))
            obs.append(self.traci.lane.getLastStepMeanSpeed(lane))
        return np.array(obs[:self.state_size], dtype=np.float32)

    def _compute_reward(self):
        # Example: negative total waiting time
        total_wait = 0
        for veh_id in self.traci.vehicle.getIDList():
            total_wait += self.traci.vehicle.getWaitingTime(veh_id)
        return -total_wait
