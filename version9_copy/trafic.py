import traci
import os

def test_model():
    sumo_binary = "sumo"  # or "sumo-gui" for GUI
    config_file = "sumo_config.sumocfg"

    # Define SUMO command
    sumo_cmd = [sumo_binary, "-c", config_file]

    # Start simulation
    traci.start(sumo_cmd)

    # Initialize your model/agent (same as in training)
    agent = YourAgentClass(...)  # load weights if needed

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Optional: observe environment
        # e.g., state = get_state()

        # Let agent act if you're doing online RL evaluation
        # action = agent.choose_action(state)
        # apply_action(action)

        step += 1

    traci.close()
    print("Testing finished.")