import traci

sumoCmd = ["sumo", "--net-file", "intersection.net.xml", "--additional-files", "traffic_lights.add.xml"]
traci.start(sumoCmd)

# Get the number of signals SUMO expects
num_signals = len(traci.trafficlight.getRedYellowGreenState("n0"))
print(f"SUMO expects {num_signals} traffic light signals.")

traci.close()