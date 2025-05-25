import random

# Define all possible routes
routes_list = [
    ("route_E2TL_EW_straight", ["E2TL", "TL2W"]),
    ("route_E2TL_NS_right", ["E2TL", "TL2N"]),
    ("route_E2TL_SW_left", ["E2TL", "TL2S"]),
    ("route_N2TL_NS_straight", ["N2TL", "TL2S"]),
    ("route_N2TL_WE_right", ["N2TL", "TL2W"]),
    ("route_N2TL_ES_left", ["N2TL", "TL2E"]),
    ("route_W2TL_WE_straight", ["W2TL", "TL2E"]),
    ("route_W2TL_EN_right", ["W2TL", "TL2N"]),
    ("route_W2TL_WS_left", ["W2TL", "TL2S"]),
    ("route_S2TL_SN_straight", ["S2TL", "TL2N"]),
    ("route_S2TL_NE_right", ["S2TL", "TL2E"]),
    ("route_S2TL_EW_left", ["S2TL", "TL2W"]),
]

# Functions to compute probabilities and configurations
def get_profile_probs(d):
    P_low = (1 - d) * 0.8 + d * 0.0
    P_medium = (1 - d) * 0.2 + d * 0.4
    P_high = (1 - d) * 0.0 + d * 0.6
    return [P_low, P_medium, P_high]

def get_profile_config(profile, d):
    if profile == 'low':
        spawn_base, spawn_target = 0.1, 0.1
        max_lanes_base, max_lanes_target = 1, 1
    elif profile == 'medium':
        spawn_base, spawn_target = 0.3, 0.5
        max_lanes_base, max_lanes_target = 2, 3
    elif profile == 'high':
        spawn_base, spawn_target = 0.5, 0.9
        max_lanes_base, max_lanes_target = 3, 4
    spawn = spawn_base + d * (spawn_target - spawn_base)
    max_lanes = round(max_lanes_base + d * (max_lanes_target - max_lanes_base))
    return {'spawn': spawn, 'max_lanes': max_lanes}

# Generate traffic for T seconds with difficulty d
T = 3600  # Example: 1 hour
d = 1.0   # Example: full difficulty
vehicles = []
for t in range(T):
    if random.random() < 0.5:
        profile_probs = get_profile_probs(d)
        profiles = ['low', 'medium', 'high']
        selected_profile_index = random.choices(range(len(profiles)), weights=profile_probs, k=1)[0]
        selected_profile = profiles[selected_profile_index]
        config = get_profile_config(selected_profile, d)
        if random.random() < config['spawn']:
            route_id = random.choice([r[0] for r in routes_list])
            departLane = random.randint(0, config['max_lanes'] - 1)
            speed = random.uniform(5, 12.8)
            veh_id = f"veh_{t}"
            vehicles.append((veh_id, route_id, t, departLane, speed))

# Write to .rou.xml
with open("traffic.rou.xml", "w") as f:
    f.write('<routes>\n')
    for route_id, edges in routes_list:
        f.write(f'    <route id="{route_id}" edges="{" ".join(edges)}"/>\n')
    for veh_id, route_id, depart, departLane, speed in vehicles:
        f.write(f'    <vehicle id="{veh_id}" route="{route_id}" depart="{depart}" departLane="{departLane}" departSpeed="{speed}"/>\n')
    f.write('</routes>\n')