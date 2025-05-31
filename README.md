# 🚦 Traffic Light Control Agent using Deep Reinforcement Learning

This project explores the use of **Deep Reinforcement Learning (DRL)** for intelligent traffic light control at a single four-way intersection, simulated using **SUMO** (Simulation of Urban Mobility).

Our goal was to minimize traffic congestion and vehicle waiting time by training an agent that learns to dynamically adjust traffic signal phases based on real-time conditions.

---

## 🧠 Project Overview

We designed and evaluated multiple RL-based traffic control strategies:

### 🔹 Fixed Duration Control
- Agent selects a traffic phase; duration is fixed (e.g., 30s).
- **Techniques:** DQN, Double DQN, Dueling DQN, PER

### 🔹 Dynamic Phase Durations
- Agent selects both phase and duration from a discrete set.
- Improves adaptability during varying traffic loads.

### 🔹 Throughput-Based Control
- Agent switches phases based on vehicle throughput (e.g., % of cars that have passed).
- Yields the best performance in terms of reduced wait times and queue lengths.

---

## 🧪 Techniques Used

- 🧠 DQN, Double DQN, Dueling DQN  
- 🧪 Prioritized Experience Replay (PER)  
- 🎯 Curriculum Learning  
- 📈 Layer Normalization, Dropout, Gradient Clipping  
- ⚙️ Dynamic Action Spaces (phase + duration or throughput thresholds)  

---

## 🛠️ Tools & Frameworks

- 🧮 PyTorch for neural network implementation  
- 🛣️ SUMO for traffic simulation  
- 🧪 Python for RL environment integration  

---

## 📈 Results

Our best-performing agent (**throughput-based control**) achieved:

- 🚗 ~60% reduction in waiting time  
- 🚦 Significant improvement in traffic flow and queue management  
- 💡 Adaptive behavior across varied traffic patterns  


---
