import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Dueling DQN
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        x = self.feature(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay_steps=2000000, buffer_capacity=10000, batch_size=64, tau=0.01, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_min) / epsilon_decay_steps
        self.batch_size = batch_size
        self.tau = tau

        # Device selection
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1), True  # Explore
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item(), False  # Exploit

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        next_actions = self.q_network(next_state).max(1)[1].detach()
        next_q_values = self.target_network(next_state).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Linear decay of epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filename="dqn_model.pth"):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename="dqn_model.pth"):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())