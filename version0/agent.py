import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values

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

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_capacity=10000, batch_size=64, tau=0.01, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau  # Soft update factor

        # Device selection
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Track exploration vs. exploitation
        self.exploration_count = 0
        self.exploitation_count = 0

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)  # Learning rate decay

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def act(self, state):
        """Select an action using epsilon-greedy policy."""
        state = np.array(state, dtype=np.float32)  # Ensure state is a NumPy array
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:  # Explore
            self.exploration_count += 1
            return random.randint(0, self.action_size - 1), True
        else:  # Exploit
            q_values = self.q_network(state)
            self.exploitation_count += 1
            return torch.argmax(q_values).item(), False

    def update(self):
        """Update the Q-network using a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Compute Q-values using standard DQN
        next_q_values = self.target_network(next_state).max(1)[0].detach()
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def reset_counts(self):
        """Reset exploration and exploitation counts for logging per episode."""
        self.exploration_count = 0
        self.exploitation_count = 0

    def save_model(self, filename="dqn_model.pth"):
        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, filename="dqn_model.pth"):
        self.q_network.load_state_dict(torch.load(filename))
