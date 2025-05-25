import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    """Dueling DQN Network with Layer Normalization"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Dueling architecture streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class ReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        state, action, reward, next_state, done = zip(*samples)
        return (
            np.array(state), 
            np.array(action), 
            np.array(reward),
            np.array(next_state), 
            np.array(done), 
            indices, 
            weights
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = min(error + 1e-5, self.max_priority)

class DQNAgent:
    """Double DQN Agent with PER and Dueling Architecture"""
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3, 
                 gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.9995, buffer_capacity=100000, batch_size=64, 
                 tau=0.005, update_every=4, epsilon_decay_steps=2000000, 
                 device=None, use_double=True, use_dueling=True, use_per=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every = update_every
        self.steps = 0
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = epsilon_decay
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.gradient_norms = []
        self.gradient_histories = {name: [] for name, _ in self.q_network.named_parameters()}

    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1), True
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item(), False

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.steps += 1
        
        if self.use_per:
            state, action, reward, next_state, done, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            state, action, reward, next_state, done = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        with torch.no_grad():
            if self.use_double:
                next_actions = self.q_network(next_state).max(1)[1]
                next_q_values = self.target_network(next_state).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_state).max(1)[0]
            
            target_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        current_q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()
        loss = (weights * (current_q_values - target_q_values).pow(2)).mean()
        
        if self.use_per:
            with torch.no_grad():
                errors = torch.abs(current_q_values - target_q_values).cpu().numpy()
            self.replay_buffer.update_priorities(indices, errors)
        
        self.optimizer.zero_grad()
        loss.backward()
        self._track_gradients()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if self.steps % self.update_every == 0:
            self._soft_update_target_network()
        
        return loss.item()

    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _track_gradients(self):
        total_norm = 0
        for name, param in self.q_network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                self.gradient_histories[name].append(param_norm)
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

    def save_model(self, filename):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']