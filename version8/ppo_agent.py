import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
        # Critic: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state):
        features = self.feature(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, ppo_epochs=10, batch_size=64, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        self.policy = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for rollouts
        self.memory = deque(maxlen=10000)

    def act(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample() if not eval_mode else torch.argmax(action_probs, dim=-1)
        log_prob = dist.log_prob(action) if not eval_mode else None
        return action.item(), log_prob is not None

    def store(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        # Enable anomaly detection for debugging (disable after confirming fix)
        # torch.autograd.set_detect_anomaly(True)

        # Convert memory to tensors efficiently
        states, actions, rewards, next_states, dones, log_probs = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor([lp if lp is not None else 0 for lp in log_probs]).to(self.device)

        # Compute advantages and returns (avoid inplace operations)
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            values = self.policy(states)[1]
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        gae_lambda = 0.95
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
            returns[t] = advantage + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Sample batch
            indices = np.random.permutation(len(states))[:self.batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]

            # Forward pass
            action_probs, values = self.policy(batch_states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            ratios = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (batch_returns - values.squeeze()).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=(self.ppo_epochs > 1))
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.memory.clear()
        return total_loss / self.ppo_epochs

    def save_model(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load_model(self, filename):
        self.policy.load_state_dict(torch.load(filename, weights_only=True))