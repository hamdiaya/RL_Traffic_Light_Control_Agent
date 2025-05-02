# ppo_agent.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from ppo_model import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim, device, gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, epochs=10, batch_size=64):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def compute_gae(self, rewards, values, masks):
        values = values + [0]
        gae, returns = 0, []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, trajectories):
        states = torch.tensor(trajectories['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(trajectories['actions']).to(self.device)
        old_log_probs = torch.tensor(trajectories['log_probs']).to(self.device)
        returns = torch.tensor(trajectories['returns']).to(self.device)
        values = torch.tensor(trajectories['values']).to(self.device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            idxs = torch.randperm(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_idx = idxs[i:i+self.batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                probs, value = self.model(batch_states)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value.squeeze(), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
