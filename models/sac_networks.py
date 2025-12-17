# models/sac_networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionBlock(nn.Module):
    """
    Two-layer MLP with LN and ReLU: [in->hidden->out]
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class ActorNetwork(nn.Module):
    """Actor for SAC: outputs mean and log_std for Gaussian policy."""
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        hid1, hid2 = config.get('actor_h1', 256), config.get('actor_h2', 64)
        self.proj = ProjectionBlock(state_dim, hid1, hid2)
        self.mu = nn.Linear(hid2, action_dim)
        self.log_std = nn.Linear(hid2, action_dim)
        self.log_std_min = config.get('log_std_min', -20)
        self.log_std_max = config.get('log_std_max', 2)

    def forward(self, state_feat):
        x = self.proj(state_feat)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state_feat):
        mu, log_std = self(state_feat)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        logp = dist.log_prob(z).sum(-1, keepdim=True)
        # Tanh correction
        logp -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, logp

class CriticNetwork(nn.Module):
    """Double-Q Critic for SAC: two critics Q1 and Q2."""
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        in_dim = state_dim + action_dim
        hid1, hid2 = config.get('critic_h1', 256), config.get('critic_h2', 64)
        self.proj1 = ProjectionBlock(in_dim, hid1, hid2)
        self.q1 = nn.Linear(hid2, 1)
        self.proj2 = ProjectionBlock(in_dim, hid1, hid2)
        self.q2 = nn.Linear(hid2, 1)

    def forward(self, state_feat, action):
        sa = torch.cat([state_feat, action], dim=-1)
        q1 = self.q1(self.proj1(sa))
        q2 = self.q2(self.proj2(sa))
        return q1, q2
