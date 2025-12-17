# models/td3_networks.py
import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    """
    Deterministic actor network for TD3. Maps state embedding to action means.
    """
    def __init__(self, state_dim, act_dim, config):
        super().__init__()
        hid = int(config.get('hidden_dim', 256))
        # simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, act_dim)
        )

    def forward(self, state):
        # state: [B, state_dim]
        # output: raw action, to be tanh-ed externally
        return self.net(state)


class CriticNetwork(nn.Module):
    """
    Twin critic network for TD3. Returns two Q-value estimates.
    Also provides Q1 for actor update convenience.
    """
    def __init__(self, state_dim, act_dim, config):
        super().__init__()
        hid = int(config.get('hidden_dim', 256))
        # Q1
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )
        # Q2
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, state, action):
        # state: [B, state_dim], action: [B, act_dim]
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        # convenience for actor update
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)