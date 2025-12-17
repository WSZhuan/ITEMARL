# agents/itemarl_td3_agent.py
import copy
import torch
import torch.optim as optim
import torch.nn as nn

from agents.base_agent import BaseAgent
from models.improve_transformer import ImproveTransformerEncoderState
from models.asynchronous import AsyncReplayBuffer
from models.asynchronous import AsyncSampler
from models.td3_networks import ActorNetwork as TD3Actor, CriticNetwork as TD3Critic


def worker_fn(env_fn, agent, replay_buffer, T_max):
    """
    each worker in its own env instance runs for T_max steps,
    and stores transitions in the replay buffer
    """
    env = env_fn()
    obs = env.reset()
    for _ in range(T_max):
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, float(done))
        obs = next_obs if not done else env.reset()


class ITEMARLTD3Agent(BaseAgent):
    """
    ITEMARL + TD3 agent
    framework: Improved Transformer Encoder -> TD3 Networks
    """

    def __init__(self, obs_dim, act_dim, seq_len, config, device=torch.device('cpu'), env_fn=None):
        # 1) use Improved Transformer Encoder
        self.transformer = ImproveTransformerEncoderState(
            seq_len=seq_len,
            config=config
        ).to(device)

        # 2) use Transformer output as encoder output
        embed_dim = int(config.get('embed_dim', obs_dim))

        # 3) use Transformer output as encoder output
        encoder = self.transformer

        # 4) use asynchronous replay buffer
        buffer_size = int(config.get('buffer_size', 1e6))
        buffer = AsyncReplayBuffer(capacity=buffer_size)

        # 5) initialize base class
        super().__init__(
            encoder=encoder,
            replay_buffer=buffer,
            batch_size=int(config.get('batch_size', 256)),
            device=device
        )

        # 6) TD3 hyperparameters
        self.gamma = float(config.get('gamma', 0.99))
        self.tau = float(config.get('tau', 0.005))
        self.policy_noise = float(config.get('policy_noise', 0.2))
        self.noise_clip = float(config.get('noise_clip', 0.5))
        self.policy_delay = int(config.get('policy_delay', 2))
        self.total_it = 0
        self.device = device
        self.env_fn = env_fn

        # 7) actor & twin critics
        self.actor = TD3Actor(embed_dim, act_dim, config).to(device)  # 输入embed_dim维
        self.actor_target = copy.deepcopy(self.actor).to(device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic = TD3Critic(embed_dim, act_dim, config).to(device)  # 输入embed_dim维
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # 8) use Adam optimizer - exclude projection block parameters
        lr = float(config.get('lr', 1e-3))
        self.actor_opt = optim.Adam(
            list(self.actor.parameters()),
            lr=lr
        )
        self.critic_opt = optim.Adam(
            list(self.transformer.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

        # 9) use multi-threaded workers
        T_max = int(config.get('max_steps', 800))
        self.num_workers = int(config.get('num_workers', 6))
        self.async_sampler = None
        if env_fn is not None:
            self.async_sampler = AsyncSampler(
                env_fn=env_fn,
                agent=self,
                buffer=self.replay_buffer,
                num_workers=self.num_workers,
                max_steps_per_episode=T_max
            )

        print(f"[ITEMARL_TD3Agent] confirm architecture:")
        print(f"  - Improved Transformer Encoder: {embed_dim} input dimension")
        print(f"  - TD3 Networks: {embed_dim} input dimension")
        print(f"  - Multi-threaded Workers: {self.num_workers}个worker")

        # 10) verify all modules are used
        self._verify_modules_used()

    def _verify_modules_used(self):
        """verify all modules are used"""
        network_info = self.get_network_info()
        print(f"[ITEMARL_TD3Agent] verify modules:")
        print(f"  ✅ Transformer parameters: {network_info['transformer_params']}")
        print(f"  ✅ Actor parameters: {network_info['actor_params']}")
        print(f"  ✅ Critic parameters: {network_info['critic_params']}")
        print(f"  ✅ Total parameters: {network_info['total_params']}")
        print(f"  ✅ TD3 hyperparameters: gamma={self.gamma}, tau={self.tau}, policy_delay={self.policy_delay}")

        # 11) check critical modules exist
        assert hasattr(self, 'encoder'), "lack of Encoder module"
        assert hasattr(self, 'transformer'), "lack of Transformer module"
        assert hasattr(self, 'actor'), "lack of Actor module"
        assert hasattr(self, 'critic'), "lack of Critic module"
        print("  ✅ All modules checked")

    def _count_params(self, module):
        """count parameters of module"""
        return sum(p.numel() for p in module.parameters())

    def get_network_info(self):
        """return network parameters info"""
        transformer_params = self._count_params(self.transformer)
        actor_params = self._count_params(self.actor)
        critic_params = self._count_params(self.critic)

        total_params = transformer_params + actor_params + critic_params

        return {
            'transformer_params': transformer_params,
            'actor_params': actor_params,
            'critic_params': critic_params,
            'total_params': total_params
        }

    def train(self):
        """start training (multi-threaded data collection + main thread update)"""
        # only start data collection threads when async_sampler exists
        if self.async_sampler:
            print(f"start {self.num_workers} EnvWorker threads...")
            self.async_sampler.start()

        # main training loop
        try:
            while True:
                # only start update when enough samples in buffer
                if len(self.replay_buffer.storage) < self.batch_size:
                    continue
                self.update()
        except KeyboardInterrupt:
            print("training interrupted")
        finally:
            # stop all worker threads
            if self.async_sampler:
                self.async_sampler.stop()
            print("training finished")

    def _actor_action(self, state, deterministic=False):
        """given encoded state [B,embed_dim], return action [B,act_dim]"""
        mu = self.actor(state)
        # TD3 uses deterministic policy
        return torch.tanh(mu)

    def _update_critics(self, st, ac, rew, st2, done):
        """compute TD3 critic loss and perform gradient step"""
        # use Transformer encoder (no projection block)
        st_enc = self.encoder(st)  # transformer -> embed_dim
        st2_enc = self.encoder(st2)  # transformer -> embed_dim

        with torch.no_grad():
            # add clipped noise to target actions
            noise = (torch.randn_like(ac) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(st2_enc) + noise).clamp(-1, 1)

            q1_t, q2_t = self.critic_target(st2_enc, a2)
            q_targ = torch.min(q1_t, q2_t)
            y = rew + self.gamma * (1 - done) * q_targ

        q1, q2 = self.critic(st_enc, ac)
        loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.item()

    def _update_actor(self, st):
        """compute TD3 actor loss and perform delayed policy update"""
        self.total_it += 1
        actor_loss = None

        # use Transformer encoder
        st_enc = self.encoder(st)  # transformer -> embed_dim

        # delayed policy update
        if self.total_it % self.policy_delay == 0:
            a_pred = self.actor(st_enc)
            actor_loss = -self.critic.Q1(st_enc, torch.tanh(a_pred)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # soft update target networks
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)
            for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return actor_loss.item() if actor_loss is not None else 0.0

    def store_transition(self, obs, action, reward, next_obs, done):
        """store transition to replay buffer"""
        # store original sequence unchanged
        self.replay_buffer.push((obs, action, reward, next_obs, done))

    def get_training_info(self):
        """return training related information"""
        return {
            'total_iterations': self.total_it,
            'replay_buffer_size': len(self.replay_buffer.storage) if hasattr(self.replay_buffer, 'storage') else 0,
            'num_workers': self.num_workers
        }