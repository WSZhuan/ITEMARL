# agents/ablation_base_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import copy

from agents.base_agent import BaseAgent
from models.asynchronous import AsyncReplayBuffer
from models.asynchronous import AsyncSampler
from models.td3_networks import ActorNetwork as TD3Actor, CriticNetwork as TD3Critic

class AblationITEMATD3Agent(BaseAgent):
    """
    ablation experiment base class with configurable components
    """

    def __init__(self, obs_dim, act_dim, seq_len, config, device=torch.device('cpu'), env_fn=None):
        # read ablation parameters from config
        self.transformer_type = config.get('transformer_block', 'improve')
        self.num_workers = int(config.get('num_workers', 5))
        self.curriculum_mode = config.get('curriculum', 'adaptive')

        # 1) create Transformer encoder
        self.transformer = self._create_transformer(seq_len, config, device)

        # 2) create embedding layer
        embed_dim = int(config.get('embed_dim', obs_dim))

        # 3) use Transformer encoder as encoder
        encoder = self.transformer

        # 4) create asynchronous replay buffer
        buffer_size = int(config.get('buffer_size', 1e6))
        buffer = AsyncReplayBuffer(capacity=buffer_size)

        # 5) initialize base class
        super().__init__(
            encoder=encoder,
            replay_buffer=buffer,
            batch_size=int(config.get('batch_size', 256)),
            device=device
        )

        # 6) create actor and critic networks
        self.gamma = float(config.get('gamma', 0.99))
        self.tau = float(config.get('tau', 0.005))
        self.policy_noise = float(config.get('policy_noise', 0.2))
        self.noise_clip = float(config.get('noise_clip', 0.5))
        self.policy_delay = int(config.get('policy_delay', 2))
        self.total_it = 0
        self.device = device
        self.env_fn = env_fn

        input_dim = embed_dim

        self.actor = TD3Actor(input_dim, act_dim, config).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic = TD3Critic(input_dim, act_dim, config).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # 8) create optimizers - include all module parameters
        lr = float(config.get('lr', 1e-3))
        self.actor_opt = optim.Adam(
            list(self.actor.parameters()),
            lr=lr
        )

        critic_params = (list(self.transformer.parameters()) +
                             list(self.critic.parameters()))

        self.critic_opt = optim.Adam(critic_params, lr=lr)

        # 9) create AsyncSampler to manage multi-threading
        T_max = int(config.get('max_steps', 800))
        self.async_sampler = None

        # only create sampler when env_fn is provided
        if env_fn is not None and self.num_workers > 0:
            # print(f"[DEBUG] create {self.num_workers} worker")
            self.async_sampler = AsyncSampler(
                env_fn=env_fn,
                agent=self,
                buffer=self.replay_buffer,
                num_workers=self.num_workers,
                max_steps_per_episode=T_max
            )

        print(f"[{self.__class__.__name__}] confirm architecture:")
        print(f"  - Transformer({self.transformer_type}): {embed_dim} dimension input")
        print(f"  - TD3 networks: {input_dim} dimension input")
        print(f"  - multi-threading: {self.num_workers} workers")
        print(f"  - curriculum learning: {self.curriculum_mode}")

    def _create_transformer(self, seq_len, config, device):
        """create Transformer encoder based on config"""
        if self.transformer_type == "improve":
            from models.improve_transformer import ImproveTransformerEncoderState
            return ImproveTransformerEncoderState(
                seq_len=seq_len,
                config=config
            ).to(device)
        else:
            from models.transformer_encoder import TransformerEncoderState
            return TransformerEncoderState(
                seq_len=seq_len,
                config=config
            ).to(device)


    def train(self):
        """train the agent with multi-threading data collection and single-threading update"""
        # start all data collection threads
        if self.async_sampler:
            print(f"start {self.num_workers} EnvWorker threads...")
            self.async_sampler.start()

        import time
        last_buffer_size = len(self.replay_buffer.buffer)
        start_time = time.time()
        last_log_time = start_time
        episode_count = 0

        print("start training loop...")

        try:
            while episode_count < 1800:
                current_time = time.time()

                # each 10 episodes or 30 seconds, report status
                if episode_count % 10 == 0 or current_time - last_log_time > 30:
                    buffer_size = len(self.replay_buffer.buffer)
                    buffer_growth = buffer_size - last_buffer_size
                    elapsed = current_time - start_time

                    # check worker status
                    active_workers = 0
                    if self.async_sampler:
                        active_workers = sum(1 for w in self.async_sampler.workers if w.is_alive())

                    print(f"[Ep {episode_count}] active workers: {active_workers}/{self.num_workers}, "
                          f"buffer size: {buffer_size} (+{buffer_growth}), "
                          f"training time: {elapsed:.1f}s")

                    last_buffer_size = buffer_size
                    last_log_time = current_time

                # only start update when enough samples in buffer
                if len(self.replay_buffer.buffer) < self.batch_size:
                    time.sleep(0.1)
                    continue

                # run training update
                critic_loss, actor_loss = self.update()
                episode_count += 1

        except KeyboardInterrupt:
            print("training interrupted")
        finally:
            # stop all worker threads
            if self.async_sampler:
                self.async_sampler.stop()
            print("training finished")

    def _actor_action(self, state, deterministic=False):
        """given encoded state, return action"""
        mu = self.actor(state)
        return torch.tanh(mu)

    def _update_critics(self, st, ac, rew, st2, done):
        """update critic networks"""
        st_enc = self.encoder(st)
        st2_enc = self.encoder(st2)

        with torch.no_grad():
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
        """update actor network"""
        self.total_it += 1
        actor_loss = None

        st_enc = self.encoder(st)

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
        self.replay_buffer.push((obs, action, reward, next_obs, done))

    def get_training_info(self):
        """return training related information"""
        buffer_size = len(self.replay_buffer.storage) if hasattr(self.replay_buffer, 'storage') else 0
        print(f"replay buffer size: {buffer_size}, worker threads: {len(self.workers)}")
        return {
            'total_iterations': self.total_it,
            'replay_buffer_size': buffer_size,
            'num_workers': len(self.workers),
            'active_workers': sum(1 for w in self.workers if w.is_alive())
        }


