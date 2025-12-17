# agents/base_agent.py
import abc
import torch
import numpy as np
import inspect

class BaseAgent(abc.ABC):
    """
    Base agent: handles observation sequence encoding and replay buffer.
    Subclasses must implement _actor_action, _update_critics, _update_actor.
    """
    def __init__(self,
                 encoder,
                 replay_buffer,
                 batch_size,
                 device=torch.device('cpu')):
        self.encoder = encoder
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.device = device
        self._training_done = False

    def select_action(self, obs_seq, deterministic=False):
        """
        obs_seq: numpy array or torch tensor of shape (N, obs_dim)
        Returns action as numpy array
        """
        # to tensor
        if not torch.is_tensor(obs_seq):
            obs = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            obs = obs_seq.unsqueeze(0).to(self.device)
        # encode sequence
        with torch.no_grad():
            state = self.encoder(obs)  # expected to return torch tensor shape [1, state_dim]
            action = self._actor_action(state, deterministic)
        # to numpy (ensure float32, squeeze batch dim) and clip for safety
        if torch.is_tensor(action):
            a = action.detach().cpu().numpy()
        else:
            a = np.array(action)
        a = np.squeeze(a).astype(np.float32)
        a = np.clip(a, -1.0, 1.0)
        return a.reshape(-1)

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store raw transition into replay buffer
        obs, next_obs: numpy arrays
        action: numpy
        reward, done: scalars
        """
        rb = self.replay_buffer
        transition = (np.array(obs, dtype=np.float32), np.array(action, dtype=np.float32), float(reward),
                      np.array(next_obs, dtype=np.float32), float(done))
        # transition = (obs, action, reward, next_obs, done)

        if hasattr(rb, 'push'):
            rb.push(transition)
        elif hasattr(rb, 'add'):
            rb.add(transition)
        elif hasattr(rb, 'append'):
            rb.append(transition)
        elif hasattr(rb, 'storage') and isinstance(rb.storage, list):
            rb.storage.append(transition)
            # respect max size if present
            if hasattr(rb, 'max_size') and len(rb.storage) > rb.max_size:
                rb.storage.pop(0)
        else:
            raise RuntimeError("Unknown replay buffer API: cannot store transition")

    def update(self):
        """
        Sample batch and perform one update step.
        NOTE: We pass raw observation sequences to subclass update methods,
        and the subclass should call self.encoder(...) inside to produce
        differentiable tensors for its own backward pass. This avoids
        reusing the same computation graph for both critic and actor.
        """
        # robust check for buffer length (support different buffer APIs)
        rb = self.replay_buffer
        rb_len = 0
        if rb is not None:
            if hasattr(rb, 'storage'):
                rb_len = len(rb.storage)
            elif hasattr(rb, 'buffer'):
                rb_len = len(rb.buffer)
            else:
                try:
                    rb_len = len(rb)
                except Exception:
                    rb_len = 0
        if rb_len < self.batch_size:
            return None, None

        # sample from buffer — assume buffer.sample returns numpy arrays in right order
        batch = self.replay_buffer.sample(self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = batch

        # to torch tensors (keep as raw observation sequences - no encoder forward here)
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        act_batch = torch.tensor(act_batch, dtype=torch.float32, device=self.device)
        rew_batch = torch.tensor(rew_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Call subclass update methods — subclasses MUST call self.encoder(...) internally
        critic_loss = self._update_critics(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)
        # actor_loss = self._update_actor(obs_batch)
        if not hasattr(self, "_actor_accepts_action"):
            try:
                sig = inspect.signature(self._update_actor)
                params = list(sig.parameters.values())
                # handle both bound and unbound methods: ignore a leading 'self' if present
                if len(params) >= 1 and params[0].name == 'self':
                    param_count = max(0, len(params) - 1)
                else:
                    param_count = len(params)
                # true if method expects at least two arguments (state, action)
                self._actor_accepts_action = (param_count >= 2)
            except Exception:
                # safe fallback
                self._actor_accepts_action = False

        if self._actor_accepts_action:
            actor_loss = self._update_actor(obs_batch, act_batch)
        else:
            actor_loss = self._update_actor(obs_batch)


        return critic_loss, actor_loss

    @abc.abstractmethod
    def _actor_action(self, state, deterministic=False):
        """Given encoded state, return action tensor"""
        pass

    @abc.abstractmethod
    def _update_critics(self, state_batch, act_batch, rew_batch, next_state_batch, done_batch):
        """Perform critic(s) update; return critic loss"""
        pass

    @abc.abstractmethod
    def _update_actor(self, state_batch):
        """Perform actor update; return actor loss"""
        pass