# models/asynchronous.py
import threading
import numpy as np

class AsyncReplayBuffer:
    """
    Thread-safe replay buffer for asynchronous experience collection.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()

    def push(self, experience):
        """
        Push an experience tuple into the buffer.
        experience: (obs, action, reward, next_obs, done)
        """
        with self.lock:
            if len(self.buffer) >= self.capacity:
                # drop oldest
                self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences uniformly.
        Returns a list of experiences.
        """
        with self.lock:
            idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in idxs]
        # transpose
        obs, acts, rews, next_obs, dones = map(np.stack, zip(*batch))
        return obs, acts, rews, next_obs, dones

class EnvWorker(threading.Thread):
    """
    Worker thread that collects experiences from its own env instance.
    """
    def __init__(self, env_fn, agent, buffer, max_steps_per_episode):
        super().__init__()
        self.env = env_fn()
        self.agent = agent
        self.buffer = buffer
        self.max_steps = max_steps_per_episode
        self.daemon = True
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            obs = self.env.reset()
            for t in range(self.max_steps):
                # select action (no learning)
                action = self.agent.select_action(obs)
                next_obs, reward, done, truncated, info = self.env.step(action)
                self.buffer.push((obs, action, reward, next_obs, float(done)))
                obs = next_obs if not done else self.env.reset()
                if self._stop_event.is_set():
                    break

    def stop(self):
        self._stop_event.set()

class AsyncSampler:
    """
    Manager for multiple EnvWorker threads.
    """
    def __init__(self, env_fn, agent, buffer, num_workers, max_steps_per_episode):
        self.workers = []
        for _ in range(num_workers):
            w = EnvWorker(env_fn, agent, buffer, max_steps_per_episode)
            self.workers.append(w)

    def start(self):
        for w in self.workers:
            w.start()

    def stop(self):
        for w in self.workers:
            w.stop()
            w.join()
