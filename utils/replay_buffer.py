# utils/replay_buffer.py

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage=[]; self.max_size=max_size
    def add(self, data):
        self.storage.append(data)
        if len(self.storage)>self.max_size: self.storage.pop(0)
    def sample(self, batch_size):
        ind = np.random.randint(0,len(self.storage), size=batch_size)
        obs,acts,rews,obs2,done = [],[],[],[],[]
        for i in ind:
            o,a,r,o2,d = self.storage[i]
            obs.append(o); acts.append(a); rews.append(r); obs2.append(o2); done.append(d)
        return np.array(obs), np.array(acts), np.array(rews), np.array(obs2), np.array(done)