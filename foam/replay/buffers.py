import numpy as np
from abc import ABC, abstractmethod
import random
from collections import deque
from ..recorder import TransitionBatch


__all__ = (
    'BaseBuffer',
    'ReplayBuffer'
)

class BaseBuffer(ABC):
    @abstractmethod
    def add(self, transition_batch):
        pass
    @abstractmethod
    def sample(self, batch_size=32):
        pass
    @abstractmethod
    def clear(self):
        pass
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def __bool__(self):
        pass
    @abstractmethod
    def __iter__(self):
        pass

class ReplayBuffer(BaseBuffer):
    def __init__(self, capacity, random_seed=None):
        self.capacity = int(capacity)
        random.seed(random_seed)
        self.random_state = random.getstate()
        self.clear() 

    def clear(self):
        self.storage = deque(maxlen=self.capacity)
        self.index = 0

    def add(self, transition_batch):
        transition_batch.idx = np.arange(self.index, self.index + transition_batch.batch_size)
        self.index += transition_batch.batch_size
        self.storage.extend(transition_batch.to_singles())

    def sample(self, batch_size=32):
        # sandwich sample in between setstate/getstate in case global random state was tampered with
        random.setstate(self.random_state)
        transitions = random.sample(self.storage, batch_size)
        self.random_state = random.getstate()
        cat = lambda x: np.concatenate(x, axis=0)
        return TransitionBatch(
            S=cat([t.S for t in transitions]),
            A=cat([t.A for t in transitions]),
            R=cat([t.R for t in transitions]),
            Done=cat([t.Done for t in transitions]),
            S_next=cat([t.S_next for t in transitions]),
            Gamma=cat([t.Gamma for t in transitions]),
            W=cat([t.W for t in transitions]),
        )

    def __len__(self):
        return len(self.storage)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self.storage)