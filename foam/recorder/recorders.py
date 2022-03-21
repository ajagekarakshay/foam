import numpy as np
from collections import deque
from itertools import islice
from .transition import TransitionBatch

__all__ = (
    'BaseRecorder',
    'NStepRecorder'
)

class BaseRecorder:
    def reset(self):
        raise NotImplementedError
    def add(self, s, a, r, done, s_next, w=1.0):
        raise NotImplementedError
    def pop(self):
        raise NotImplementedError
    def flush(self):
        if not self:
            raise ValueError("Empty recorder")
        while self:
            self.pop()
    def __bool__(self):
        raise NotImplementedError
    
class NStepRecorder(BaseRecorder):
    def __init__(self, n=1, gamma=0.9, bootstrap=False):
        super().__init__()
        self.n = n
        self.gamma = gamma
        self.bootstrap = bootstrap
        self.reset()

    def reset(self):
        self.deque_s = deque([])
        self.deque_r = deque([])
        self.done = False
        
    def __len__(self):
        return len(self.deque_s)

    def add(self, s, a, r, done, s_next):
        if self.done and len(self):
            raise ValueError("Episode completed : flush before adding")
        self.deque_s.append( (s,a,r,done,s_next) )
        self.deque_r.append( r )
        self.done = bool(done)

    def pop(self):
        if not self:
            raise ValueError("Empty recorder")

        s,a,r,done,s_next = self.deque_s.popleft()

        if self.bootstrap:
            raise NotImplementedError("bootstrapping rewards not yet implemented")
        else:
            _ = self.deque_r.popleft()

        return TransitionBatch.from_single(
            s=s, a=a, r=r, done=done, s_next=s_next, gamma=self.gamma
        )

    def popN(self, n):
        if not self:
            raise ValueError("Empty recorder")
        if n > len(self):
            raise ValueError("Not enough transitions")
        
        transitions = [self.pop() for _ in range(n)]
        cat = lambda *x: np.concatenate(x, axis=0)
        return TransitionBatch(
            S=cat([t.S for t in transitions]),
            A=cat([t.A for t in transitions]),
            R=cat([t.R for t in transitions]),
            Done=cat([t.Done for t in transitions]),
            S_next=cat([t.S_next for t in transitions]),
            Gamma=cat([t.Gamma for t in transitions]),
            W=cat([t.W for t in transitions]),
        )
    
    def __bool__(self):
        return bool(len(self))