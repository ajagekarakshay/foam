import numpy as np

__all__ = (
    'TransitionBatch',
    'TransitionLoader'
)


class TransitionBatch:
    def __init__(self, S, A, R, Done, S_next, Gamma, W=None, idx=None):
        self.S = S
        self.A = A
        self.R = R
        self.Done = Done
        self.S_next = S_next
        self.Gamma = Gamma
        self.W = np.ones_like(R) if W is None else W
        self.idx = np.arange(R.shape[0], dtype='int32') if idx is None else idx
    
    @classmethod
    def from_single(cls, s, a, r, done, s_next, gamma, w=1, idx=None):
        return cls(
            S=single_to_batch(s),
            A=single_to_batch(a),
            R=single_to_batch(r),
            Done=single_to_batch(done),
            S_next=single_to_batch(s_next),
            Gamma = single_to_batch(gamma),
            W=single_to_batch(float(w)),
            idx=single_to_batch(idx) if idx is not None else None
        )
    
    @property
    def batch_size(self):
        return np.shape(self.R)[0]
    
    def __len__(self):
        return self.batch_size

    # @property
    # def shape(self):
    #     return (self.batch_size, )

    def to_singles(self):
        if self.batch_size == 1:
            yield self
            return
        
        zipped = zip(self.S, self.A, self.R, self.Done, self.S_next, self.W)

        for s,a,r,done,s_next,w in zipped:
            yield TransitionBatch.from_single(
                s=s, a=a, r=r, done=done, s_next=s_next, w=w
            ) 
    
def single_to_batch(x):
    return np.expand_dims(x, axis=0)


class TransitionLoader:
    def __init__(self, transitions:TransitionBatch, batch_size=None, shuffle=False):
        self.batch_size = batch_size if batch_size is not None else len(transitions)
        self.num_batches = int(np.ceil(len(transitions) / self.batch_size))
        if self.batch_size == len(transitions):
            self.data = [transitions]
        else:
            # Implement mini-batching of transition batch
            pass
        self._index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < self.num_batches:
            result = self.data[self._index]
            self._index += 1
            return result
        self._index = 0
        raise StopIteration

    def __len__(self):
        return self.num_batches
