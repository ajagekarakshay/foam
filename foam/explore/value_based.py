import random
import numpy as np

__all__ = (
    'EpsilonGreedy',
    'BoltzmannPolicy'
)

class EpsilonGreedy:
    def __init__(self, epsilon, q, epsilon_min=0.01):
        self.epsilon = epsilon
        self.q = q
        self.epsilon_min = 0.01

    def act(self, s, explore=True):
        if random.random() < self.epsilon and explore:
            #print("Random action ")
            return self.q.env.action_space.sample()
        else:
            #print("Action : ", end=" ")
            qs = self.q(s, training=False)
            qs = self.q.proc.invert(qs)
            qs = qs.ravel() 
            assert len(qs) == self.q.env.action_space.n
            return np.argmax(qs)

    def update(self, decay=0.995):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= decay

class BoltzmannPolicy:
    def __init__(self):
        pass