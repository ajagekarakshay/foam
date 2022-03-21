import random
import numpy as np
import copy

__all__ = (
    'OUprocess',
)

class OUprocess:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, pi, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.pi = pi
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

    def act(self, s, explore=True):
        #print("Action : ", end=" ")
        noise = self.sample() if explore else 0.0
        action = self.pi.act(s) + noise
        return action