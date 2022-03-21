from abc import ABC, abstractmethod

class Objective(ABC):
    @abstractmethod
    def flush(self): # clears autograd memory
        pass

    @abstractmethod
    def compute_loss(self, transition_batch):
        pass

    @abstractmethod
    def compute_grads(self, loss):
        pass

    @abstractmethod
    def process_grads(self, grads, transition_batch):
        pass

    @abstractmethod
    def update_step(self, transition_batch):
        pass

    @abstractmethod
    def update(self, transition_batch):
        pass