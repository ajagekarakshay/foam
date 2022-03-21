from abc import ABC, abstractmethod

class RLmodel(ABC):
    @abstractmethod
    def gradients(self, loss):
        pass
    
    @abstractmethod
    def update(self, grads):
        pass

    @abstractmethod
    def clear(self):
        # clear gradients
        pass