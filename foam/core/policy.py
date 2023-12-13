from abc import ABC, abstractmethod

from conrl.core.tf_base import TFBaseFunction
from .torch_base import TorchBaseFunction
import torch

__all__ = (
    'Policy', 'StochasticPolicy'
)

class Policy:
    """
    General Policy class
    """
    def __new__(cls, *args, model=None, **kwargs):
        if torch.nn.Module in model.__class__.__mro__:
            #print("Torch policy instantiated")
            return Policy_torch(*args, model=model, **kwargs)
        elif tf.keras.Model in model.__class__.__mro__:
            #print("TF policy instantiated")
            return Policy_tf(*args, model=model, **kwargs)
        else:
            return None

class StochasticPolicy:
    """
    General Stochastic Policy class
    """
    def __new__(cls, *args, model=None, **kwargs):
        if torch.nn.Module in model.__class__.__mro__:
            print("Torch stochastic policy instantiated")
            return StochasticPolicy_torch(*args, model=model, **kwargs)
        elif tf.keras.Model in model.__class__.__mro__:
            print("TF stochastic policy instantiated")
            return StochasticPolicy_tf(*args, model=model,  **kwargs)
        else:
            return None    

class Policy_torch(TorchBaseFunction):
    """
    Policy class for PyTorch models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, s, training=True):
        """
        Computes policy output for given state as per current policy.
        """
        s = self.proc(s)
        with torch.inference_mode(mode=not training):
            a = self.model(s)
        return a

    def act(self, s):
        """
        Computes actions for given states as per current policy.
        """
        a = self(s, training=False)
        return self.proc.invert(a).ravel()


class Policy_tf(TFBaseFunction):
    def __init__(self, env, model=None):
        super().__init__(*args, **kwargs) 

    def call(self, s, training=True):
        """
        Computes policy output for given state as per TF policy.
        """
        pass

    def act(self, s):
        a = self(s, training=False)
        return self.proc.invert(a).ravel()



class StochasticPolicy_torch:
    def __init__(self) -> None:
        """TODO"""
        pass



class StochasticPolicy_tf:
    def __init__(self) -> None:
        """TODO"""
        pass