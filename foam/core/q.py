import torch
from .torch_base import TorchBaseFunction, TorchProcessor
import tensorflow as tf
from .tf_base import TFBaseFunction, TFProcessor

__all__ = (
    'Q'
)

def Q(*argv, **kwargs):
    """
    General Q-function class
    """
    if torch.nn.Module in argv[1].__class__.__mro__:
        return Q_torch(*argv, **kwargs)
    if tf.keras.Model in argv[1].__class__.__mro__:
        return Q_tf(*argv, **kwargs)

class Q_torch(TorchBaseFunction):    
    """
    Q function class for PyTorch models.
    """         
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.proc = TorchProcessor(self.state_dim, self.action_dim)

    def forward(self, s, a=None, training=True):
        s = self.proc(s)
        a = self.proc(a) if a is not None and type(a) is not torch.Tensor else a
        with torch.inference_mode(mode=not training):
            value = self.model(s, a) if a is not None else self.model(s)
        return value

class Q_tf(TFBaseFunction):
    """
    Q function class for TensorFlow models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.proc = TFProcessor(self.state_dim, self.action_dim)
    
    def call(self, s, a=None, training=True):
        s = self.proc(s)
        a = self.proc(a) if a is not None and type(a.shape) is not tf.TensorShape else a
        value = self.model(s, a, training=training) if a is not None else self.model(s, training=training)
        return value