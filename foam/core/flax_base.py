import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import copy
from torch.utils.data import DataLoader
from .other import _set_dimensions


class FlaxBaseFunction(nn.Module):
                            
    def __init__(self, env, model, optimizer, optim_args={}, trainer_args={}):
        super().__init__()
        self.env = env
        self.model = model
        self.optimizer = optimizer(**optim_args)
        self.optim_args = optim_args
        _set_dimensions(self, env)
        self.proc = TorchProcessor(self.state_dim, self.action_dim, device=self.accelerator.device)
        self.compile()

    def forward(self, s, a=None, training=True):
        raise NotImplementedError

    def soft_update(self, other, tau):
        for old_param, new_param in zip(self.model.parameters(), other.model.parameters()):
            old_param.data.copy_(tau*new_param.data + (1.0-tau)*old_param.data)

    def copy(self, deep=False):
        if deep:
            model_copy = copy.deepcopy(self.model)
            optimizer = self.optimizer
            return self.__class__(self.env, model_copy, type(optimizer), optim_args=self.optim_args)
        return copy.copy(self)
    
    def compile(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        #setattr(self.proc, "device", self.accelerator.device)
        #self.proc.device = self.accelerator.device

    def _clear(self):
        optimizer = self.optimizer
        optimizer.zero_grad()
        return optimizer, None

    def _gradients(self, loss, **kwargs):
        self.accelerator.backward(loss)
    
    def _apply_grads(self, optimizer, grads=None):
        optimizer.step()

    def _update(self, transition_batch, **kwargs):
        #for tb in self.accelerator.prepare( DataLoader([transition_batch]) ): pass
        loss = self.train_step(transition_batch, **kwargs)
        return loss


class JaxProcessor:
    def __init__(self, state_dim, action_dim, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        #if device is None:
        #    device = torch.device("cpu")
        #self.device = device

    def __call__(self, arg, vectorize=False):
        assert type(arg) is np.ndarray, "Processor argument should be numpy array"
        arg = torch.from_numpy(arg).float().to(self.device)
        arg = arg.view(-1, arg.shape[-1]) if self.state_dim in arg.shape or self.action_dim in arg.shape or vectorize else arg
        return arg
    
    def invert(self, arg:torch.Tensor):
        return arg.cpu().numpy()

    def max(self, target, dim, **kwargs):
        values, _ = target.max(dim, **kwargs)
        return values

    def multiply(self, target, *others):
        result = target
        for other in others: result = result.mul(self(other))
        return result

    def gather(self, target, dim, idx):
        idx = torch.from_numpy(idx).view(target.size(0), -1).to(self.device)
        return target.gather(dim, idx)

    def add(self, target, *args):
        result = target
        for arg in args: result += self(arg)
        return result
    
    def reshape(self, target, shape):
        return target.reshape(shape)

    def mean(self, target, dim=None, **kwargs):
        if dim is None:
            return torch.mean(target, **kwargs)
        return torch.mean(target, dim=dim, **kwargs)