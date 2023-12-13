import numpy as np
import torch
from accelerate import Accelerator
import copy
from torch.utils.data import DataLoader
from .other import _set_dimensions


class TorchBaseFunction(torch.nn.Module):
                            
    def __init__(self, env, model, optimizer, optim_args={}, trainer_args={}):
        """
        Initializes the basefunction object for a PyTorch model.

        Parameters:
            env (object): The environment object.
            model (object): The model object.
            optimizer (object): The optimizer object.
            optim_args (dict): The optional arguments for the optimizer. Default is an empty dictionary.
            trainer_args (dict): The optional arguments for the trainer. Default is an empty dictionary.
        """
        super().__init__()
        self.env = env
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optim_args)
        self.optim_args = optim_args
        _set_dimensions(self, env)
        self.accelerator = Accelerator()
        self.proc = TorchProcessor(self.state_dim, self.action_dim, device=self.accelerator.device)
        self.compile()

    def forward(self, s, a=None, training=True):
        raise NotImplementedError

    def soft_update(self, other, tau):
        """
        Soft updates the parameters of the current model with the parameters of another model using a given tau value.

        Parameters:
            other (object): The other model whose parameters will be used for the update.
            tau (float): The weighting factor for the update. It determines the proportion of the new parameters that will be used for the update.

        Returns:
            None
        """
        for old_param, new_param in zip(self.model.parameters(), other.model.parameters()):
            old_param.data.copy_(tau*new_param.data + (1.0-tau)*old_param.data)

    def copy(self, deep=False):
        """
        Copies the model and optimizer.

        Parameters:
            deep (bool): Whether to copy the model and optimizer. Default is False.

        Returns:
            TorchBaseFunction: The copied model and optimizer.
        """
        if deep:
            model_copy = copy.deepcopy(self.model)
            optimizer = self.optimizer
            return self.__class__(self.env, model_copy, type(optimizer), optim_args=self.optim_args)
        return copy.copy(self)
    
    def compile(self):
        """
        Compiles the model and optimizer by preparing them for the accelerator.

        Parameters:
            None

        Returns:
            None
        """
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

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


class TorchProcessor:
    """
    A processor for PyTorch models that converts numpy arrays to PyTorch tensors.

    Attributes:
        state_dim (int): The dimensionality of the state space.
        action_dim (int): The dimensionality of the action space.
        device (torch.device): The device on which the processor will be used.
    """
    def __init__(self, state_dim, action_dim, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        if device is None:
            device = torch.device("cpu")
        self.device = device

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