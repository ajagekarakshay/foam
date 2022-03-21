import numpy as np
import torch
import pytorch_lightning as pl
import copy
from .other import _set_dimensions
from ..recorder import TransitionLoader

class TorchBaseFunction(pl.LightningModule):
    
    # done for strong referencing
    trainer = pl.Trainer(
                         max_epochs=1,
                         min_epochs=1, 
                         logger=False, 
                         checkpoint_callback=False,
                         weights_summary=None,
                         progress_bar_refresh_rate=0,
                         num_sanity_val_steps=0,
                         auto_select_gpus=True,
                         #gpus=1,
                         accelerator="auto"
                        )
                            
    def __init__(self, env, model, optimizer, optim_args={}, trainer_args={}):
        super().__init__()
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.optim_args = optim_args
        self.automatic_optimization = False
        _set_dimensions(self, env)
        self.proc = TorchProcessor(self.state_dim, self.action_dim)

    def forward(self, s, a=None, training=True):
        raise NotImplementedError

    def soft_update(self, other, tau):
        for old_param, new_param in zip(self.model.parameters(), other.model.parameters()):
            old_param.data.copy_(tau*new_param.data + (1.0-tau)*old_param.data)

    def copy(self, deep=False):
        if deep:
            model_copy = copy.deepcopy(self.model)
            optimizer = self.optimizer
            return self.__class__(self.env, model_copy, type(optimizer), optim_args=optimizer.defaults)
        return copy.copy(self)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optim_args)

    # def _setup(self, transition_batch):
    #     self.transition_batch = transition_batch

    def _clear(self):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        return optimizer, None

    def _gradients(self, loss, **kwargs):
        self.manual_backward(loss)
    
    def _apply_grads(self, optimizer, grads=None):
        optimizer.step()

    def _update(self, transition_batch):
        dataloader = TransitionLoader(transition_batch)
        TorchBaseFunction.trainer.fit(self, dataloader)
        if "loss" not in TorchBaseFunction.trainer.logged_metrics.keys():
            raise KeyError("Use self.log('loss', value) in training step")
        return self.trainer.logged_metrics["loss"]


class TorchProcessor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def __call__(self, arg, vectorize=False):
        assert type(arg) is np.ndarray, "Processor argument should be numpy array"
        arg = torch.from_numpy(arg).float()
        arg = arg.view(-1, arg.shape[-1]) if self.state_dim in arg.shape or self.action_dim in arg.shape or vectorize else arg
        return arg
    
    def invert(self, arg:torch.Tensor):
        return arg.numpy()

    def max(self, target, dim, **kwargs):
        values, _ = target.max(dim, **kwargs)
        return values

    def multiply(self, target, *others):
        result = target
        for other in others: result = result.mul(self(other))
        return result

    def gather(self, target, dim, idx):
        idx = torch.from_numpy(idx).view(target.size(0), -1)
        return target.gather(dim, idx)

    def add(self, target, *args):
        result = target
        for arg in args: result += self(arg)
        return result
    
    def reshape(self, target, shape):
        return target.reshape(shape)

