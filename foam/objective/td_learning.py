import gym
from .base import Objective
from ..core.other import _bind_method

class TDLearning(Objective):
    def __init__(self, q, q_targ=None, pi_targ = None, loss_function=None):
        self.q = q
        self.q_targ = q_targ
        self.pi_targ = pi_targ
        self.loss_function = loss_function
        self.proc = q.proc

    def flush(self):
        return self.q._clear()
        
    def process_grads(self, grads):
        # Do nothing
        return grads

    def update_step(self, transition_batch, idx=None):
        optimizer, tape = self.flush()
        loss = self.compute_loss(transition_batch)
        grads = self.compute_grads(loss, tape=tape)
        grads = self.process_grads(grads)
        self.q._apply_grads(optimizer, grads)
        return loss

    def update(self, transition_batch):
        # Perform training step
        loss = self.update_step(transition_batch)
        return {f"{self.__class__.__name__}/loss": loss}


class QLearning(TDLearning):
    def __init__(self, q, q_targ=None, pi_targ=None, loss_function=None):
        super().__init__(q, q_targ=q_targ, pi_targ=pi_targ, loss_function=loss_function)
        assert isinstance(self.q.env.action_space, gym.spaces.Discrete)

    def compute_target(self, transition_batch):
        Q_targets_next =  self.q_targ(transition_batch.S_next, training=False) 
        Q_targets_next = self.proc.max(Q_targets_next, dim=1)
        Q_targets = self.proc.multiply(Q_targets_next, 1-transition_batch.Done, transition_batch.Gamma)
        Q_targets = self.proc.add( Q_targets, transition_batch.R )
        return self.proc.reshape(Q_targets, (-1,1))

    def compute_loss(self, transition_batch):
        targets = self.compute_target(transition_batch)
        Q_expected = self.q(transition_batch.S)
        Q_expected = self.q.proc.gather(Q_expected, dim=1, idx=transition_batch.A)
        loss = self.loss_function(Q_expected, targets)    
        return loss

    def compute_grads(self, loss, **kwargs):
        return self.q._gradients(loss, **kwargs)

    

class QLearningwithPolicy(TDLearning):
    def __init__(self, q, q_targ=None, pi_targ=None, loss_function=None):
        super().__init__(q, q_targ=q_targ, pi_targ=pi_targ, loss_function=loss_function)
    
    def compute_target(self, transition_batch):
        A_next = self.pi_targ(transition_batch.S_next, training=False)
        Q_sa_next = self.q_targ(transition_batch.S_next, A_next, training=False).ravel()
        Q_targets = self.proc.multiply(Q_sa_next, 1-transition_batch.Done, transition_batch.Gamma)
        Q_targets = self.proc.add( Q_targets, transition_batch.R )
        return self.proc.reshape(Q_targets, (-1,1))

    def compute_loss(self, transition_batch):
        targets = self.compute_target(transition_batch)
        Q_expected = self.q(transition_batch.S, transition_batch.A)
        assert Q_expected.shape == targets.shape
        loss = self.loss_function(Q_expected, targets)
        return loss

    def compute_grads(self, loss, **kwargs):
        return self.q._gradients(loss, **kwargs)