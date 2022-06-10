import gym
from .base import Objective

class SARSA(Objective):
    def __init__(self, q, q_targ=None, loss_function=None):
        self.q = q
        self.q_targ = q if q_targ is None else q_targ
        self.loss_function = loss_function
        self.proc = q.proc

    def compute_target(self, transition_batch):
        assert A_next is not None, "A_next should have been recorded"
        A_next = transition_batch.A_next
        Q_targets_next =  self.q_targ(transition_batch.S_next, A_next, training=False) 
        Q_targets = self.proc.multiply(Q_targets_next, 1-transition_batch.Done, transition_batch.Gamma)
        Q_targets = self.proc.add( Q_targets, transition_batch.R )
        return self.proc.reshape(Q_targets, (-1,1))

    def compute_loss(self, transition_batch):
        targets = self.compute_target(transition_batch)
        Q_expected = self.q(transition_batch.S, transition_batch.A)
        loss = self.loss_function(Q_expected, targets)    
        return loss

    def compute_grads(self, loss, **kwargs):
        return self.q._gradients(loss, **kwargs)

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