import torch
import tensorflow as tf
from .base import Objective
from ..core.other import _bind_method

class DeterministicPG(Objective):
    def __init__(self, pi, q):
        super().__init__()
        self.pi = pi
        self.q = q
        self.proc = pi.proc

    def flush(self):
        self.q._clear()
        return self.pi._clear()

    def compute_loss(self, transition_batch):
        Apred = self.pi(transition_batch.S)
        Q_SApred_neg = -self.q(transition_batch.S, Apred)
        return self.proc.mean(Q_SApred_neg)
    
    def process_grads(self, grads):
        # Do nothing
        return grads

    def compute_grads(self, loss, **kwargs):
        return self.pi._gradients(loss, **kwargs)

    def update_step(self, transition_batch, idx=None):
        optimizer, tape = self.flush()
        loss = self.compute_loss(transition_batch)
        grads = self.compute_grads(loss, tape=tape)
        grads = self.process_grads(grads)
        self.pi._apply_grads(optimizer, grads)
        return loss
    
    def update(self, transition_batch):
        # Perform training step
        loss = self.update_step(transition_batch)
        return {f"{self.__class__.__name__}/loss": loss}