import numpy as np
import tensorflow as tf
from tensorflow import keras
import copy
from .other import _set_dimensions


class TFBaseFunction(keras.Model):
    
    def __init__(self, env, model, optimizer, optim_args={}, trainer_args={}):
        super().__init__()
        self.env = env
        self.model = model
        self.optimizer = optimizer(**optim_args)
        self.optim_args = optim_args
        _set_dimensions(self, env)
        self.proc = TFProcessor(self.state_dim, self.action_dim)
        super().compile()

    def call(self, s, a=None, training=True):
        raise NotImplementedError

    def soft_update(self, other, tau):
        updated_params = [tau*new_param + (1.0-tau)*old_param \
                    for old_param, new_param in zip(self.model.get_weights(), other.model.get_weights())]
        self.model.set_weights(updated_params)

    def copy(self, deep=False):
        if deep:
            model_copy = copy.deepcopy(self.model)
            optimizer = self.optimizer
            return self.__class__(self.env, model_copy, type(optimizer), optim_args=optimizer.defaults)
        return copy.copy(self)
    
    def _clear(self):
        optimizer = self.optimizer
        tape = tf.GradientTape()
        tape._push_tape()
        return optimizer, tape

    def _gradients(self, loss, **kwargs):
        tape = kwargs["tape"]
        tape._pop_tape()
        grads = tape.gradient(loss, self.model.trainable_weights)
        return grads

    def _apply_grads(self, optimizer, grads=None):
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def _update(self, transition_batch, **kwargs):
        loss = self.train_step(transition_batch, **kwargs)
        return loss


class TFProcessor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def __call__(self, arg, vectorize=False, **kwargs):
        #assert type(arg) is np.ndarray, "Processor argument should be numpy array"
        arg = tf.convert_to_tensor(arg, dtype=tf.float32) if type(arg) is np.ndarray else arg
        arg = tf.reshape(arg, (-1, arg.shape[-1])) if self.state_dim in arg.shape \
                or self.action_dim in arg.shape or vectorize else arg
        return arg
    
    def invert(self, arg):
        return arg.numpy()

    def max(self, target, dim, **kwargs):
        values = tf.math.reduce_max(target, axis=dim, **kwargs)
        return values

    def multiply(self, target, *others):
        result = target
        for other in others: result = tf.math.multiply(result, other)
        return result

    def gather(self, target, dim, idx):
        idx = idx.reshape(target.shape[0], -1)
        return tf.gather(target, idx, batch_dims=dim)

    def add(self, target, *args):
        result = target
        for arg in args: result += self(arg)
        return result

    def reshape(self, target, shape):
        return tf.reshape(target, shape)

    def mean(self, target, dim, **kwargs):
        return tf.math.reduce_mean(target, axis=dim, **kwargs)