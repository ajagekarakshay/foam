from gym import spaces
from gym.spaces.discrete import Discrete

def _set_dimensions(obj, env):
    if isinstance(env.observation_space, spaces.Box) or isinstance(env.observation_space, spaces.MultiBinary):
        obj.state_dim = env.observation_space.shape[-1]
    else:
        print("Cannot figure out state size : Think a little")
    
    if isinstance(env.action_space, spaces.Box):
        obj.action_dim = env.action_space.shape[-1]
    elif isinstance(env.action_space, spaces.Discrete):
        obj.action_dim = env.action_space.n
    else:
        print("Cannot figure out action size : Think a little")

def _bind_method(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)

