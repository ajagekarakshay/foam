#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[35]:


import numpy as np
import gym
from tqdm.auto import tqdm
import random
import coax
import jax.numpy as jnp
import optax
import haiku as hk
import jax



seed = 3407
random.seed(seed)
# torch.manual_seed(seed)

# In[5]:


env = gym.make('Pendulum-v1')
env.seed(seed)

env_mon = coax.wrappers.TrainMonitor(env, name="pendulum", tensorboard_dir=f"./pendulum")


# %%

def actor(S, is_training):
    layers = (hk.Linear(400), jax.nn.relu,
              hk.Linear(300), jax.nn.relu,
              hk.Linear(1), jnp.tanh,
              hk.Reshape(env.action_space.shape))
    model = hk.Sequential(layers)
    mu = model(S)
    return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}

def critic(S,A, is_training):
    m1 = hk.Sequential((hk.Linear(400), jax.nn.relu))
    xs = m1(S)
    x = jnp.concatenate((xs,A), axis=-1)
    layers = (
              hk.Linear(300), jax.nn.relu,
              hk.Linear(1),
              )
    m2 = hk.Sequential(layers)
    return m2(x).ravel()


# In[31]:

pi = coax.Policy(actor, env_mon)
q = coax.Q(critic, env_mon, action_preprocessor=pi.proba_dist.preprocess_variate)

# target network
q_targ = q.copy(deep=True)
pi_targ = pi.copy(deep=True)


# In[55]:


noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=0.2, theta=0.15)


# In[56]:


tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)


qlearning = coax.td_learning.QLearning(q, pi_targ, q_targ, 
                        loss_function=coax.value_losses.mse, 
                        optimizer=optax.adam(1e-3))

dpg = coax.policy_objectives.DeterministicPG(pi, q_targ, 
                            optimizer=optax.adam(1e-4))


# In[57]:


episodes = 500
for _ in tqdm(range(episodes), desc="Episode"):
    s = env_mon.reset()
    noise.reset()
    

    for t in range(195):
        a = noise(pi(s))
        #a = np.clip(a, -1, 1) ### ############################added cliping
        s_next,r,done,_ = env_mon.step(a)

        tracer.add(s,a,r,done)
        while tracer:
            transition = tracer.pop()
            buffer.add(transition)

        if done:
            break

        s = s_next

        if len(buffer) >= 128:
            #break
            transition_batch = buffer.sample(batch_size=128)
            metrics_q = qlearning.update(transition_batch)
            #print(list(q.parameters())[0])
            metrics_pi = dpg.update(transition_batch)
            #print(list(pi.parameters())[0])
            env_mon.record_metrics(metrics_q)
            env_mon.record_metrics(metrics_pi)

            q_targ.soft_update(q, tau=5E-3)
            #print(list(q_targ.parameters())[0])
            pi_targ.soft_update(pi, tau=5E-3)
            #print(list(pi_targ.parameters())[0])
    
    noise.sigma *= 0.99 ###################
