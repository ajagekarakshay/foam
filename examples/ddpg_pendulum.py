#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import os, wandb
os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()


# In[35]:


import numpy as np
import gym
from tqdm.auto import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import conrl as crl
from conrl import  Q, Policy


seed = 3407
random.seed(seed)
torch.manual_seed(seed)

# In[5]:


env = gym.make('Pendulum-v1')
env.seed(seed)


# In[52]:


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        #self.reset_parameters()

    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) #* 2 ################################################### changed this


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        #self.reset_parameters()

    # def reset_parameters(self):
    #     self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# In[31]:


logger_config = {
    "project":"pendulum"
}

env_mon = crl.wrapper.MonitorStats(env, 
                crl.logger.WandbRL, **logger_config)


# In[53]:


import copy

actor = Actor(3, 1, seed=seed)
critic = Critic(3,1, seed=seed)

actor_copy = Actor(3, 1, seed=seed)
actor_copy.load_state_dict(copy.deepcopy(actor.state_dict()))
critic_copy = Critic(3,1, seed=seed)
critic_copy.load_state_dict(copy.deepcopy(critic.state_dict()))


# In[54]:


q = Q(env_mon, critic, torch.optim.Adam, optim_args=dict(lr=1E-3))
#q_targ = q.copy(deep=True)
q_targ = Q(env_mon, critic_copy, torch.optim.Adam, optim_args=dict(lr=1E-3))

pi = Policy(env_mon, model=actor, optimizer=torch.optim.Adam, optim_args=dict(lr=1E-4))
#pi_targ = pi.copy(deep=True) #deep=True)
pi_targ = Policy(env_mon, model=actor_copy, optimizer=torch.optim.Adam, optim_args=dict(lr=1E-4))


# In[55]:


pi_noisy = crl.explore.OUprocess(pi, 1, seed)


# In[56]:


tape = crl.recorder.NStepRecorder(n=1, gamma=0.99)
buffer = crl.replay.ReplayBuffer(capacity=100000)

qlearning = crl.objective.QLearningwithPolicy(q, 
                                            q_targ=q,
                                            pi_targ=pi_targ, 
                                            loss_function=F.mse_loss)

dpg = crl.objective.DeterministicPG(pi, q)


# In[57]:


episodes = 500
for _ in tqdm(range(episodes), desc="Episode"):
    s = env_mon.reset()
    pi_noisy.reset()
    

    for t in range(195):
        a = pi_noisy.act(s)
        #a = np.clip(a, -1, 1) ### ############################added cliping
        s_next,r,done,_ = env_mon.step(a)

        tape.add(s,a,r,done,s_next)
        while tape:
            transition = tape.pop()
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
    
    pi_noisy.sigma *= 0.99 ###################

wandb.finish()