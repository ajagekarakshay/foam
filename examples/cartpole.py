import gym
import conrl as crl
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import keras
import torch.nn as nn
import torch.nn.functional as F
import torch

import os, wandb
os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()

env = gym.make('CartPole-v0').unwrapped

class DQN_tf(keras.Model):

    def __init__(self):
        super().__init__()
        self.m = keras.Sequential([
                                keras.Input(env.observation_space.shape),
                                keras.layers.Dense(24, activation="tanh"),
                                keras.layers.Dense(48, activation="tanh"),
                                keras.layers.Dense(2)
                                ])

    def call(self, x):
        return self.m(x)

class DQN_torch(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 24)
        self.l2 = nn.Linear(24,48)
        self.l4 = nn.Linear(48,2)

    def forward(self, x):
        x = torch.tanh(self.l1(x)) #F.relu(self.l1(x))
        x = torch.tanh(self.l2(x)) #F.relu(self.l2(x))
        return self.l4(x)



# mode = "TF"
# model = DQN_tf()
# optimizer = keras.optimizers.Adam
# loss_function = tf.keras.losses.MeanSquaredError()

mode = "Torch"
model = DQN_torch()
optimizer = torch.optim.Adam
loss_function = nn.MSELoss()



logger = crl.logger.WandbRL
logger_config = {
    "project":"cartpole",
    "group":mode,
}

env_mon = crl.wrapper.MonitorStats(env, 
                        logger=logger, **logger_config)


q = crl.Q(env_mon, model, optimizer)
q_targ = q.copy()

pi = crl.explore.EpsilonGreedy(epsilon=1, q=q)

tape = crl.recorder.NStepRecorder(n=1, gamma=1)
buffer = crl.replay.ReplayBuffer(capacity=100000)

qlearning = crl.objective.QLearning(q, q_targ=q, 
                                    loss_function=loss_function)

#%%
episodes = 1000
for _ in tqdm(range(episodes), desc="Episode"):
    
    s = env_mon.reset()

    for t in range(env_mon.spec.max_episode_steps):
        a = pi.act(s)
        s_next,r,done,_ = env_mon.step(a)

        tape.add(s,a,r,done,s_next)
        while tape:
            transition = tape.pop()
        buffer.add(transition)

        s = s_next

        if done:
            break

        if len(buffer) >= 64:
            #break
            transition_batch = buffer.sample(batch_size=64)
            metrics_q = qlearning.update(transition_batch)
            env_mon.record_metrics(metrics_q)
            pi.update(decay=0.995)