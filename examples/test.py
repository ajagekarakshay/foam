import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from tqdm.auto import tqdm

import conrl as crl
from conrl import Q

import os, wandb
os.environ["WANDB_API_KEY"] = "29ef10a563126c993b31f306e5a17f55b237e430"
wandb.login()


env = gym.make('CartPole-v0').unwrapped

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(4, 24)
        self.l2 = nn.Linear(24,48)
        self.l4 = nn.Linear(48,2)

    def forward(self, x):
        x = torch.tanh(self.l1(x)) 
        x = torch.tanh(self.l2(x)) 
        return self.l4(x)


logger = crl.logger.WandbRL
logger_config = {
    "project":"test"
}

env_mon = crl.wrapper.MonitorStats(env, 
                        logger=logger, **logger_config)

m = DQN()
q = Q(env_mon, m, torch.optim.Adam, 
      optim_args={"lr":0.01, "weight_decay":0.01})
q_targ = q.copy()

pi = crl.explore.EpsilonGreedy(epsilon=1, q=q)

tape = crl.recorder.NStepRecorder(n=1, gamma=1)
buffer = crl.replay.ReplayBuffer(capacity=100000)

qlearning = crl.objective.QLearning(q, q_targ=q, 
                                    loss_function=F.mse_loss)


episodes = 1000
for _ in tqdm(range(episodes), desc="Episode"):
    
    s = env_mon.reset()

    for t in range(env_mon.spec.max_episode_steps):
        a = pi.act(s)
        s_next,r,done,_ = env_mon.step(a)

        if done:
            r = -200
            tape.add(s,a,r,done,s_next)
            while tape:
                transition = tape.pop()
            buffer.add(transition)
            break

        tape.add(s,a,r,done,s_next)
        while tape:
            transition = tape.pop()
        buffer.add(transition)

        s = s_next

        if len(buffer) >= 32:
            #break
            transition_batch = buffer.sample(batch_size=32)
            qlearning.update(transition_batch)

            pi.update(decay=0.995)

wandb.finish()