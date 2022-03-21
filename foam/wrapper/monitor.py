import numpy as np
import time
import gym
from collections import deque
from tqdm.notebook import tqdm

__all__ = (
    'MonitorStats',
)

class StreamingSample:
    def __init__(self, maxlen, random_seed=None):
        self._deque = deque(maxlen=maxlen)
        self._count = 0
        self._rnd = np.random.RandomState(random_seed)

    def reset(self):
        self._deque = deque(maxlen=self.maxlen)
        self._count = 0

    def append(self, obj):
        self._count += 1
        self._deque.append(obj)

    @property
    def values(self):
        return list(self._deque)
    @property
    def maxlen(self):
        return self._deque.maxlen

    def __len__(self):
        return len(self._deque)

    def __bool__(self):
        return bool(self._deque)


class MonitorStats(gym.Wrapper):
    def __init__(
            self, env,
            logger,
            log_all_metrics=False,
            smoothing_window=10,
            **logger_kwargs):

        super().__init__(env)
        self.log_all_metrics = log_all_metrics
        self.smoothing_window = smoothing_window
        self.logger = logger(**logger_kwargs)
        self.reset_global()

    def reset_global(self):
        self.T = 0
        self.ep = 0
        self.t = 0
        self.G = 0.0
        self.avg_G = 0.0
        self._n_avg_G = 0.0
        self._ep_starttime = time.time()
        self._ep_metrics = {}
        self._ep_actions = StreamingSample(maxlen=1000)
        self._scores = deque(maxlen=self.smoothing_window)
        self._period = {'T': {}, 'ep': {}}

    def reset(self):
        # write logs from previous episode:
        if self.ep:
            self._scores.append(self.G)
            self._write_episode_logs()
            tqdm.write(f"\rEpisode: {self.ep} | Timesteps: {self.t} |\
                    Avg_reward: {self.avg_r} | Returns: {self.G}",
                    )
        # increment global counters:
        self.T += 1
        self.ep += 1
        # reset episodic counters:
        self.t = 0
        self.G = 0.0
        self._ep_starttime = time.time()
        self._ep_metrics = {}
        self._ep_actions.reset()

        return self.env.reset()

    @property
    def dt_ms(self):
        if self.t <= 0:
            return np.nan
        return 1000 * (time.time() - self._ep_starttime) / self.t

    @property
    def avg_r(self):
        if self.t <= 0:
            return np.nan
        return self.G / self.t

    @property
    def smooth_G(self):
        return np.mean(self._scores)

    def step(self, a):
        self._ep_actions.append(a)
        s_next, r, done, info = self.env.step(a)
        if info is None:
            info = {}
        info['monitor'] = {'T': self.T, 'ep': self.ep}
        self.t += 1
        self.T += 1
        self.G += r

        return s_next, r, done, info

    def record_metrics(self, metric:dict):
        self.logger.log_metrics(metric, step=self.T)

    def _write_episode_logs(self):
        if self.logger is not None:
            metrics = {
                'episode/episode': self.ep,
                'episode/avg_reward': self.avg_r,
                'episode/return': self.G,
                'episode/smooth_return': self.smooth_G,
                'episode/steps': self.t,
                'episode/avg_step_duration_ms': self.dt_ms}
            self.logger.log_metrics(
                                    metrics, 
                                    step=self.T )
            if self._ep_actions:
                if isinstance(self.action_space, gym.spaces.Discrete):
                    bins = np.arange(self.action_space.n + 1)
                else:
                    bins = 64 # Default in wandb histogram 
                self.logger.log_histogram({"episode/actions":self._ep_actions.values}, 
                                            step=self.T, bins=bins)