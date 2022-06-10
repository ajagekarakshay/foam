from .td_learning import QLearning, QLearningwithPolicy
from .policy_gradient import DeterministicPG
from .sarsa import SARSA

__all__ = (
    'QLearning',
    'QLearningwithPolicy',
    'DeterministicPG'
)