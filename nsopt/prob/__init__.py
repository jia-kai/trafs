"""benchmark problems"""

__all__ = ['LassoRegression', 'LassoClassification',
           'MaxOfAbs',
           'MaxQ', 'MXHILB', 'ChainedLQ', 'ChainedCB3I', 'ChainedCB3II',
           'DistanceGame']

from .l1_reg import LassoRegression, LassoClassification
from .max_of_abs import MaxOfAbs
from .hmm_bench import MaxQ, MXHILB, ChainedLQ, ChainedCB3I, ChainedCB3II
from .distance_game import DistanceGame
