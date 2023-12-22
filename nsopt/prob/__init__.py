"""benchmark problems"""

__all__ = ['LassoRegression', 'LassoClassification', 'MaxOfAbs',
           'GeneralizedMXHILB', 'DistanceGame']

from .l1_reg import LassoRegression, LassoClassification
from .max_of_abs import MaxOfAbs
from .mxhilb import GeneralizedMXHILB
from .distance_game import DistanceGame
