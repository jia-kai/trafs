"""benchmark problems"""

__all__ = [
    "MXHILB",
    "ChainedCB3I",
    "ChainedCB3II",
    "ChainedLQ",
    "DistanceGame",
    "LassoClassification",
    "LassoRegression",
    "MaxOfAbs",
    "MaxQ",
]

from .distance_game import DistanceGame
from .hmm_bench import MXHILB, ChainedCB3I, ChainedCB3II, ChainedLQ, MaxQ
from .l1_reg import LassoClassification, LassoRegression
from .max_of_abs import MaxOfAbs
