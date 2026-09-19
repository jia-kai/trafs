from collections.abc import Sequence
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

__version__: str

class NonnegativeConeT:
    def __init__(self, dim: int) -> None: ...

class ZeroConeT:
    def __init__(self, dim: int) -> None: ...

class SecondOrderConeT:
    def __init__(self, dim: int) -> None: ...

type Cone = NonnegativeConeT | ZeroConeT | SecondOrderConeT

class SolverStatus(Enum):
    Solved: int
    AlmostSolved: int
    MaxIterations: int
    MaxTime: int
    NumericalError: int
    InsufficientProgress: int

class DefaultSettings:
    max_iter: int
    verbose: bool

class DefaultSolution:
    x: Sequence[float]
    z: npt.NDArray[np.float64]
    obj_val: float
    status: SolverStatus

class DefaultSolver:
    def __init__(
        self,
        P: csc_matrix,
        q: npt.NDArray[np.float64],
        A: csc_matrix,
        b: npt.NDArray[np.float64],
        cones: Sequence[Cone],
        settings: DefaultSettings,
    ) -> None: ...
    def solve(self) -> DefaultSolution: ...
