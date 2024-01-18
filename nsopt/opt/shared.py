import attrs
import numpy as np
import numpy.typing as npt

import typing
from abc import ABCMeta, abstractmethod

@attrs.frozen
class LipschitzConstants:
    """Lipschitz constants (and other related information) of a convex
    function"""

    D: float
    """lower bound of initial optimality gap"""

    R: float
    """diameter of the feasible set"""

    L: float
    """Lipschitz constant"""

    alpha: float
    """strong convexity"""

    beta: float
    """weak smoothness"""


@attrs.frozen
class OptimizationResult:
    """final optimization result"""

    optimal: bool
    """whether the solution is optimal (i.e., whether it converged within the
    requested tolerance)"""

    x: npt.NDArray
    """the optimal x"""

    fval: float
    """the final objective value"""

    fval_hist: npt.NDArray = attrs.field(
        default=np.array([], dtype=np.float64),
        repr=False)
    """history of function values"""

    iter_times: npt.NDArray = attrs.field(
        default=np.array([], dtype=np.float64),
        repr=False)
    """finish time of each iteration (relative to start time)"""

    iters: int = 0
    """number of iterations"""

    ls_tot_iters: int = 0
    """total number of iterations for line search, summed over all x
    iterations"""

    time: float = 0.0
    """total running time"""


@attrs.define(slots=True)
class TRAFSStep:
    """a step of the TRAFS method
    Note: ``dx_dg <= 0`` should always hold. If ``dx_dg == 0`` but
    ``df_lb_is_global`` is False, then we assume we have numerical issues.
    """

    dx: npt.NDArray
    """the step vector"""

    dx_dg: float
    """max of dx @ dg with dg in subdifferential"""

    df_lb: float
    """lower bound of the objective relative to current value minus slack of
    functional subdifferential; should be non-positive"""

    df_lb_is_global: bool
    """whether the lower bound is global (True) or local (False); see
    ``df_lb_thresh`` in :func:`Optimizable.SubDiff.reduce_trafs`.
    """

    @classmethod
    def make_zero(cls, dim: int, df_lb_is_global: bool) -> "TRAFSStep":
        return cls(np.zeros(dim, dtype=np.float64), 0, 0, df_lb_is_global)


class Optimizable(metaclass=ABCMeta):
    """optimizable function"""

    x0: npt.NDArray
    """initial point"""

    pgd_default_lr: float = 1.0
    """default learning rate for projected gradient descent"""

    class SubDiff(metaclass=ABCMeta):
        """the functional subdifferential at a point"""

        @abstractmethod
        def take_arbitrary(self) -> npt.NDArray:
            """take an arbitrary subgradient"""

        @abstractmethod
        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:
            """reduce the subdifferential to a step vector per the TRAFS method

            :param subg_slack: the functional subgradient bound (i.e., function
                value should be no smaller than the linear approximation minus
                this bound; the smoothness-predicted quadratic upper bound
                should be valid within a ball of a radius at least linear to
                this bound). The implementation should try to use a smaller
                ``subg_slack`` value if ``norm_bound`` is small.
            :param df_lb_thresh: compute a global ``df_lb`` value if ``df_lb >=
                df_lb_thresh``; otherwise ``df_lb`` can be local
            :param norm_bound: the bound of the norm of the step vector
            :param state: a dictionary for storing temporary variables of the
                method
            """

    @typing.overload
    @abstractmethod
    def eval(self, x: npt.NDArray) -> float:
        pass

    @typing.overload
    @abstractmethod
    def eval(self, x: npt.NDArray, *, need_grad: typing.Literal[False]) -> float:
        pass

    @typing.overload
    @abstractmethod
    def eval(self, x: npt.NDArray, *, need_grad: typing.Literal[True]) -> tuple[
            float, SubDiff]:
        pass

    @abstractmethod
    def eval(self, x: npt.NDArray, *, need_grad: bool=False) -> typing.Union[
            float, tuple[float, SubDiff]]:
        """function value at a single point
        :param x: (n, ) ndarray
        """

    @abstractmethod
    def eval_batch(self, x: npt.NDArray) -> npt.NDArray:
        """function value at a batch of points
        :param x: (batch_size, n) ndarray
        """

    @abstractmethod
    def proj(self, x: npt.NDArray) -> npt.NDArray:
        """projection onto the feasible set"""

    def get_optimal_value(self) -> typing.Optional[float]:
        """get the optimal value if available"""
        return None


class KnownLipschitzOptimizable(Optimizable):
    """problems with known Lipschitz constants"""

    @abstractmethod
    def eval_cvx_params(self) -> LipschitzConstants:
        """optional implementation to compute parameters for strongly convex
        functions"""


class UnconstrainedOptimizable(Optimizable):
    """unconstrained optimization problem"""

    def proj(self, x: npt.NDArray) -> npt.NDArray:
        return x


class ProximalGradOptimizable(UnconstrainedOptimizable):
    """functions optimizable with proximal gradient descent for unconstrained
    optimization

    min  F(x) = f(x) + g(x) where f is convex and smooth and g is convex

    Note that we have assumed that the problem is unconstrained.
    """
    @typing.overload
    @abstractmethod
    def prox_f(self, x: npt.NDArray, *,
               need_grad: typing.Literal[False]) -> float:
        pass

    @typing.overload
    @abstractmethod
    def prox_f(self, x: npt.NDArray, *,
               need_grad: typing.Literal[True]) -> tuple[float, npt.NDArray]:
        pass

    @abstractmethod
    def prox_f(self, x: npt.NDArray, *, need_grad: bool=False) -> typing.Union[
            float, tuple[float, npt.NDArray]]:
        """evaluate f(x) and grad of f at x

        :param x: (n, ) ndarray
        """

    @abstractmethod
    def prox_f_batch(self, x: npt.NDArray) -> npt.NDArray:
        """evaluate f(x) for a batch of points

        :param x: (batch_size, n) ndarray
        """
    @abstractmethod
    def prox_g(self, x: npt.NDArray) -> float:
        """evaluate g(x)

        :param x: (n, ) ndarray
        """

    @abstractmethod
    def prox_minx(self, ynew: npt.NDArray, L: float) -> npt.NDArray:
        """compute argmin Q(x, y, L) w.r.t. x
        where Q(x, y, L) = f(y) + <grad f(y), x - y> + L/2 ||x - y||^2 + g(x),
        which is equal to argmin g(x) + L/2 ||x - (y - 1/L grad f(y))||^2

        :param ynew: (n, ) ndarray, the value of y - 1/L grad f(y)
        :param L: learning rate
        """
