from ..opt.shared import (UnconstrainedOptimizable, StronglyConvexOptimizable,
                          StronglyConvexParams, TRAFSStep)
from .utils import UnconstrainedFuncSubDiffHelper
from ..utils import setup_pyx_import

import numpy as np
import numpy.typing as npt
import attrs

with setup_pyx_import():
    from .max_of_abs_utils import max_of_abs_subd

class MaxOfAbs(UnconstrainedOptimizable, StronglyConvexOptimizable):
    """Test problem taken from the paper Quasi-monotone Subgradient Methods for
    Nonsmooth Convex Minimization:

    max(abs(x[0]), max(abs(x[i] - 2x[i-1]) for i in 1..n))
    """

    x0: npt.NDArray

    @attrs.frozen
    class SubDiff(StronglyConvexOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper(f_lb_norm_bound=4)

        fval: np.float64
        x0: np.float64
        abs1: npt.NDArray
        abs1_inp: npt.NDArray

        def take_arbitrary(self):
            G = max_of_abs_subd(0, self.fval, self.x0, self.abs1, self.abs1_inp)
            if G is None:
                return np.zeros(self.abs1.shape[0] + 1, dtype=np.float64)
            grad = np.ascontiguousarray(G.mean(axis=1))
            return np.squeeze(grad, 1)

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:
            G = max_of_abs_subd(
                subg_slack, self.fval, self.x0, self.abs1, self.abs1_inp)
            if G is None:
                return TRAFSStep.make_zero(self.abs1.shape[0] + 1, True)

            if G.shape[1] == 1:
                g = G.toarray()[:, 0]
                return self._helper.reduce_with_min_grad(g, df_lb_thresh,
                                                         norm_bound)

            # the QP solver has some numerical issues when n = 16; use the SOCP
            # instead
            return self._helper.reduce_from_cvx_hull_socp(
                G, df_lb_thresh, norm_bound, state)


    def __init__(self, n: int):
        self.x0 = np.ones(n, dtype=np.float64)

    def eval(self, x: npt.NDArray, *, need_grad=False):
        assert x.ndim == 1
        assert x.size >= 2
        abs1_inp = x[1:] - 2*x[:-1]
        abs1 = np.abs(abs1_inp)
        fval = np.maximum(np.abs(x[0]), abs1.max())
        if need_grad:
            return fval, self.SubDiff(fval, x[0], abs1, abs1_inp)
        return fval

    def eval_batch(self, x: npt.NDArray):
        abs0 = np.abs(x[:, 0])
        abs1 = np.abs(x[:, 1:] - 2*x[:, :-1])
        return np.maximum(abs0, abs1.max(axis=1))

    def eval_cvx_params(self) -> StronglyConvexParams:
        n = self.x0.size
        return StronglyConvexParams(
            D=self.eval(self.x0),
            R=np.sqrt(n),
            L=np.sqrt(5),
            alpha=0,
            beta=0)
