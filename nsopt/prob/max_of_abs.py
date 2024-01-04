from ..opt.shared import (UnconstrainedOptimizable, KnownLipschitzOptimizable,
                          LipschitzConstants, TRAFSStep)
from .utils import UnconstrainedFuncSubDiffHelper, print_once
from ..utils import setup_pyx_import

import numpy as np
import numpy.typing as npt
import attrs

with setup_pyx_import():
    from .kernels import max_of_abs_subd

class MaxOfAbs(UnconstrainedOptimizable, KnownLipschitzOptimizable):
    """Test problem taken from the paper Quasi-monotone Subgradient Methods for
    Nonsmooth Convex Minimization:

    max(abs(x[0]), max(abs(x[i] - 2x[i-1]) for i in 1..n))
    """

    x0: npt.NDArray

    @attrs.frozen
    class SubDiff(KnownLipschitzOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper()

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

            if self._helper.cvx_hull_prefer_socp or self.abs1.size > 100:
                print_once('Use SOCP to solve dx')
                return self._helper.reduce_from_cvx_hull_socp(
                    G, df_lb_thresh, norm_bound, state,
                    force_clarabel=True,
                )

            # QP primal is slower than clarabel for high dimensions but seems to
            # have better solution quality
            print_once('Use QP-direct to solve dx')
            return self._helper.reduce_from_cvx_hull_qp_direct(
                G, df_lb_thresh, norm_bound, state,
            )

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

    def eval_cvx_params(self) -> LipschitzConstants:
        n = self.x0.size
        return LipschitzConstants(
            D=self.eval(self.x0),
            R=np.sqrt(n),
            L=np.sqrt(5),
            alpha=0,
            beta=0)

    def get_optimal_value(self):
        return 0.0

    def __repr__(self):
        return f'MaxOfAbs(n={self.x0.size})'
