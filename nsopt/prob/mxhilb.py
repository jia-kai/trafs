from ..opt.shared import UnconstrainedOptimizable, LipschitzConstants, TRAFSStep
from .utils import UnconstrainedFuncSubDiffHelper, mosek, print_once
from ..utils import setup_pyx_import

import numpy as np
import numpy.typing as npt
import attrs

with setup_pyx_import():
    from .mxhilb_utils import mxhilb_comp_batch, mxhilb_subd

class GeneralizedMXHILB(UnconstrainedOptimizable):
    """the generalized MXHILB, see [1]

    [1] Haarala, M. and Miettinen, K. and Maekelae, M. M., New limited memory
    bundle method for large-scale nonsmooth optimization.
    """

    x0: npt.NDArray

    @attrs.frozen
    class SubDiff(UnconstrainedOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper(
            qp_eps=1e-3,
            qp_iters=200,
        )

        fval: float
        recips: npt.NDArray
        comp: npt.NDArray

        def take_arbitrary(self):
            G = mxhilb_subd(1e-9, self.fval, self.recips, self.comp)
            if G is None:
                return TRAFSStep.make_zero(self.comp.size, True)
            return np.mean(G, axis=1)

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:
            G = mxhilb_subd(subg_slack, self.fval, self.recips, self.comp)
            if G is None:
                return TRAFSStep.make_zero(self.comp.size, True)

            if G.shape[1] == 1:
                return self._helper.reduce_with_min_grad(
                    G[:, 0], df_lb_thresh, norm_bound)

            # MOSEK is faster than PIQP
            # PIQP faster than Clarabel
            if mosek is None:
                print_once('Use PIQP to solve min norm grad')
                return self._helper.reduce_from_cvx_hull_qp(
                    G, df_lb_thresh, norm_bound, state)
            print_once('Use MOSEK to solve best dx')
            return self._helper.reduce_from_cvx_hull_socp(
                G, df_lb_thresh, norm_bound, state)


    def __init__(self, n: int):
        self.x0 = np.ones(n, dtype=np.float64)
        self._recips = np.reciprocal(np.arange(1, n * 2, dtype=np.float64))

    def eval(self, x: npt.NDArray, *, need_grad=False):
        assert x.ndim == 1
        assert x.size >= 1
        comp = np.squeeze(mxhilb_comp_batch(self._recips, x[np.newaxis, :]),
                          axis=1)
        fval = np.abs(comp).max()
        if need_grad:
            return fval, self.SubDiff(fval, self._recips, comp)
        else:
            return fval

    def eval_batch(self, x: npt.NDArray):
        comp = mxhilb_comp_batch(self._recips, x)
        return np.abs(comp).max(axis=0)

    def eval_cvx_params(self) -> LipschitzConstants:
        n = self.x0.size
        return LipschitzConstants(
            D=self.eval(self.x0),
            R=np.sqrt(n),
            L=np.linalg.norm(self._recips[:n], ord=2),
            alpha=0,
            beta=0)

    def __repr__(self):
        return f'MXHILB(n={self.x0.size})'
