"""benchmark problems from [1]

[1] Haarala, M. and Miettinen, K. and Maekelae, M. M., New limited memory
bundle method for large-scale nonsmooth optimization. """

from ..opt.shared import UnconstrainedOptimizable, LipschitzConstants, TRAFSStep
from .utils import UnconstrainedFuncSubDiffHelper, mosek, print_once
from ..utils import setup_pyx_import

import numpy as np
import numpy.typing as npt
import attrs
import typing

with setup_pyx_import():
    from .kernels import (
        mxhilb_comp_batch, mxhilb_subd, chained_lq_subd, chained_cb3_I_subd)

class MaxQ(UnconstrainedOptimizable):
    """max x_i^2"""
    x0: npt.NDArray

    @attrs.frozen
    class SubDiff(UnconstrainedOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper()

        fval: float
        x: npt.NDArray
        comp: npt.NDArray

        def take_arbitrary(self):
            ret = np.zeros_like(self.comp)
            ret[np.argmax(self.comp)] = 2 * self.x[np.argmax(self.comp)]
            return ret

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:

            act_mask = (self.fval - self.comp) <= subg_slack
            act_idx = np.flatnonzero(act_mask)
            a = 2 * self.x[act_idx]
            if np.abs(a).min() <= 1e-10:
                return TRAFSStep.make_zero(self.x.size, True)
            # min_{p in Delta(n)} sum (a_i p_i)^2
            a2r = np.reciprocal(np.square(a))
            gc = np.zeros_like(self.x)
            gc[act_idx] = (a * a2r) / a2r.sum()
            return self._helper.reduce_with_min_grad(
                gc, df_lb_thresh, norm_bound,
                dx_dg_fn=lambda dx: (a * dx[act_idx]).max()
            )

    def __init__(self, n: int):
        x0 = np.arange(1, n + 1, dtype=np.float64)
        x0[n // 2:] *= -1
        x0 *= np.sqrt(n) / np.linalg.norm(x0, ord=2)
        self.x0 = x0

    def eval(self, x: npt.NDArray, *, need_grad=False):
        assert x.ndim == 1
        assert x.size >= 1
        comp = np.square(x)
        fval = comp.max()
        if need_grad:
            return fval, self.SubDiff(fval, x, comp)
        else:
            return fval

    def eval_batch(self, x: npt.NDArray):
        return np.square(x).max(axis=1)

    def get_optimal_value(self):
        return 0.0

    def __repr__(self):
        return f'MaxQ(n={self.x0.size})'


class MXHILB(UnconstrainedOptimizable):
    """max |c_i| where c = Mx, M is the Hilbert matrix"""
    x0: npt.NDArray

    @attrs.frozen
    class SubDiff(UnconstrainedOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper()

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
            if mosek is None or self._helper.cvx_hull_prefer_qp:
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

    def get_optimal_value(self):
        return 0.0

    def __repr__(self):
        return f'MXHILB(n={self.x0.size})'


class ChainedLQ(UnconstrainedOptimizable):
    """max (-x_i - x_{i+1}, -x_i - x_{i+1} + x_i^2 + x_{i+1}^2 - 1)"""
    x0: npt.NDArray

    @attrs.frozen
    class SubDiff(UnconstrainedOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper(
            socp_check_succ_rate=0.8,
        )

        x: npt.NDArray
        comp: npt.NDArray
        """[n, 2] array of the components of the max"""

        def take_arbitrary(self):
            act_idx = np.argmax(self.comp, axis=1)
            ret = np.zeros_like(self.x)
            act0 = np.flatnonzero(act_idx == 0)
            act1 = np.flatnonzero(act_idx == 1)
            ret[act0] += -1
            ret[act0 + 1] += -1
            ret[act1] += -1 + 2 * self.x[act1]
            ret[act1 + 1] += -1 + 2 * self.x[act1 + 1]
            return ret

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:

            desc = chained_lq_subd(subg_slack, self.x, self.comp)

            # clarabel performs better on sparse problems
            return self._helper.reduce_from_multi_cvx_hull_socp(
                desc, df_lb_thresh, norm_bound, state,
                force_clarabel=True,
            )


    def __init__(self, n: int):
        self.x0 = np.full(n, -0.5, dtype=np.float64)

    def eval(self, x: npt.NDArray, *, need_grad=False):
        x2 = np.square(x)
        c0 = -x[:-1] - x[1:]
        c1 = c0 + x2[:-1] + x2[1:] - 1
        fval = np.maximum(c0, c1).sum()
        if need_grad:
            comp = np.stack([c0, c1], axis=1)
            return fval, self.SubDiff(x, comp)
        else:
            return fval

    def eval_batch(self, x: npt.NDArray):
        x2 = np.square(x)
        comp = x2[:, :-1] + x2[:, 1:] - 1
        fval = (-x.sum(axis=1) * 2 + x[:, 0] + x[:, -1] +
                np.maximum(comp, 0).sum(axis=1))
        return fval

    def get_optimal_value(self):
        return -np.sqrt(2) * (self.x0.size - 1)

    def __repr__(self):
        return f'ChainedLQ(n={self.x0.size})'


class ChainedCB3I(UnconstrainedOptimizable):
    """max(x_i^4 + x_{i+1}^2, (2-x_i)^2 + (2-x_{i+1})^2,
        2e^{-x_i + x_{i+1}})"""
    x0: npt.NDArray

    pgd_default_lr = 1e-2

    @attrs.frozen
    class SubDiff(UnconstrainedOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper(
            socp_check_succ_rate=0.6,
        )

        x: npt.NDArray
        comp: npt.NDArray
        """[n, 3] array of the components"""

        def take_arbitrary(self):
            cidx = np.argmax(self.comp, axis=1)
            act0 = np.flatnonzero(cidx == 0)
            act1 = np.flatnonzero(cidx == 1)
            act2 = np.flatnonzero(cidx == 2)
            grad = np.zeros_like(self.x)
            x = self.x

            grad[act0] += 4 * np.power(x[act0], 3)
            grad[act0 + 1] += 2 * x[act0 + 1]

            grad[act1] -= 4 - 2 * x[act1]
            grad[act1 + 1] -= 4 - 2 * x[act1 + 1]

            c2 = self.comp[act2, 2]
            grad[act2] -= c2
            grad[act2 + 1] += c2

            return grad

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:
            desc = chained_cb3_I_subd(subg_slack, self.x, self.comp)

            return self._helper.reduce_from_multi_cvx_hull_socp(
                desc, df_lb_thresh, norm_bound, state,
                force_clarabel=True,
            )

    def __init__(self, n: int):
        self.x0 = np.full(n, 2, dtype=np.float64)

    def eval(self, x: npt.NDArray, *, need_grad=False):
        xi = x[:-1]
        xi1 = x[1:]
        c0 = np.power(xi, 4) + np.square(xi1)
        c1 = np.square(2 - xi) + np.square(2 - xi1)
        c2 = 2 * np.exp(xi1 - xi)
        fval = np.maximum(np.maximum(c0, c1), c2).sum()
        if need_grad:
            comp = np.stack([c0, c1, c2], axis=1)
            return fval, self.SubDiff(x, comp)
        else:
            return fval

    def eval_batch(self, x: npt.NDArray):
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        c0 = np.power(xi, 4) + np.square(xi1)
        c1 = np.square(2 - xi) + np.square(2 - xi1)
        c2 = 2 * np.exp(xi1 - xi)
        fval = np.maximum(np.maximum(c0, c1), c2).sum(axis=1)
        return fval

    def get_optimal_value(self):
        return 2 * (self.x0.size - 1)

    def __repr__(self):
        return f'ChainedCB3I(n={self.x0.size})'


class ChainedCB3II(UnconstrainedOptimizable):
    """max sum(x_i^4 + x_{i+1}^2, (2-x_i)^2 + (2-x_{i+1})^2,
        2e^{-x_i + x_{i+1}})"""
    x0: npt.NDArray

    pgd_default_lr = 1e-2

    @attrs.frozen
    class SubDiff(UnconstrainedOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper()

        grads_fn: list[typing.Callable[[], npt.NDArray]]
        """callables to compute the component gradients"""

        comp: npt.NDArray
        """[3] array of the components"""

        def take_arbitrary(self):
            return self.grads_fn[np.argmax(self.comp)]()

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:
            comp = self.comp
            grads_fn = self.grads_fn
            fval = comp.max()
            act_mask = (fval - comp) <= subg_slack
            G = []
            for i in range(comp.size):
                if act_mask[i]:
                    G.append(grads_fn[i]())

            if len(G) == 1:
                return self._helper.reduce_with_min_grad(
                    G[0], df_lb_thresh, norm_bound)

            G = np.stack(G, axis=1)
            return self._helper.reduce_from_cvx_hull_qp(
                G, df_lb_thresh, norm_bound, state,
            )

    def __init__(self, n: int):
        self.x0 = np.full(n, 2, dtype=np.float64)

    def eval(self, x: npt.NDArray, *, need_grad=False):
        xi = x[:-1]
        xi1 = x[1:]
        c0 = np.power(xi, 4) + np.square(xi1)
        c1 = np.square(2 - xi) + np.square(2 - xi1)
        c2 = 2 * np.exp(xi1 - xi)
        comp = np.array([c0.sum(), c1.sum(), c2.sum()], dtype=np.float64)
        fval = comp.max()

        def g0():
            g = np.zeros_like(x)
            g[:-1] += 4 * np.power(xi, 3)
            g[1:] += 2 * xi1
            return g
        def g1():
            g = np.zeros_like(x)
            g[:-1] -= 4 - 2 * xi
            g[1:] -= 4 - 2 * xi1
            return g
        def g2():
            g = np.zeros_like(x)
            g[:-1] -= c2
            g[1:] += c2
            return g

        if need_grad:
            return fval, self.SubDiff([g0, g1, g2], comp)
        else:
            return fval

    def eval_batch(self, x: npt.NDArray):
        xi = np.ascontiguousarray(x[:, :-1])
        xi1 = np.ascontiguousarray(x[:, 1:])
        c0 = (np.power(xi, 4) + np.square(xi1)).sum(axis=1)
        c1 = (np.square(2 - xi) + np.square(2 - xi1)).sum(axis=1)
        c2 = (2 * np.exp(xi1 - xi)).sum(axis=1)
        fval = np.maximum(np.maximum(c0, c1), c2)
        return fval

    def get_optimal_value(self):
        return 2 * (self.x0.size - 1)

    def __repr__(self):
        return f'ChainedCB3II(n={self.x0.size})'
