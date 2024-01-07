from .simplex import SimplexConstrainedOptimizable
from .utils import make_stable_rng, SOCPSolverBase
from ..opt.shared import TRAFSStep
from ..utils import setup_pyx_import

import attrs
import math
import numpy as np
import numpy.typing as npt

import typing

with setup_pyx_import():
    from .kernels import distance_game_subd

SIMPLEX_DIAMETER = np.sqrt(2) + 1e-4

@attrs.frozen
class DistanceGame(SimplexConstrainedOptimizable):
    """an artificial zero-sum game where the cost is a_i^T x + || B_i x ||_2 +
    KL(x, p_i) where the opponent choses i"""

    x0: npt.NDArray

    m: int
    """number of actions of the opponent"""

    k: int
    """projection dimension"""

    n: int
    """dimension of the decision variable"""

    A: npt.NDArray
    """(m, n) matrix"""

    B: npt.NDArray
    """(m, k, n) tensor"""

    P: npt.NDArray
    """(m, n) matrix"""

    kl_eps: float = 1e-8
    """epsilon for the KL divergence"""

    pgd_default_lr: float = 1e-4

    @attrs.define
    class SubDiff(SimplexConstrainedOptimizable.SubDiff):
        fval: float
        x: npt.NDArray
        A: npt.NDArray
        Ax: npt.NDArray
        B: npt.NDArray
        Bx: npt.NDArray
        Bnorm: npt.NDArray
        P: npt.NDArray
        comp: npt.NDArray
        kl_eps: float

        _prev_trafs_arg: typing.Any = None
        _prev_cvx_hull: typing.Any = None
        _prev_trafs_step: typing.Any = None

        def _get_convex_hull(self, slack: float) -> tuple[
                npt.NDArray, tuple[npt.NDArray, npt.NDArray]]:
            """compute the convex hull of the subdifferential

            :return: (g0, (g1, h1)): the subdifferential is
                g = x @ g0 + y @ (g1 + h1 @ t)
            note that shape of h1 is (act_comp, k, n) and h1@t is reduced in the
            middle dimension (t in unit ball in R^k)
            where x and y are in the simplex of dimensions corresponding to
            number of active components
            """
            delta = self.fval - self.comp
            sub_slack = slack - delta
            act_mask = sub_slack >= 0
            x = self.x
            g_kl = (
                np.log(x + self.kl_eps) + x / (x + self.kl_eps))[np.newaxis]
            g_kl = g_kl - np.log(self.P[act_mask] + self.kl_eps)

            B = self.B[act_mask]
            BtBx = np.squeeze(
                np.transpose(B, (0, 2, 1)) @ self.Bx[act_mask][:, :, np.newaxis],
                axis=2
            )
            return distance_game_subd(
                comp_slack=sub_slack[act_mask],
                g_kl=g_kl,
                A=self.A[act_mask],
                Ax=self.Ax[act_mask],
                B=B,
                BtBx=BtBx,
                Bx_norm=self.Bnorm[act_mask]
            )

        def take_arbitrary(self):
            g0, (g1, _) = self._get_convex_hull(0)
            assert g0.shape[0] > 0 and g1.shape[0] == 0
            return g0[0]

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict) -> TRAFSStep:

            cvx_hull = self._get_convex_hull(subg_slack)
            arg = (df_lb_thresh, norm_bound)
            pch = self._prev_cvx_hull
            def eq(x, y):
                return x.shape == y.shape and np.all(x == y)
            if (arg == self._prev_trafs_arg and
                    (eq(cvx_hull[0], pch[0]) and
                     eq(cvx_hull[1][0], pch[1][0]) and
                     eq(cvx_hull[1][1], pch[1][1]))):
                return self._prev_trafs_step

            self._prev_cvx_hull = cvx_hull
            self._prev_trafs_arg = arg
            self._prev_trafs_step = self._do_reduce_trafs(
                cvx_hull, df_lb_thresh, norm_bound)
            return self._prev_trafs_step

        def _do_reduce_trafs(self,
                cvx_hull, df_lb_thresh: float, norm_bound: float) -> TRAFSStep:

            sol = self._solve_trafs(cvx_hull, norm_bound)

            g0, (g1, h1) = cvx_hull

            df_l = sol.dobj
            df_l_is_g = norm_bound >= SIMPLEX_DIAMETER

            assert df_l <= sol.pobj + 1e-6

            if df_l >= 0:
                # if df_l >= 0, then we know that even if norm_bound <
                # SIMPLEX_DIAMETER we still have the global df_l >= 0
                return TRAFSStep.make_zero(self.x.shape[0], True)

            if df_l >= df_lb_thresh and not df_l_is_g:
                # try to use the solver to compute a global df_l
                sol_g = self._solve_trafs(cvx_hull, -1)
                df_l = sol_g.dobj
                df_l_is_g = True
                assert df_l < 0

            dx = sol.x
            dx_dg = -np.inf
            if g0.shape[0]:
                dx_dg = max(dx_dg, (g0 @ dx).max())
            if g1.shape[0]:
                dx_dg = max(
                    dx_dg,
                    (g1 @ dx + np.linalg.norm(h1 @ dx, axis=1, ord=2)).max()
                )

            if sol.is_optimal:
                np.testing.assert_allclose(dx_dg, sol.pobj, atol=1e-6, rtol=1e-6)

            return TRAFSStep(dx, dx_dg, df_l, df_l_is_g)

        def _solve_trafs(self, cvx_hull, norm_bound: float):
            if norm_bound >= SIMPLEX_DIAMETER:
                norm_bound = -1

            x0 = self.x
            dim = x0.shape[0]
            solver = SOCPSolverBase.make(dim=dim)

            x_low = -x0
            x_high = 1 - x0

            # we scale x to have norm_bound = 1 for better numerical stability
            if norm_bound > 0:
                # clip to [-1, 1] because ||x|| <= 1
                x_low = np.maximum(x_low / norm_bound, -1)
                x_high = np.minimum(x_high / norm_bound, 1)
                solver.add_x_norm_bound()

            (solver
             .add_x_lower(x_low)
             .add_x_higher(x_high)
             .add_eq(np.ones((1, dim), dtype=np.float64),
                     np.zeros(1, dtype=np.float64)))

            g0, (g1, h1) = cvx_hull
            socp_pdim = h1.shape[1]
            assert g0.ndim == 2 and g0.shape[1] == dim
            assert g1.ndim == 2 and g1.shape[1] == dim
            assert h1.shape == (g1.shape[0], socp_pdim, dim)

            # g0 @ x <= u
            if g0.shape[0]:
                solver.add_ineq(g0)

            for i in range(g1.shape[0]):
                solver.add_socp(g1[i], h1[i])

            ret = solver.solve()
            np.testing.assert_allclose(ret.x.sum(), 0, atol=1e-5)
            if norm_bound > 0:
                ret = ret * norm_bound
            assert (ret.x + self.x).min() >= -1e-7, (ret.x + self.x).min()
            return ret

    def __init__(self, A: npt.NDArray, B: npt.NDArray, P: npt.NDArray):
        assert A.ndim == 2
        assert B.ndim == 3
        assert P.ndim == 2
        m, k, n = B.shape
        assert A.shape == (m, n)
        assert P.shape == (m, n)
        x0 = np.empty(n)
        x0.fill(1 / n)
        self.__attrs_init__(x0=x0, m=m, k=k, n=n, A=A, B=B, P=P)

    @classmethod
    def _get_kl(cls, x: npt.NDArray, P: npt.NDArray, eps: float) -> npt.NDArray:
        kl0 = np.sum(x * np.log(x + eps))
        kl1 = np.sum(x[np.newaxis, :] * np.log(P + eps), axis=1)
        return kl0 - kl1

    def eval(self, x: npt.NDArray, *, need_grad=False):
        assert x.ndim == 1
        assert x.shape[0] == self.n
        x = np.maximum(x, 0)

        Ax = self.A @ x
        Bx = self.B @ x
        Bnorm = np.linalg.norm(Bx, axis=1, ord=2)
        kl = self._get_kl(x, self.P, self.kl_eps)

        comp = np.abs(Ax)
        comp += Bnorm
        comp += kl
        fval = comp.max()
        if need_grad:
            sub_diff = self.SubDiff(
                fval=fval, x=x, A=self.A,
                Ax=Ax, B=self.B, Bx=Bx, Bnorm=Bnorm,
                P=self.P, comp=comp, kl_eps=self.kl_eps)
            return fval, sub_diff
        return fval

    def eval_batch(self, x: npt.NDArray):
        assert x.ndim == 2 and x.shape[1] == self.n
        x = np.maximum(x, 0)

        Ax = x @ self.A.T   # (batch, m)
        Bxl2 = np.linalg.norm(self.B @ x.T, axis=1, ord=2).T     # (batch, m)
        kl_1 = np.sum(x * np.log(x + self.kl_eps), axis=1,
                      keepdims=True)        # (batch, 1)
        kl_2 = np.sum(x[:, np.newaxis, :] *
                      np.log(self.P[np.newaxis] + self.kl_eps),
                      axis=2)               # (batch, m)
        kl = kl_1 - kl_2
        comp = np.abs(Ax)
        comp += Bxl2
        comp += kl
        return comp.max(axis=1)

    def __repr__(self):
        return f'DistanceGame(n={self.n}, m={self.m}, k={self.k})'

    @classmethod
    def gen_random(cls, n: int,
                   m: typing.Optional[int]=None,
                   k: typing.Optional[int]=None,
                   rng: typing.Optional[np.random.Generator] = None
                   ) -> "DistanceGame":
        if rng is None:
            rng = make_stable_rng(cls)

        assert n > 0
        if m is None:
            m = n * 3

        if k is None:
            k = (n + 3) // 4 + 1

        assert type(n) is int and type(m) is int and type(k) is int

        A = rng.normal(size=(m, n))
        B = rng.normal(size=(m, k, n))
        P = rng.exponential(scale=1, size=(m, n))
        P /= P.sum(axis=1, keepdims=True)

        x0 = np.empty(n)
        x0.fill(1 / n)

        stat = lambda x: np.median(np.abs(x))
        # scale A and B so they have comparable magnitudes as KL
        kl = stat(cls._get_kl(x0, P, 1e-8))
        A *= kl / stat(A @ x0)
        B *= kl / stat(np.linalg.norm(B @ x0, axis=1, ord=2))

        A.flags.writeable = False
        B.flags.writeable = False
        P.flags.writeable = False

        return cls(A, B, P)

    @classmethod
    def print_info(cls):
        # see https://www.mscand.dk/article/view/10655/8676
        # compute the probability that B_i contains the origin
        for k in range(2, 100):
            n = (k - 1) * 4 - 2
            assert k == (n + 3) // 4 + 1
            m = sum(math.comb(n - 1, i) for i in range(k))
            p = 1 - m / (2 ** (n - 1))
            print(f'{k=} {n=}: p(B_i contains origin) = {p}')


if __name__ == '__main__':
    DistanceGame.print_info()
