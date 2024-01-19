from ..opt.shared import TRAFSStep
from ..utils import setup_pyx_import
from .simplex import projection_simplex

import attrs
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

import abc
import os
import sys
import typing

import piqptr as piqp
import clarabel
if os.getenv('NSOPT_USE_CLARABEL') == '1':
    mosek = None
else:
    try:
        import mosek
    except ImportError:
        mosek = None

print('Solver versions:\n'
      f' piqptr: {piqp.__version__}\n'
      f' clarabel: {clarabel.__version__}\n'
      f' mosek: {mosek.Env.getversion() if mosek is not None else "N/A"}')

DenseOrSparse = typing.Union[npt.NDArray, sp.csc_matrix]

def make_stable_rng(cls) -> np.random.Generator:
    """make a stable random number generator for a class"""
    seq = list(map(ord, cls.__name__))
    return np.random.default_rng(seq)

def print_once(msg: str, *, _done: set[str] = set()) -> None:
    if msg in _done:
        return
    print(msg)
    _done.add(msg)

class SOCPSolverBase(metaclass=abc.ABCMeta):
    """base class for SOCP solvers to solve min u s.t. constraints(x, u)"""
    @attrs.frozen
    class Result:
        is_optimal: bool
        """whether the solution is optimal"""

        x: npt.NDArray = attrs.field(repr=False)
        """the optimal solution"""

        pobj: float
        """the optimal primal objective value"""

        dobj: float
        """the optimal dual objective value"""

        solver: str
        """name of the solver"""

        def __mul__(self, s: float) -> typing.Self:
            return type(self)(
                is_optimal=self.is_optimal,
                x=self.x * s,
                pobj=self.pobj * s,
                dobj=self.dobj * s,
                solver=self.solver,
            )

    @abc.abstractmethod
    def add_x_lower(self, x_low: npt.NDArray) -> typing.Self:
        """add the constraint x >= x_low"""

    @abc.abstractmethod
    def add_aux_lower(self, aux_low: npt.NDArray) -> typing.Self:
        """add the constraint u >= aux_low"""

    @abc.abstractmethod
    def add_x_higher(self, x_high: npt.NDArray) -> typing.Self:
        """add the constraint x <= x_high"""

    @abc.abstractmethod
    def add_eq(self, v: npt.NDArray, b: npt.NDArray) -> typing.Self:
        """add the constraint v @ x = b; v should be [m, dim] and b should be [m]
        """

    @abc.abstractmethod
    def add_ineq(self, g: DenseOrSparse,
                 ui: typing.Optional[npt.NDArray] = None) -> typing.Self:
        """add an inequality constraint ``g @ x <= u[ui]``, where ``g`` is a
        matrix

        :param ui: the index into the auxiliary variable ``u``; if not
            specified, it will be the first one. Its shape should be
            ``[g.shape[0]]``.
        """

    @abc.abstractmethod
    def add_socp(self, g: npt.NDArray, h: npt.NDArray) -> typing.Self:
        """add a second-order cone constraint ``g @ x + ||h @ x||_2 <= u``"""

    @abc.abstractmethod
    def add_x_norm_bound(self) -> typing.Self:
        """add the constraint ||x||_2 <= 1"""

    @abc.abstractmethod
    def solve(self) -> Result:
        """solve the problem"""

    @classmethod
    def make(cls, dim: int, dim_aux: int = 1,
             coeff_vec: typing.Optional[npt.NDArray] = None,
             force_clarabel=False) -> "SOCPSolverBase":
        # we do not use cvxpy because its compilation is too slow (each
        # iteration may have a different shape, which requires rebulding the
        # model and recompiling)
        if mosek is not None and not force_clarabel:
            print_once('Use MOSEK as the SOCP solver')
            return MosekSOCPSolver(dim=dim, dim_aux=dim_aux,
                                   coeff_vec=coeff_vec)
        if clarabel is not None:
            print_once('Use Clarabel as the SOCP solver')
            return ClarabelSOCPSolver(dim=dim, dim_aux=dim_aux,
                                      coeff_vec=coeff_vec)
        raise RuntimeError('no SOCP solver is available'
                           ' (need mosek or clarabel)')


@attrs.frozen
class ClarabelSOCPSolver(SOCPSolverBase):
    dim: int
    """dimension of the step vector (num of variables is ``dim + dim_aux``)"""

    dim_aux: int = 1
    """number of auxiliary variables"""

    coeff_vec: typing.Optional[npt.NDArray] = None
    """coefficients of the cost function; if not specified, there must be a
    single auxiliary variable and it is the cost"""

    max_iter: int = 50
    """max number of iterations"""

    verbose: bool = False
    """whether to print verbose output of the solver"""

    As: list[sp.csc_matrix] = attrs.field(factory=list)
    """the sparse matrices in the constraints"""

    bs: list[npt.NDArray] = attrs.field(factory=list)
    """the offset vectors in the constraints"""

    cones: list = attrs.field(factory=list)
    """the cones in the constraints"""

    _eye_x: sp.csc_matrix = attrs.field(init=False, default=None)

    @property
    def eye_x(self) -> sp.csc_matrix:
        """make a sparse identity matrix for x"""
        if self._eye_x is None:
            I = sp.eye(self.dim, self.dim + self.dim_aux, dtype=np.float64,
                       format='csc')
            object.__setattr__(self, '_eye_x', I)
        return self._eye_x

    def add_x_lower(self, x_low: npt.NDArray) -> typing.Self:
        # x >= L ==> Ix - L >= 0 ==>  -(-Ix) + (-L) >= 0
        self.As.append(-self.eye_x)
        self.bs.append(-x_low)
        self.cones.append(clarabel.NonnegativeConeT(self.dim))
        return self

    def add_aux_lower(self, aux_low: npt.NDArray) -> typing.Self:
        rows = np.arange(self.dim_aux, dtype=np.int32)
        cols = rows + self.dim
        data = np.full(self.dim_aux, -1, dtype=np.float64)
        self.As.append(sp.csc_matrix(
            (data, (rows, cols)), shape=(self.dim_aux, self.dim + self.dim_aux),
            dtype=np.float64))
        self.bs.append(-aux_low)
        self.cones.append(clarabel.NonnegativeConeT(self.dim_aux))
        return self

    def add_x_higher(self, x_high: npt.NDArray) -> typing.Self:
        # x <= H ==> -(Ix) + H >= 0
        self.As.append(self.eye_x)
        self.bs.append(x_high)
        self.cones.append(clarabel.NonnegativeConeT(self.dim))
        return self

    def add_eq(self, v: npt.NDArray, b: npt.NDArray) -> typing.Self:
        assert v.ndim == 2 and b.ndim == 1 and v.shape[0] == b.size
        tmp = np.zeros((v.shape[0], self.dim + self.dim_aux), dtype=np.float64)
        tmp[:, :-self.dim_aux] = v
        self.As.append(sp.csc_matrix(tmp))
        self.bs.append(b)
        self.cones.append(clarabel.ZeroConeT(1))
        return self

    def add_ineq(self, g: DenseOrSparse, ui=None) -> typing.Self:
        # g0 @ x <= u[ui] ==> -[g0, -1] @ [x, u[ui]] >= 0
        assert g.ndim == 2
        n = g.shape[0]
        assert g.shape[1] == self.dim

        if not isinstance(g, sp.csc_matrix):
            g = sp.csc_matrix(g)

        if ui is None:
            ui = np.zeros(n, dtype=np.int32)
        else:
            assert ui.shape == (n, ) and ui.dtype == np.int32
        u_rows = np.arange(n, dtype=np.int32)
        u_data = np.full(n, -1, dtype=np.float64)
        u_mat = sp.csc_matrix((u_data, (u_rows, ui)),
                              dtype=np.float64, shape=(n, self.dim_aux))
        self.As.append(sp.hstack([g, u_mat], format='csc'))
        self.bs.append(np.zeros(n, dtype=np.float64))
        self.cones.append(clarabel.NonnegativeConeT(n))
        return self

    def add_socp(self, g: npt.NDArray, h: npt.NDArray) -> typing.Self:
        # gi @ x + ||hi @ x||_2 <= u
        # ==> ||hi @ x||_2 <= u - gi @ x
        # ==> ||-hi @ x||_2 <= -[gi, -1] @ [x, u]
        assert (g.shape == (self.dim,) and h.ndim == 2 and self.dim_aux == 1
                and h.shape[1] == self.dim)
        tmp = np.zeros((h.shape[0] + 1, self.dim + 1), dtype=np.float64)
        tmp[0, :-1] = g
        tmp[0, -1] = -1
        tmp[1:, :-1] = h
        self.As.append(sp.csc_matrix(tmp))
        self.bs.append(np.zeros(tmp.shape[0], dtype=np.float64))
        self.cones.append(clarabel.SecondOrderConeT(tmp.shape[0]))
        return self

    def add_x_norm_bound(self) -> typing.Self:
        """add the constraint ||x||_2 <= 1"""
        # s[0] = 1
        # s[1:] = x
        # s = -([0; -I]x) + [1; 0]
        self.As.append(sp.vstack([
            sp.csc_matrix((1, self.dim + self.dim_aux), dtype=np.float64),
            -self.eye_x]))
        tmp = np.zeros(self.dim + 1, dtype=np.float64)
        tmp[0] = 1
        self.bs.append(tmp)
        self.cones.append(clarabel.SecondOrderConeT(self.dim + 1))
        return self

    def solve(self) -> SOCPSolverBase.Result:
        assert self.dim > 0 and self.dim_aux >= 0
        A = sp.vstack(self.As).tocsc()
        b = np.concatenate(self.bs)
        n = self.dim + self.dim_aux
        P = sp.csc_matrix((n, n), dtype=np.float64)
        if self.coeff_vec is None:
            assert self.dim_aux == 1
            q = np.zeros(n, dtype=np.float64)
            q[-1] = 1
        else:
            q = self.coeff_vec
            assert q.shape == (n, )
        setting = clarabel.DefaultSettings()
        setting.max_iter = self.max_iter
        setting.verbose = self.verbose
        solver = clarabel.DefaultSolver(P, q, A, b, self.cones, setting)
        solution = solver.solve()

        x = np.ascontiguousarray(solution.x[:self.dim])
        assert x.shape == (self.dim, ), (
            x.shape, len(solution.x), self.dim, self.dim_aux)
        if self.coeff_vec is None:
            u = solution.x[-1]
            np.testing.assert_allclose(u, solution.obj_val)
        else:
            u = solution.obj_val
        S = clarabel.SolverStatus
        assert solution.status in (
            S.Solved, S.AlmostSolved, S.MaxIterations, S.MaxTime,
            S.NumericalError, S.InsufficientProgress), (
            f'solver failed with status {solution.status}')

        # CLARABEL does not return the dual objective value; we have to compute
        # it ourselves
        dual_obj = -b @ solution.z
        assert dual_obj <= u + 1e-7, (dual_obj, u, dual_obj - u)
        dual_obj = min(dual_obj, u) # numerical error may cause dual_obj > u

        if solution.status == S.Solved:
            np.testing.assert_allclose(dual_obj, u, atol=1e-6, rtol=1e-6)

        return self.Result(
            is_optimal=solution.status == S.Solved,
            x=x,
            pobj=float(u),
            dobj=float(dual_obj),
            solver='Clarabel',
        )

@attrs.define
class MosekSOCPSolver(SOCPSolverBase):
    dim: int
    """dimension of the step vector (num of variables is ``dim + dim_aux``)"""

    dim_aux: int = 1
    """number of auxiliary variables"""

    coeff_vec: typing.Optional[npt.NDArray] = None
    """coefficients of the cost function; if not specified, there must be a
    single auxiliary variable and it is the cost"""

    verbose: bool = False
    """whether to print verbose output of the solver"""

    _env: "mosek.Env" = attrs.field(init=False, default=None)
    _task: "mosek.Task" = attrs.field(init=False, default=None)

    _xlow: npt.NDArray = attrs.field(init=False, default=None)
    _xhigh: npt.NDArray = attrs.field(init=False, default=None)
    _auxlow: npt.NDArray = attrs.field(init=False, default=None)
    _has_norm_bound: bool = False

    _lin_cons: int = 0
    _afe_cons: int = 0

    def __attrs_post_init__(self):
        self._env = mosek.Env()
        task = self._env.Task(self.dim * 2, self.dim + 1)
        self._task = task
        task.appendvars(self.dim + self.dim_aux)
        if self.coeff_vec is None:
            task.putcj(self.dim, 1) # minimize u, the last variable
        else:
            assert self.coeff_vec.shape == (self.dim + self.dim_aux, )
            task.putclist(np.arange(self.dim + self.dim_aux, dtype=np.int32),
                          self.coeff_vec)

        task.putobjsense(mosek.objsense.minimize)

        ip = mosek.iparam
        task.putintparam(ip.num_threads, 1)

        if self.verbose:
            for i in range(self.dim):
                task.putvarname(i, f'x{i}')
            task.putvarname(self.dim, 'u')
            def streamprinter(text):
                sys.stdout.write(text)
                sys.stdout.flush()
            task.set_Stream(mosek.streamtype.log, streamprinter)
        else:
            task.putintparam(ip.log, 0)

    def add_x_lower(self, x_low: npt.NDArray) -> typing.Self:
        assert self._xlow is None
        self._xlow = x_low
        return self

    def add_x_higher(self, x_high: npt.NDArray) -> typing.Self:
        assert self._xhigh is None
        self._xhigh = x_high
        return self

    def add_aux_lower(self, aux_low: npt.NDArray) -> typing.Self:
        assert self._auxlow is None
        self._auxlow = aux_low
        return self

    def add_eq(self, v: npt.NDArray, b: npt.NDArray) -> typing.Self:
        # v @ x == b
        assert v.ndim == 2 and b.ndim == 1 and v.shape[0] == b.size
        assert v.shape[1] == self.dim
        task = self._task
        task.appendcons(v.shape[0])
        r = self._lin_cons
        self._lin_cons += v.shape[0]
        idx = np.arange(self.dim, dtype=np.int32)
        for i in range(v.shape[0]):
            task.putarow(r + i, idx, v[i])
            task.putconbound(r + i, mosek.boundkey.fx, b[i], b[i])
        return self

    def add_ineq(self, g: DenseOrSparse, ui=None) -> typing.Self:
        # g @ x <= u
        assert g.ndim == 2
        assert g.shape[1] == self.dim
        nr_cons = g.shape[0]
        if nr_cons == 0:
            return self

        task = self._task
        task.appendcons(nr_cons)
        r = self._lin_cons
        self._lin_cons += nr_cons
        if isinstance(g, np.ndarray):
            idx = np.arange(self.dim, dtype=np.int32)
            g = np.ascontiguousarray(g)
            for i in range(nr_cons):
                task.putarow(r + i, idx, g[i])
        else:
            g = g.tocoo()
            task.putaijlist(g.row + r, g.col, g.data)

        row_idx = np.arange(r, r + nr_cons, dtype=np.int32)

        task.putconboundlist(row_idx, [mosek.boundkey.up] * nr_cons,
                             np.full(nr_cons, -np.inf, dtype=np.float64),
                             np.zeros(nr_cons, dtype=np.float64))
        if ui is None:
            ui = np.full(nr_cons, self.dim, dtype=np.int32)
        else:
            assert (ui.shape == (nr_cons, ) and
                    ui.min() >= 0 and
                    ui.max() < self.dim_aux and
                    ui.dtype == np.int32)
            ui = ui + self.dim

        task.putaijlist(row_idx, ui, np.full(nr_cons, -1, dtype=np.float64))
        return self

    def add_socp(self, g: npt.NDArray, h: npt.NDArray) -> typing.Self:
        # g @ x + ||h @ x||_2 <= u
        assert g.ndim == 1 and g.shape[0] == self.dim
        assert h.ndim == 2 and h.shape[1] == self.dim
        task = self._task
        n = h.shape[0] + 1
        task.appendafes(n)
        r = self._afe_cons
        self._afe_cons += n

        idx = np.arange(self.dim, dtype=np.int32)
        task.putafefrow(r, idx, -g)
        task.putafefentry(r, self.dim, 1)

        for i in range(h.shape[0]):
            task.putafefrow(r + i + 1, idx, h[i])

        dom = task.appendquadraticconedomain(n)
        task.appendacc(dom, np.arange(r, r + n, dtype=np.int32), None)
        return self

    def add_x_norm_bound(self) -> typing.Self:
        # ||x||_2 <= 1
        if self._has_norm_bound:
            return
        self._has_norm_bound = True
        task = self._task
        n = self.dim + 1
        task.appendafes(n)
        r = self._afe_cons
        self._afe_cons += n

        idx = np.arange(self.dim, dtype=np.int32)
        task.putafefentrylist(r + idx + 1, idx,
                              np.ones(self.dim, dtype=np.float64))
        task.putafeg(r, 1)
        dom = task.appendquadraticconedomain(n)
        task.appendacc(dom, np.arange(r, r + n, dtype=np.int32), None)
        return self

    def _setup_var_bound(self):
        x_low = self._xlow
        x_high = self._xhigh
        if x_low is None and self._has_norm_bound:
            x_low = -np.ones(self.dim, dtype=np.float64)
        if x_high is None and self._has_norm_bound:
            x_high = np.ones(self.dim, dtype=np.float64)

        keyt = mosek.boundkey
        bc_map = {
            (True, True): keyt.fr,
            (True, False): keyt.up,
            (False, True): keyt.lo,
            (False, False): keyt.ra
        }
        bc = [bc_map[(x_low is None, x_high is None)]] * self.dim
        if x_low is None:
            x_low = np.full(self.dim, -np.inf, dtype=np.float64)
        if x_high is None:
            x_high = np.full(self.dim, np.inf, dtype=np.float64)

        task = self._task
        task.putvarboundlist(np.arange(self.dim, dtype=np.int32),
                                   bc, x_low, x_high)
        if self._auxlow is None:
            task.putvarbound(self.dim, keyt.fr, -np.inf, np.inf)
        else:
            assert self._auxlow.shape == (self.dim_aux, )
            task.putvarboundlist(
                np.arange(self.dim, self.dim + self.dim_aux, dtype=np.int32),
                [keyt.lo] * self.dim_aux,
                self._auxlow,
                np.full(self.dim_aux, np.inf, dtype=np.float64))

    def solve(self) -> SOCPSolverBase.Result:
        self._setup_var_bound()
        task = self._task
        status = task.optimize()
        if self.verbose:
            task.solutionsummary(mosek.streamtype.log)
            # task.writedata("/tmp/prob.ptf"); assert 0
        c = mosek.rescode
        assert status in (c.ok, c.trm_stall, c.trm_max_iterations), (
            f'bad optimizer status: {status}')

        xx = task.getxx(mosek.soltype.itr)
        x = np.ascontiguousarray(xx[:self.dim], dtype=np.float64)
        u = task.getprimalobj(mosek.soltype.itr)
        if self.coeff_vec is None:
            np.testing.assert_allclose(xx[-1], u)
        dual_obj = task.getdualobj(mosek.soltype.itr)
        assert dual_obj <= u + 5e-6, (dual_obj, u, dual_obj - u, status)
        dual_obj = min(dual_obj, u)

        if status == c.ok:
            np.testing.assert_allclose(dual_obj, u, atol=1e-6, rtol=1e-6)

        return self.Result(
            is_optimal=status == c.ok,
            x=x,
            pobj=float(u),
            dobj=float(dual_obj),
            solver='MOSEK',
        )

@attrs.frozen
class SumOfCvxHullDesc:
    """description of a sum of convex hulls:

        { bias + G @ x } where x is the same size as ui, x[i] and x[j] are
        convex combination coefficients of the same convex hull when ui[i] ==
        ui[j]. All convex hulls include the origin (so the caller should shift
        one of the vertices to the origin by including it in bias).
    """

    bias: npt.NDArray
    G: DenseOrSparse
    ui: npt.NDArray
    """the index of the convex hull for each column of ``G``; must be
    contiguously numbered from 0"""

    nr_hull: int

    def __attrs_post_init__(self):
        n, p = self.G.shape
        assert self.bias.shape == (n, )
        assert self.ui.shape == (p, ) and self.ui.dtype == np.int32
        assert 0 <= self.nr_hull <= p


@attrs.frozen
class UnconstrainedFuncSubDiffHelper:
    """helper class for computing the TRAFS subgradient from the functional
    subdifferential for unconstrained problems. The problem is essentially
    finding the minimum norm subgradient."""

    dx_l2_max: float = 1e5
    """maximum L2 norm of the solved step"""

    df_g_norm_bound_thresh: float = 1e-4
    """maximum norm bound for the current lower bound of df to be considered as
    global"""

    f_lb_norm_bound_mul: float = 1
    """L2 norm bound multiplier (multiplied with sqrt(n)) to compute the lower
    bound of the objective delta"""

    qp_eps: float = 1e-3
    """tolerance for the QP solver"""

    qp_iters: int = 200
    """max number of iterations for the QP solver"""

    qp_min_pobj: float = 1e-13
    """abort the QP solver if the primal objective (i.e., min grad norm) is
    below this value"""

    cvx_hull_prefer_qp: bool = os.getenv('NSOPT_CVX_HULL_PREFER_QP') == '1'
    cvx_hull_prefer_socp: bool = os.getenv('NSOPT_CVX_HULL_PREFER_SOCP') == '1'

    def reduce_with_min_grad(
            self, gc: npt.NDArray,
            df_lb_thresh: float, norm_bound: float, *,
            dx_dg_fn: typing.Optional[
                typing.Callable[[npt.NDArray], float]]=None
        ) -> TRAFSStep:
        """Reduce to a subgradient when the min-length subgradient has been
        solved.

        :param dx_dg_fn: a function that computes dx_dg given dx
        """
        gc_norm = np.linalg.norm(gc, ord=2)

        # min_d max_g d@g = max_g -|g| = -min_g |g| >= -gc_norm even if gc is
        # not the global minimum, so we have a sound estimation of df_l
        df_l = -gc_norm * self.f_lb_norm_bound_mul * np.sqrt(gc.size)
        df_is_g = norm_bound <= self.df_g_norm_bound_thresh

        # although the problem is unconstrained, we assume we are optimizing
        # within the unit ball around current solution
        dx = gc * (-min(norm_bound, self.dx_l2_max) / np.maximum(gc_norm, 1e-9))
        if dx_dg_fn is None:
            dx_dg = float(np.dot(dx, gc))
        else:
            dx_dg = float(dx_dg_fn(dx))

        return TRAFSStep(dx, dx_dg, df_l, df_is_g)

    def reduce_from_cvx_hull_socp(
            self, G: DenseOrSparse, df_lb_thresh: float, norm_bound: float,
            state: dict,
            force_clarabel=False) -> TRAFSStep:
        """Reduce to a subgradient given the convex hull. The vertices of the
        convex hull are columns of ``G``. Use an SOCP solver to compute the
        result.
        """
        xdim = G.shape[0]
        prev_G = state.get('cvx_hull_prev_G')
        def all_eq(a, b):
            if isinstance(G, np.ndarray):
                return np.all(a == b)
            else:
                return (a != b).count_nonzero() == 0

        if prev_G is not None and prev_G.shape == G.shape and all_eq(prev_G, G):
            result = state['cvx_hull_prev_result']
        else:
            result = (SOCPSolverBase.make(dim=xdim,
                                          force_clarabel=force_clarabel)
                      .add_ineq(G.T)
                      .add_x_norm_bound()
                      .solve())
            state['cvx_hull_prev_G'] = G
            state['cvx_hull_prev_result'] = result

        return self.reduce_with_socp_result(
            result, df_lb_thresh, norm_bound,
            dx_dg_fn=lambda dx: (dx @ G).max(),
            state=state,
        )

    def reduce_with_socp_result(
            self, result: SOCPSolverBase.Result,
            df_lb_thresh: float, norm_bound: float,
            dx_dg_fn: typing.Callable[[npt.NDArray], float],
            state: dict) -> TRAFSStep:
        """get the TRAFS step from the result of an SOCP solver that solves

                min_{x in B[1]} max_{g in G} d @ g
        """
        norm_bound = min(norm_bound, self.dx_l2_max)
        xdim = result.x.size
        pobj_recompute = dx_dg_fn(result.x)

        tol = state.setdefault('socp_pboj_check_tol', 1e-6)
        if not np.allclose(pobj_recompute, result.pobj, atol=tol, rtol=tol):
            print('Warning: SOCP objective does not match recomputed bound:'
                  f' solver={result.solver} optimal={result.is_optimal}'
                  f' obj={result.pobj:g} expect={pobj_recompute:g}'
                  f' diff={abs(result.pobj - pobj_recompute):g} {tol=:.2g}')
            assert (not result.is_optimal or tol < 1e-5) and (tol < 0.01), (
                pobj_recompute, result.pobj, tol, result)
            state['socp_pboj_check_tol'] = tol * 2

        # dual obj should be no larger than primal obj
        assert result.dobj - pobj_recompute <= 1e-7*max(1, abs(result.dobj)), (
            pobj_recompute, result.dobj, result.dobj - pobj_recompute)

        df_lb = (min(result.dobj, pobj_recompute) *
                 self.f_lb_norm_bound_mul * np.sqrt(xdim))

        result = result * norm_bound
        dx_dg = dx_dg_fn(result.x)
        df_is_g = norm_bound <= self.df_g_norm_bound_thresh

        return TRAFSStep(
            dx=result.x,
            dx_dg=dx_dg,
            df_lb=df_lb,
            df_lb_is_global=df_is_g)

    def reduce_from_cvx_hull_qp(
            self, G: DenseOrSparse,
            df_lb_thresh: float, norm_bound: float, state: dict) -> TRAFSStep:
        """similar to ``reduce_from_cvx_hull_socp`` but use a QP solver
        instead."""
        assert piqp is not None, 'PIQP is required'
        if isinstance(G, sp.csc_matrix):
            is_sparse = True
        else:
            is_sparse = False
            assert isinstance(G, np.ndarray)

        def ret_from_gc(gc: npt.NDArray) -> TRAFSStep:
            state['cvx_hull_qp_prev_G'] = G
            state['cvx_hull_qp_prev_gc'] = gc
            return self.reduce_with_min_grad(
                gc, df_lb_thresh, norm_bound,
                dx_dg_fn=lambda dx: (dx @ G).max()
            )

        if is_sparse:
            all_eq = lambda a, b: (a != b).count_nonzero() == 0
            GtG = (G.T @ G).tocsc()
        else:
            all_eq = lambda a, b: np.all(a == b)
            GtG = G.T @ G

        prev_G = state.get('cvx_hull_qp_prev_G')
        if prev_G is not None and prev_G.shape == G.shape and all_eq(prev_G, G):
            return ret_from_gc(state['cvx_hull_qp_prev_gc'])

        dim = G.shape[1]
        qp_P = GtG
        qp_c = np.zeros(dim, dtype=np.float64)
        qp_A = np.ones((1, dim), dtype=np.float64)
        qp_b = np.ones(1, dtype=np.float64)
        qp_G = -GtG # GtG @ x > 0 ensures descent progress
        qp_h = np.zeros(dim, dtype=np.float64)
        if is_sparse:
            qp_A = sp.csc_matrix(qp_A)
        qp_lb = np.zeros(dim, dtype=np.float64)
        qp_ub = np.ones(dim, dtype=np.float64)
        if is_sparse:
            solver = piqp.SparseSolver()
        else:
            solver = piqp.DenseSolver()
        # solver.settings.verbose = True

        # disable preconditioner since it seems to cause problems in calculating
        # primal_inf (i.e., primal_inf is small, but qp_G @ x <= qp_h is
        # violated)
        solver.settings.preconditioner_iter = 0
        def term_cb(result):
            i = result.info
            if i.primal_inf > 1e-8:
                return False
            return (i.primal_obj < self.qp_min_pobj or
                    GtG.dot(result.x).min() > 0)
        solver.settings.custom_term_cb = term_cb
        solver.settings.eps_abs = self.qp_eps
        solver.settings.eps_rel = self.qp_eps
        solver.settings.eps_duality_gap_abs = self.qp_eps
        solver.settings.eps_duality_gap_rel = self.qp_eps
        solver.settings.max_iter = self.qp_iters
        solver.setup(qp_P, qp_c, qp_A, qp_b, qp_G, qp_h, qp_lb, qp_ub)
        solver.solve()
        return ret_from_gc(G.dot(projection_simplex(solver.result.x)))

    def reduce_from_cvx_hull_qp_direct(
            self, G: DenseOrSparse,
            df_lb_thresh: float, norm_bound: float, state: dict) -> TRAFSStep:
        """use a QP formulation to solve dx directly"""
        xdim = G.shape[0]
        qp_P = sp.eye(xdim, dtype=np.float64, format='csc')
        qp_c = np.zeros(xdim, dtype=np.float64)
        qp_A = sp.csc_matrix((0, xdim), dtype=np.float64)
        qp_b = np.zeros(0, dtype=np.float64)
        qp_G = sp.csc_matrix(G.T)
        qp_h = np.full(G.shape[1], -1, dtype=np.float64)
        qp_lb = np.full(xdim, -1e10, dtype=np.float64)
        qp_ub = np.full(xdim, 1e10, dtype=np.float64)
        solver = piqp.SparseSolver()
        # solver.settings.verbose = True
        solver.settings.preconditioner_scale_cost = True
        solver.setup(qp_P, qp_c, qp_A, qp_b, qp_G, qp_h, qp_lb, qp_ub)
        status = solver.solve()
        if status in (piqp.PIQP_PRIMAL_INFEASIBLE,
                      piqp.PIQP_DUAL_INFEASIBLE):
            return TRAFSStep.make_zero(xdim, True)

        info = solver.result.info
        assert info.primal_obj >= info.dual_obj * (1 - 1e-6) > 0, (
            info.primal_obj, info.dual_obj, info.primal_obj - info.dual_obj)
        dx = solver.result.x.copy()
        dx *= min(norm_bound, self.dx_l2_max) / np.linalg.norm(dx, ord=2)
        dx_dg = (dx @ G).max()

        df_lb = (-1 / np.sqrt(2 * info.dual_obj) * self.f_lb_norm_bound_mul *
                 np.sqrt(xdim))
        df_is_g = norm_bound <= self.df_g_norm_bound_thresh

        return TRAFSStep(
            dx=dx,
            dx_dg=dx_dg,
            df_lb=df_lb,
            df_lb_is_global=df_is_g)

    def reduce_from_multi_cvx_hull_socp(
            self, desc: SumOfCvxHullDesc,
            df_lb_thresh: float, norm_bound: float,
            state: dict,
            force_clarabel=False) -> TRAFSStep:
        """Reduce to a subgradient given multiple convex hulls defined by
        ``desc``. Use an SOCP solver to compute the result.
        """
        prev_desc = state.get('multi_cvx_hull_prev_desc')
        def all_eq(a, b):
            if isinstance(a, int):
                return a == b
            if a.shape != b.shape:
                return False
            if isinstance(a, np.ndarray):
                return np.all(a == b)
            else:
                return (a != b).count_nonzero() == 0

        xdim = desc.G.shape[0]
        pdim = desc.nr_hull
        if prev_desc is not None and all(
            all_eq(i, j) for i, j in zip(*map(attrs.astuple, (prev_desc, desc)))
        ):
            result = state['multi_cvx_hull_prev_result']
        else:
            cost_v = np.empty(xdim + pdim, dtype=np.float64)
            cost_v[:xdim] = desc.bias
            cost_v[xdim:] = 1
            result = (SOCPSolverBase.make(
                        dim=xdim, dim_aux=pdim, coeff_vec=cost_v,
                        force_clarabel=force_clarabel)
                      .add_ineq(desc.G.T, desc.ui)
                      .add_aux_lower(np.zeros(pdim, dtype=np.float64))
                      .add_x_norm_bound()
                      .solve())
            state['multi_cvx_hull_prev_desc'] = desc
            state['multi_cvx_hull_prev_result'] = result

        def dx_dg_fn(dx):
            with setup_pyx_import():
                from .kernels import reduce_multi_cvx_hull_max_sum
            return ((desc.bias @ dx) +
                    reduce_multi_cvx_hull_max_sum(
                        desc.nr_hull,  dx @ desc.G, desc.ui))

        return self.reduce_with_socp_result(
            result, df_lb_thresh, norm_bound,
            dx_dg_fn=dx_dg_fn,
            state=state,
        )
