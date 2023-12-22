from ..opt.shared import TRAFSStep
from .simplex import projection_simplex

import abc
import attrs
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import sys

import typing

try:
    # this is a fork of PIQP that supports the TRAFS objective (so we can set a
    # reduced accuracy parameter)
    import piqptr as piqp
except ImportError:
    piqp = None
try:
    import clarabel
except ImportError:
    clarabel = None
try:
    import mosek
except ImportError:
    mosek = None

DenseOrSparse = typing.Union[npt.NDArray, sp.csc_matrix]

def make_stable_rng(cls) -> np.random.Generator:
    """make a stable random number generator for a class"""
    seq = list(map(ord, cls.__name__))
    return np.random.default_rng(seq)


class SOCPSolverBase(metaclass=abc.ABCMeta):
    """base class for SOCP solvers to solve min u s.t. constraints(x, u)"""
    @attrs.frozen
    class Result:
        is_optimal: bool
        """whether the solution is optimal"""

        x: npt.NDArray
        """the optimal solution"""

        pobj: float
        """the optimal primal objective value"""

        dobj: float
        """the optimal dual objective value"""

        def __mul__(self, s: float) -> typing.Self:
            return type(self)(
                is_optimal=self.is_optimal,
                x=self.x * s,
                pobj=self.pobj * s,
                dobj=self.dobj * s
            )

    @abc.abstractmethod
    def add_x_lower(self, x_low: npt.NDArray) -> typing.Self:
        """add the constraint x >= x_low"""

    @abc.abstractmethod
    def add_x_higher(self, x_high: npt.NDArray) -> typing.Self:
        """add the constraint x <= x_high"""

    @abc.abstractmethod
    def add_eq(self, v: npt.NDArray, b: npt.NDArray) -> typing.Self:
        """add the constraint v @ x = b; v should be [m, dim] and b should be [m]
        """

    @abc.abstractmethod
    def add_ineq(self, g: DenseOrSparse) -> typing.Self:
        """add an inequality constraint ``g @ x <= u``, where ``g`` is a
        matrix"""

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
    def make(cls, dim: int) -> "SOCPSolverBase":
        # we do not use cvxpy because its compilation is too slow (each
        # iteration may have a different shape, which requires rebulding the
        # model and recompiling)
        if mosek is not None:
            return MosekSOCPSolver(dim=dim)
        if clarabel is not None:
            return ClarabelSOCPSolver(dim=dim)
        raise RuntimeError('no SOCP solver is available'
                           ' (need mosek or clarabel)')


@attrs.frozen
class ClarabelSOCPSolver(SOCPSolverBase):
    dim: int
    """dimension of the step vector (num of variables is ``dim + 1``)"""

    max_iter: int = 30
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
            I = sp.eye(self.dim, self.dim + 1, dtype=np.float64, format='csc')
            object.__setattr__(self, '_eye_x', I)
        return self._eye_x

    def add_x_lower(self, x_low: npt.NDArray) -> typing.Self:
        # x >= L ==> Ix - L >= 0 ==>  -(-Ix) + (-L) >= 0
        self.As.append(-self.eye_x)
        self.bs.append(-x_low)
        self.cones.append(clarabel.NonnegativeConeT(self.dim))
        return self

    def add_x_higher(self, x_high: npt.NDArray) -> typing.Self:
        # x <= H ==> -(Ix) + H >= 0
        self.As.append(self.eye_x)
        self.bs.append(x_high)
        self.cones.append(clarabel.NonnegativeConeT(self.dim))
        return self

    def add_eq(self, v: npt.NDArray, b: npt.NDArray) -> typing.Self:
        assert v.ndim == 2 and b.ndim == 1 and v.shape[0] == b.size
        tmp = np.zeros((v.shape[0], self.dim + 1), dtype=np.float64)
        tmp[:, :-1] = v
        self.As.append(sp.csc_matrix(tmp))
        self.bs.append(b)
        self.cones.append(clarabel.ZeroConeT(1))
        return self

    def add_ineq(self, g: DenseOrSparse) -> typing.Self:
        # g0 @ x <= u ==> -[g0, -1] @ [x, u] >= 0
        assert g.ndim == 2
        n = g.shape[0]
        assert g.shape[1] == self.dim

        if not isinstance(g, sp.csc_matrix):
            g = sp.csc_matrix(g)

        self.As.append(sp.hstack(
            [g, sp.csc_matrix(-np.ones((n, 1), dtype=np.float64))],
            format='csc'
        ))
        self.bs.append(np.zeros(n, dtype=np.float64))
        self.cones.append(clarabel.NonnegativeConeT(n))
        return self

    def add_socp(self, g: npt.NDArray, h: npt.NDArray) -> typing.Self:
        # gi @ x + ||hi @ x||_2 <= u
        # ==> ||hi @ x||_2 <= u - gi @ x
        # ==> ||-hi @ x||_2 <= -[gi, -1] @ [x, u]
        assert (g.shape == (self.dim,) and h.ndim == 2
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
        n = self.dim + 1
        self.As.append(sp.vstack([
            sp.csc_matrix((1, n), dtype=np.float64),
            -self.eye_x]))
        tmp = np.zeros(n, dtype=np.float64)
        tmp[0] = 1
        self.bs.append(tmp)
        self.cones.append(clarabel.SecondOrderConeT(n))
        return self

    def solve(self) -> SOCPSolverBase.Result:
        A = sp.vstack(self.As).tocsc()
        b = np.concatenate(self.bs)
        n = self.dim + 1
        P = sp.csc_matrix((n, n), dtype=np.float64)
        q = np.zeros(n, dtype=np.float64)
        q[-1] = 1
        setting = clarabel.DefaultSettings()
        setting.max_iter = self.max_iter
        setting.verbose = self.verbose
        solver = clarabel.DefaultSolver(P, q, A, b, self.cones, setting)
        solution = solver.solve()

        x = np.ascontiguousarray(solution.x[:-1])
        u = solution.x[-1]
        np.testing.assert_allclose(u, solution.obj_val)
        S = clarabel.SolverStatus
        assert solution.status in (S.Solved, S.AlmostSolved,
                                   S.MaxIterations, S.MaxTime), (
            f'solver failed with status {solution.status}')

        # CLARABEL does not return the dual objective value; we have to compute
        # it ourselves
        dual_obj = -b @ solution.z
        assert dual_obj <= u

        if solution.status == S.Solved:
            np.testing.assert_allclose(dual_obj, u, atol=1e-6, rtol=1e-6)

        return self.Result(
            is_optimal=solution.status == S.Solved,
            x=x,
            pobj=float(u),
            dobj=float(dual_obj),
        )

@attrs.define
class MosekSOCPSolver(SOCPSolverBase):
    dim: int
    """dimension of the step vector (num of variables is ``dim + 1``)"""

    max_iter: int = 100
    """max number of iterations"""

    verbose: bool = False
    """whether to print verbose output of the solver"""

    _env: "mosek.Env" = attrs.field(init=False, default=None)
    _task: "mosek.Task" = attrs.field(init=False, default=None)

    _xlow: npt.NDArray = attrs.field(init=False, default=None)
    _xhigh: npt.NDArray = attrs.field(init=False, default=None)
    _has_norm_bound: bool = False

    _lin_cons: int = 0
    _afe_cons: int = 0

    def __attrs_post_init__(self):
        self._env = mosek.Env()
        task = self._env.Task(self.dim * 2, self.dim + 1)
        self._task = task
        task.appendvars(self.dim + 1)
        task.putcj(self.dim, 1) # minimize u, the last variable
        task.putobjsense(mosek.objsense.minimize)

        ip = mosek.iparam
        task.putintparam(ip.num_threads, 1)
        task.putintparam(ip.intpnt_max_iterations, self.max_iter)
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

    def add_ineq(self, g: DenseOrSparse) -> typing.Self:
        # g @ x <= u
        assert g.ndim == 2
        assert g.shape[1] == self.dim
        task = self._task
        task.appendcons(g.shape[0])
        r = self._lin_cons
        self._lin_cons += g.shape[0]
        if isinstance(g, np.ndarray):
            idx = np.arange(self.dim, dtype=np.int32)
            g = np.ascontiguousarray(g)
            for i in range(g.shape[0]):
                task.putarow(r + i, idx, g[i])
        else:
            g = g.tocoo()
            task.putaijlist(g.row + r, g.col, g.data)

        for i in range(g.shape[0]):
            task.putaij(r + i, self.dim, -1)
            task.putconbound(r + i, mosek.boundkey.up, -np.inf, 0)
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
            x_low = np.empty(self.dim, dtype=np.float64)
            x_low.fill(-np.inf)
        if x_high is None:
            x_high = np.empty(self.dim, dtype=np.float64)
            x_high.fill(np.inf)
        self._task.putvarboundlist(np.arange(self.dim, dtype=np.int32),
                                   bc, x_low, x_high)
        self._task.putvarbound(self.dim, keyt.fr, -np.inf, np.inf)

    def solve(self) -> SOCPSolverBase.Result:
        self._setup_var_bound()
        task = self._task
        status = task.optimize()
        if self.verbose:
            task.solutionsummary(mosek.streamtype.log)
            # task.writedata("/tmp/prob.ptf"); assert 0
        c = mosek.rescode
        assert status in (c.ok, c.trm_stall, c.trm_stall), (
            f'bad optimizer status: {status}')

        xx = task.getxx(mosek.soltype.itr)
        x = np.ascontiguousarray(xx[:-1], dtype=np.float64)
        u = task.getprimalobj(mosek.soltype.itr)
        np.testing.assert_allclose(xx[-1], task.getprimalobj(mosek.soltype.itr))
        dual_obj = task.getdualobj(mosek.soltype.itr)
        # mosek may return a dual larger than the primal
        assert dual_obj <= u + 1e-6, (dual_obj, u, dual_obj - u)
        dual_obj = min(dual_obj, u)

        if status == c.ok:
            np.testing.assert_allclose(dual_obj, u, atol=1e-6, rtol=1e-6)

        return self.Result(
            is_optimal=status == c.ok,
            x=x,
            pobj=float(u),
            dobj=float(dual_obj),
        )

@attrs.frozen
class UnconstrainedFuncSubDiffHelper:
    """helper class for computing the TRAFS subgradient from the functional
    subdifferential for unconstrained problems. The problem is essentially
    finding the minimum norm subgradient."""

    box_size: float = 10.0
    """max coordinate of the box; although the problem is unconstrained, we
    still assume that max abosulte value of each coordinate is bounded. A
    solution is optimal if it can not be improved within a ball of radius 1.
    """

    f_lb_norm_bound: float = 1
    """L2 norm bound to compute the lower bound of the objective delta"""

    qp_eps: float = 1e-4
    """tolerance for the QP solver"""

    qp_iters: int = 500
    """max number of iterations for the QP solver"""

    def reduce_grad_range(
            self, glow: npt.NDArray, ghigh: npt.NDArray,
            df_lb_thresh: float, norm_bound: float) -> TRAFSStep:
        """Reduce to a subgradient from the subdifferential defined by
        ``{ g | glow <= g <= ghigh }``.

        Note that the inputs ``glow`` and ``ghigh`` are modified in-place.
        """

        # compute gc within [glow, ghigh] that is closest to 0
        glow = np.maximum(glow, 0, out=glow)
        ghigh = np.minimum(ghigh, 0, out=ghigh)
        gc = np.add(glow, ghigh, out=ghigh)
        return self.reduce_with_min_grad(gc, df_lb_thresh, norm_bound)

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
        if gc_norm == 0:
            return TRAFSStep.make_zero(gc.size, True)

        # although the problem is unconstrained, we assume we are optimizing
        # within the unit ball around current solution
        dx = gc * (-min(norm_bound, 1) / np.maximum(gc_norm, 1e-9))
        if dx_dg_fn is None:
            dx_dg = np.dot(dx, gc)
        else:
            dx_dg = dx_dg_fn(dx)
        # min_d max_g d@g = max_g -|g| = -min_g |g| >= -gc_norm even if gc is
        # not the global minimum
        df_l = -gc_norm * self.f_lb_norm_bound
        if dx_dg >= 0:
            return TRAFSStep.make_zero(gc.size, False)
        if norm_bound >= 1:
            df_l = min(-np.abs(gc).sum() * self.box_size, df_l)
            df_is_g = True
        else:
            # consider the df bound within unit ball as a global bound if the
            # norm bound is small enough
            df_is_g = norm_bound <= 1e-3
        return TRAFSStep(dx, dx_dg, df_l, df_is_g)

    def reduce_from_cvx_hull_socp(
            self, G: DenseOrSparse, df_lb_thresh: float, norm_bound: float,
            state: dict) -> TRAFSStep:
        """Reduce to a subgradient given the convex hull. The vertices of the
        convex hull are columns of ``G``. Use an SOCP solver to compute the
        result.
        """
        norm_bound = min(norm_bound, self.box_size)
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
            result = (SOCPSolverBase.make(dim=xdim)
                      .add_ineq(G.T)
                      .add_x_norm_bound()
                      .solve())
            state['cvx_hull_prev_G'] = G
            state['cvx_hull_prev_result'] = result

        df_lb = result.dobj * self.f_lb_norm_bound
        result = result * norm_bound
        dx_dg = (result.x @ G).max()
        np.testing.assert_allclose(dx_dg, result.pobj, atol=1e-6, rtol=1e-6)
        assert dx_dg >= result.dobj - 1e-6

        df_is_g = norm_bound <= 1e-3
        if df_lb >= -1e-9:
            return TRAFSStep.make_zero(xdim, df_is_g)

        df_is_g = df_is_g and result.is_optimal
        if dx_dg >= 0:
            return TRAFSStep.make_zero(xdim, df_is_g)

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
        qp_G = np.zeros((0, dim), dtype=np.float64)
        if is_sparse:
            qp_A = sp.csc_matrix(qp_A)
            qp_G = sp.csc_matrix(qp_G)
        qp_h = np.zeros(0, dtype=np.float64)
        qp_lb = np.zeros(dim, dtype=np.float64)
        qp_ub = np.ones(dim, dtype=np.float64)
        if is_sparse:
            solver = piqp.SparseSolver()
        else:
            solver = piqp.DenseSolver()
        solver.settings.check_trafs_obj = True
        solver.settings.eps_abs = self.qp_eps
        solver.settings.eps_rel = self.qp_eps
        solver.settings.eps_duality_gap_abs = self.qp_eps
        solver.settings.eps_duality_gap_rel = self.qp_eps
        #solver.settings.verbose = True
        solver.settings.max_iter = self.qp_iters
        solver.setup(qp_P, qp_c, qp_A, qp_b, qp_G, qp_h, qp_lb, qp_ub)
        solver.solve()
        return ret_from_gc(G.dot(projection_simplex(solver.result.x)))
