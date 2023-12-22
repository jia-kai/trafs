from .shared import Optimizable, OptimizationResult, TRAFSStep
from ..utils import setup_pyx_import
with setup_pyx_import():
    from .trafs_utils import RotationBuffer

import attrs
import numpy as np

import time
import typing
import enum
import copy

@attrs.define(kw_only=True)
class TRAFSSolver:
    """the Trust Region Adversarial Functional Subdifferential method"""

    PROBLEM_CLASS = Optimizable

    eps_term: float = 1e-4
    """eps to check termination (i.e., function value is guranteed to be within
        this range of optimum)"""

    ls_min_step: float = 1e-9
    """minimum step length in line search before giving up"""

    ls_tau: float = .8
    """line search step decay"""

    ls_rho: float = .5
    """line search sufficient decrease constant"""

    ls_batch_size: int = 8
    """number of points to be concurrently evaluated in line search"""""

    norm_hist_winsize: int = 8
    """window size for the step norm history"""

    norm_hist_mul: float = 1 / ls_tau**2
    """multiplier to be applied to max norm in history to get current norm
    bound"""

    norm_incr_mul: float = 2
    """multiplier to be applied to norm bound when -dx @ dg is too small"""

    subg_slack_decay: float = .5
    """eps decay factor when -dx @ dg is not big enough"""

    subg_slack_incr: float = 1.5
    """eps increase factor when -dx @ dg is big enough"""

    subg_slack_max: float = 1.0
    """max eps in functional subdifferential"""

    subg_slack_tune_prob: float = .2
    """probability to automatically tune eps by computing another step using the
    same xk"""

    subg_slack_tune_thresh: float = 1.2
    """threshold of relative change of objective value decrease to accept the
    new eps in automatically eps tuning"""

    subg_slack_est_winsize: int = 8
    """window size for the slack estimation"""

    subg_slack_est_mul_min: float = .5
    """min multiplier to be applied to do slack estimation"""

    subg_slack_est_mul_max: float = 1e5
    """max multiplier to be applied to do slack estimation"""

    max_iters: typing.Optional[int] = None
    """max number of iterations"""

    verbose: bool = False
    """whether to print per-iteration information"""

    verbose_iters: int = 1000
    """number of iterations between verbose prints"""

    max_fail_iters: int = 10
    """max number of failed iterations before giving up"""

    rng: np.random.Generator = attrs.field(
        factory=lambda: np.random.default_rng(42)
    )

    def solve(self, obj: Optimizable) -> OptimizationResult:
        """solve with adatptive steps"""
        xk = obj.x0.copy()
        assert xk.ndim == 1
        runtime = TRAFSRuntime(self, obj)
        while True:
            xk, status = runtime.next_iter(xk)
            if status != IterStatus.succeeded:
                break

        xk = obj.proj(xk)
        return OptimizationResult(
            optimal=status == IterStatus.optimal,
            x=xk, fval=obj.eval(xk), fval_hist=np.array(runtime.fval_hist),
            iter_times=np.array(runtime.iter_times),
            iters=runtime.iters, ls_tot_iters=runtime.ls_tot_iters,
            time=time.time() - runtime.time_start
        )

    def logmsg(self, msg):
        """log a message"""
        if self.verbose:
            print(msg, flush=True)


@attrs.frozen
class LineSearchResult:
    """result of batched line search;
    ``dx_new``, ``xnew``, ``fvals_new`` are lists where each item corresponds to
    a batch"""

    last_step: float
    """the last step length"""

    dx_new: list[np.ndarray]
    xnew: list[np.ndarray]
    fvals_new: list[np.ndarray]


class IterStatus(enum.Enum):
    succeeded = 'succeeded'
    """current iteration succeeded, move on to next iteration"""

    optimal = 'optimal'
    """optimal solution found"""

    failed = 'failed'
    """failed to find a solution; do not run any more iterations"""


@attrs.define
class TRAFSRuntime:
    """stateful iterations of the TRAFS method; used internally by :class:
    `TRAFSSolver`"""

    obj: Optimizable
    solver: "TRAFSSolver"

    subg_slack: float
    """current eps in functional subdifferential"""

    min_subg_slack: float

    ls_steps_init: np.ndarray
    ls_steps_grow: np.ndarray

    norm_hist: RotationBuffer
    subg_slack_est_hist: RotationBuffer

    rng: np.random.Generator

    subg_slack_est_mul: float = 1.0
    last_subg_slack: float = 1.0
    """last successful eps in functional subdifferential"""

    subg_slack_tune_dir: int = 0
    """direction of eps tuning: -1 for decrease, 1 for increase, 0 for random"""

    fval_lb: float = -np.inf
    time_start: float = attrs.field(factory=time.time)
    ls_tot_iters: int = 0
    iters: int = 0
    iter_times: list[float] = attrs.field(factory=list)
    fval_hist: list[float] = attrs.field(factory=list)
    failed_iters: int = 0
    obj_grad_state: dict = attrs.field(factory=dict)

    def __init__(self, solver: "TRAFSSolver", obj: Optimizable):
        self.__attrs_init__(
            obj=obj,
            solver=solver,
            subg_slack=solver.eps_term / 2,
            min_subg_slack=min(max(solver.eps_term / 10, 1e-8),
                               solver.eps_term / 2),
            ls_steps_init=np.power(solver.ls_tau,
                                   np.arange(solver.ls_batch_size)),
            ls_steps_grow=np.power(solver.ls_tau, solver.ls_batch_size),
            norm_hist=RotationBuffer(solver.norm_hist_winsize),
            subg_slack_est_hist=RotationBuffer(solver.subg_slack_est_winsize),
            rng=copy.deepcopy(solver.rng)
        )

    def _randomize_subg_slack(self, reason: str):
        """try a random functional subdifferential slack after failure to find a
        descent direction"""
        k = np.log(100)
        self.subg_slack = max(
            np.exp(self.rng.uniform(-k, k)) * self.last_subg_slack,
            self.min_subg_slack)
        self.logmsg(
            f'failed to find descent direction due to {reason};'
            f' try eps {self.subg_slack:.3g}')

    def _batched_linesearch(self, grad: TRAFSStep, fval, xk) -> LineSearchResult:
        all_dx_new = []
        all_xnew = []
        all_fvals_new = []
        def make_ret(last_step):
            return LineSearchResult(
                last_step=last_step,
                dx_new=all_dx_new,
                xnew=all_xnew,
                fvals_new=all_fvals_new
            )

        ls_steps = self.ls_steps_init.copy()
        ls_rho = self.solver.ls_rho
        min_step = self.solver.ls_min_step
        while True:
            self.ls_tot_iters += 1
            dx_new = grad.dx[np.newaxis, :] * ls_steps[:, np.newaxis]
            xnew = xk[np.newaxis, :] + dx_new
            fvals_new = self.obj.eval_batch(xnew)
            all_dx_new.append(dx_new)
            all_xnew.append(xnew)
            all_fvals_new.append(fvals_new)
            valid_mask = (fvals_new <= fval + ls_rho * grad.dx_dg * ls_steps)
            if np.any(valid_mask):
                return make_ret(ls_steps[np.argmax(valid_mask)])
            if ls_steps[0] < min_step:
                return make_ret(ls_steps[-1])
            ls_steps *= self.ls_steps_grow

    def logmsg(self, msg):
        if self.solver.verbose:
            self.solver.logmsg(f'{self.iters}: ' + msg)

    def next_iter(self, xk: np.ndarray) -> tuple[np.ndarray, IterStatus]:
        """do the next iteration
        :return: (new xk, status)
        """
        obj = self.obj
        solver = self.solver

        self.iters += 1
        if solver.max_iters is not None and self.iters >= solver.max_iters:
            self.logmsg(f'max iters reached')
            return xk, IterStatus.failed

        self.failed_iters += 1
        if self.failed_iters > self.solver.max_fail_iters:
            self.logmsg('did not converge due to too many failed iterations')
            return xk, IterStatus.failed

        xk = obj.proj(xk)
        fval, sub_diff = obj.eval(xk, need_grad=True)
        self.fval_hist.append(fval)
        try:
            return self._do_next_iter(xk, fval, sub_diff)
        finally:
            self.iter_times.append(time.time() - self.time_start)

    def _tune_subg_slack(
            self,
            xk, fval,
            ls_result: LineSearchResult, grad: TRAFSStep,
            get_grad: typing.Callable[[float], TRAFSStep]) -> tuple[
                LineSearchResult, TRAFSStep]:
        subg_slack = self.subg_slack
        solver = self.solver
        tune_dir = self.subg_slack_tune_dir
        if tune_dir == 0:
            tune_dir = int(self.rng.integers(2)) * 2 - 1
        if (tune_dir < 0 and
                subg_slack * solver.subg_slack_decay < self.min_subg_slack):
            tune_dir = 1

        if tune_dir == 1:
            k = solver.subg_slack_incr
        else:
            assert tune_dir == -1
            k = solver.subg_slack_decay

        new_slack = subg_slack * k
        slack_mul = k
        while (self.min_subg_slack <= new_slack and
               new_slack <= solver.subg_slack_max):
            new_grad = get_grad(new_slack)
            if not np.all(new_grad.dx == grad.dx):
                break
            slack_mul *= k
            new_slack *= k
        else:
            self.logmsg(f'auto eps tuning({tune_dir}):'
                        f' grad does not change up to eps {new_slack:.3g}')
            return ls_result, grad
        if new_grad.dx_dg >= 0:
            if new_grad.df_lb_is_global:
                assert tune_dir == 1
                self.subg_slack_tune_dir = -1
            else:
                self.subg_slack_tune_dir = 0
            self.logmsg(f'auto eps tuning({tune_dir}):'
                        ' new grad makes no progress')
            return ls_result, grad

        new_ls = self._batched_linesearch(new_grad, fval, xk)
        old_decr = ls_result.last_step * grad.dx_dg
        new_decr = new_ls.last_step * new_grad.dx_dg

        if solver.verbose:
            msg = (
                f'auto eps tuning({tune_dir}): dir={tune_dir} ls_step:'
                f' {ls_result.last_step:.3g} => {new_ls.last_step:.3g}'
                f' dx@dg: {grad.dx_dg:.3g} => {new_grad.dx_dg:.3g}')

        if new_decr <= old_decr * solver.subg_slack_tune_thresh:
            if solver.verbose:
                msg += ' (accepted)'
            self.subg_slack = new_slack
            self.subg_slack_est_mul = max(
                solver.subg_slack_est_mul_min,
                min(self.subg_slack_est_mul * slack_mul,
                    solver.subg_slack_est_mul_max))
            ls_result = new_ls
            grad = new_grad
            self.subg_slack_tune_dir = tune_dir
        else:
            if solver.verbose:
                msg += ' (rejected)'
            self.subg_slack_tune_dir = 0

        if solver.verbose:
            self.logmsg(msg)

        return ls_result, grad

    def _do_next_iter(
            self, xk: np.ndarray, fval: float,
            sub_diff: Optimizable.SubDiff) -> tuple[np.ndarray, IterStatus]:
        obj = self.obj
        solver = self.solver

        norm_bound = self.norm_hist.max() * solver.norm_hist_mul
        if norm_bound < 0:
            # no history available
            norm_bound = np.inf
        else:
            norm_bound = np.maximum(norm_bound, obj.trafs_norm_min)

        def get_grad(slack):
            return sub_diff.reduce_trafs(
                subg_slack=slack,
                df_lb_thresh=self.fval_lb + slack - fval,
                norm_bound=norm_bound,
                state=self.obj_grad_state
            )

        subg_slack = self.subg_slack
        grad = get_grad(subg_slack)
        assert grad.df_lb <= 0

        fval_lb = self.fval_lb
        if grad.df_lb_is_global:
            fval_lb = max(fval_lb, fval + grad.df_lb - subg_slack)
            self.fval_lb = fval_lb
            assert fval_lb <= fval + 1e-6, (
                f'(lb={self.fval_lb:.3g}) > (fval={fval:.3g}):'
                f'd={self.fval_lb - fval:.3g}'
                f' |x|={np.linalg.norm(xk, ord=2):.3g}'
                f' {grad.df_lb=:.3g} eps={subg_slack:.3g}'
            )

        if fval - fval_lb <= solver.eps_term:
            self.logmsg(f'finished with {fval_lb=:.3g} {fval=:.3g}'
                        f' (df_lb={grad.df_lb:.3g} eps={subg_slack:.3g}'
                        f' ||x||={np.linalg.norm(xk, ord=2):.3g})')
            return xk, IterStatus.optimal

        if grad.dx_dg == 0:
            # no progress can be made in this iteration
            if grad.df_lb_is_global:
                # if this assertion fails, we should have found a solution,
                # unless df_lb, df_lb_is_global, and dx_dg are inconsistent
                assert subg_slack > self.min_subg_slack
                self.subg_slack *= self.solver.subg_slack_decay
                self.subg_slack_est_mul = max(
                    self.solver.subg_slack_est_mul_min,
                    self.subg_slack_est_mul * self.solver.subg_slack_decay)
                self.logmsg(f'eps decayed to {self.subg_slack:.3g}'
                            ' due to dx@dg = 0')
                # this iteration is considered successful as we have decreased
                # eps with global bound
                self.last_subg_slack = subg_slack
                self.failed_iters = 0
            else:
                self._randomize_subg_slack('dx@dg = 0 without global guarantee')
            return xk, IterStatus.succeeded

        ls_result = self._batched_linesearch(grad, fval, xk)

        if (self.iters >= 2 and ls_result.last_step < 1 and
                self.rng.uniform() < solver.subg_slack_tune_prob):
            subg_slack_tuned = True
            ls_result, grad = self._tune_subg_slack(
                xk, fval, ls_result, grad, get_grad)
            subg_slack = self.subg_slack
        else:
            subg_slack_tuned = False

        fvals_new = np.array(ls_result.fvals_new)
        idx0, idx1 = np.unravel_index(np.argmin(fvals_new), fvals_new.shape)
        # use the min ecountered so far, which is not worse than the line
        # search termination condition
        if fvals_new[idx0, idx1] >= fval:
            # line search is considered failed only if the best point makes no
            # progress at all (even if termination condition is not met, we
            # still accept the solution if it makes progress)
            self._randomize_subg_slack(
                f'no progress after line search (dx@dg={grad.dx_dg:.3g})')
            return xk, IterStatus.succeeded

        self.last_subg_slack = subg_slack
        self.failed_iters = 0

        dx_l2 = np.linalg.norm(ls_result.dx_new[idx0][idx1], ord=2)
        df = fval - fvals_new[idx0, idx1]
        xk = ls_result.xnew[idx0][idx1]
        self.norm_hist.put(dx_l2)
        self.subg_slack_est_hist.put(df * self.iters)

        if solver.verbose and self.iters % solver.verbose_iters == 0:
            self.logmsg(
                f'f={fval:.2g}(d={df:.1e},lb={fval_lb:.1g})'
                f' g={grad.dx_dg:.1e}'
                f' dx={dx_l2:<.1e}'
                f' ls={ls_result.last_step:<.1g}'
                f' nb={norm_bound:.0e}'
                f' eps={subg_slack:.1g}/{self.subg_slack_est_mul:.1g}')

        # adjust slack
        if not subg_slack_tuned:
            if self.iters == 1:
                subg_slack = solver.subg_slack_max
            elif grad.dx_dg <= -subg_slack * (solver.subg_slack_incr * 2 - 1):
                # increase mul if it seems safe, i.e., delta >= 2 * next_slack;
                # note that dx_dg is estimated as -(delta - this_slack)
                self.subg_slack_est_mul = min(
                    self.subg_slack_est_mul * solver.subg_slack_incr,
                    solver.subg_slack_est_mul_max)
                subg_slack *= solver.subg_slack_incr
                self.logmsg(
                    f'eps mul increased to {self.subg_slack_est_mul:.3g}')

            subg_slack = max(
                self.min_subg_slack,
                min(
                    subg_slack,
                    solver.subg_slack_max,
                    self.subg_slack_est_hist.max() * self.subg_slack_est_mul))

        # if fval_lb is good enough we can use this bound
        subg_slack = min(subg_slack, (fval - fval_lb) / 2)

        self.subg_slack = subg_slack

        return xk, IterStatus.succeeded
