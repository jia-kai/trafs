from .shared import Optimizable, OptimizationResult, TRAFSStep
from ..utils import setup_pyx_import, CPUTimer
with setup_pyx_import():
    from .trafs_utils import RotationBuffer

import attrs
import numpy as np

import typing
import enum
import copy

@attrs.frozen(kw_only=True)
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

    subg_slack_init: float = 0.1
    """functional subdifferential eps to use in the first iteration"""

    subg_slack_decay: float = .5
    """eps decay factor when -dx @ dg is not big enough"""

    subg_slack_incr: float = 1.5
    """eps increase factor when -dx @ dg is big enough"""

    subg_slack_tune_prob: float = .2
    """probability to automatically tune eps by computing another step using the
    same xk"""

    subg_slack_tune_prob_high: float = .8
    """probability to tune eps when previous tuning succeeded"""

    subg_slack_tune_max_nr_try: int = 12
    """max number of tries if subdiff does not change during tuning the
    subdifferential eps"""

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
            time=runtime.timer.elapsed(),
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

    retry = 'retry'
    """retry current iteration (do not record iter number and history); used
    internally by :class:`TRAFSRuntime`"""

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

    subg_slack_tune_prob: float

    rng: np.random.Generator

    subg_slack_tune_cnt: int = 0
    """number of times eps is tuned; used to control overhead of random
    tuning"""

    subg_slack_est_mul: float = 1.0
    last_subg_slack: float = 1.0
    """last successful eps in functional subdifferential"""

    subg_slack_tune_dir: int = 0
    """direction of eps tuning: -1 for decrease, 1 for increase, 0 for random"""

    subg_slack_is_from_random: bool = False
    """when progress is stalled, this flag is True if slack randomization is
    ever used"""

    fval_lb: float = -np.inf
    timer: CPUTimer = attrs.field(factory=CPUTimer)
    ls_tot_iters: int = 0
    iters: int = 0
    iter_times: list[float] = attrs.field(factory=list)
    fval_hist: list[float] = attrs.field(factory=list)
    failed_iters: int = 0
    obj_grad_state: dict = attrs.field(factory=dict)

    def __init__(self, solver: "TRAFSSolver", obj: Optimizable):
        # copy the rng state to ensure multiple runs with the same RNG are
        # reproducible
        rng_seed = copy.deepcopy(solver.rng).bytes(16)
        self.__attrs_init__(
            obj=obj,
            solver=solver,
            subg_slack=solver.subg_slack_init,
            min_subg_slack=min(solver.eps_term / 10, 1e-10),
            ls_steps_init=np.power(solver.ls_tau,
                                   np.arange(solver.ls_batch_size)),
            ls_steps_grow=np.power(solver.ls_tau, solver.ls_batch_size),
            norm_hist=RotationBuffer(solver.norm_hist_winsize),
            subg_slack_est_hist=RotationBuffer(solver.subg_slack_est_winsize),
            subg_slack_tune_prob=solver.subg_slack_tune_prob,
            rng=np.random.default_rng(
                list(rng_seed) +
                list(map(ord, obj.__class__.__name__)) +
                [obj.x0.size]
            )
        )

    def _randomize_subg_slack(self, reason: str):
        """try a random functional subdifferential slack after failure to find a
        descent direction"""
        self.subg_slack_is_from_random = True
        kmin = max(np.log(self.min_subg_slack / self.subg_slack), -np.log(100))
        kmax = np.log(1000)
        self.subg_slack = max(
            np.exp(self.rng.uniform(kmin, kmax)) * self.last_subg_slack,
            self.min_subg_slack)
        self.logmsg(
            f'failed to find descent direction due to {reason};'
            f' randomize eps {self.subg_slack:.3g}')

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
        xk, status = self._do_next_iter(xk, fval, sub_diff)

        if status == IterStatus.retry:
            self.iters -= 1
            status = IterStatus.succeeded
        else:
            self.fval_hist.append(fval)
            self.iter_times.append(self.timer.elapsed())
        return xk, status

    def _tune_subg_slack(
            self,
            xk, fval,
            ls_result: LineSearchResult, grad: TRAFSStep,
            get_grad: typing.Callable[[float], TRAFSStep]) -> tuple[
                bool, LineSearchResult, TRAFSStep]:
        """try tuning the eps
        :return: whether better eps is found, new line search result, new grad
        """
        self.subg_slack_tune_cnt += 1
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
        nr_try = 0
        while (new_slack >= self.min_subg_slack and
               nr_try < solver.subg_slack_tune_max_nr_try):
            new_grad = get_grad(new_slack)
            if not np.all(new_grad.dx == grad.dx):
                break
            slack_mul *= k
            new_slack *= k
            nr_try += 1
        else:
            self.logmsg(f'auto eps tuning({tune_dir}):'
                        ' grad does not change up to eps'
                        f' {subg_slack:.3g}=>{new_slack:.3g} ({nr_try} tries)')
            self.subg_slack_tune_dir = -tune_dir
            return False, ls_result, grad
        if new_grad.dx_dg >= 0:
            if new_grad.df_lb_is_global:
                assert tune_dir == 1
                self.subg_slack_tune_dir = -1
            else:
                self.subg_slack_tune_dir = 0
            self.logmsg(f'auto eps tuning({tune_dir}):'
                        ' new grad makes no progress')
            return False, ls_result, grad

        new_ls = self._batched_linesearch(new_grad, fval, xk)
        old_decr = ls_result.last_step * grad.dx_dg
        new_decr = new_ls.last_step * new_grad.dx_dg

        if solver.verbose:
            msg = (
                f'auto eps tuning({tune_dir}): '
                f' mul={slack_mul:.3g}'
                f' ls_step: {ls_result.last_step:.3g} => {new_ls.last_step:.3g}'
                f' dx@dg: {grad.dx_dg:.3g} => {new_grad.dx_dg:.3g}')

        if new_decr < old_decr:
            if solver.verbose:
                msg += ' (accepted)'
            self.subg_slack = new_slack
            self.subg_slack_est_mul = max(
                solver.subg_slack_est_mul_min,
                min(self.subg_slack_est_mul * slack_mul,
                    solver.subg_slack_est_mul_max))
            self.subg_slack_tune_dir = tune_dir
            ls_result = new_ls
            grad = new_grad
            accepted = True
        else:
            if solver.verbose:
                msg += ' (rejected)'
            self.subg_slack_tune_dir = -tune_dir
            accepted = False

        if solver.verbose:
            self.logmsg(msg)

        return accepted, ls_result, grad

    def _do_next_iter(
            self, xk: np.ndarray, fval: float,
            sub_diff: Optimizable.SubDiff) -> tuple[np.ndarray, IterStatus]:
        solver = self.solver

        norm_bound = self.norm_hist.max() * solver.norm_hist_mul
        if norm_bound < 0:
            # no history available
            norm_bound = np.inf

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

        if grad.dx_dg >= 0:
            # no progress can be made in this iteration

            def decay_slack():
                self.subg_slack *= solver.subg_slack_decay
                self.subg_slack_est_mul = max(
                    solver.subg_slack_est_mul_min,
                    self.subg_slack_est_mul * solver.subg_slack_decay)

            if grad.df_lb_is_global:
                if subg_slack * solver.subg_slack_decay <= self.min_subg_slack:
                    self._randomize_subg_slack(
                        'dx@dg = 0 with global guarantee, but eps is too small')
                    return xk, IterStatus.retry

                decay_slack()
                self.logmsg(f'eps decayed to {self.subg_slack:.3g}'
                            ' due to dx@dg = 0')
                if not self.subg_slack_is_from_random:
                    # this iteration is considered successful as we have
                    # decreased eps with global bound
                    self.last_subg_slack = subg_slack
                    self.failed_iters = 0
                else:
                    # do not count the current iteration as a failed one (so
                    # only slack randomization is counted as a failed iteration)
                    assert self.failed_iters > 0
                    self.failed_iters -= 1
            else:
                if (self.subg_slack * solver.subg_slack_decay ** 5 >
                        self.min_subg_slack):
                    # when current slack is large, it is likely that
                    # df_lb_is_global being false is due to the solver is unable
                    # to prove global lower bound
                    decay_slack()
                    self.logmsg(f'eps decayed to {self.subg_slack:.3g}'
                                ' due to dx@dg = 0 without global guarantee but'
                                ' large enough')
                    # similar to above, only count random slack as failed
                    assert self.failed_iters > 0
                    self.failed_iters -= 1
                else:
                    self._randomize_subg_slack(
                        'dx@dg = 0 without global guarantee')
            return xk, IterStatus.retry

        ls_result = self._batched_linesearch(grad, fval, xk)

        subg_slack_tuned = False
        def run_slack_tune(force_dir=None) -> bool:
            nonlocal subg_slack_tuned, subg_slack, ls_result, grad
            if force_dir is not None:
                self.subg_slack_tune_dir = force_dir
            subg_slack_tuned, ls_result, grad = self._tune_subg_slack(
                xk, fval, ls_result, grad, get_grad)
            subg_slack = self.subg_slack
            if subg_slack_tuned:
                self.subg_slack_tune_prob = solver.subg_slack_tune_prob_high
            else:
                self.subg_slack_tune_prob = solver.subg_slack_tune_prob
            return subg_slack_tuned

        if all(i.min() >= fval for i in ls_result.fvals_new):
            # no progress after line search; try tuning eps
            if subg_slack > self.min_subg_slack:
                run_slack_tune(force_dir=-1)
            if not subg_slack_tuned:
                run_slack_tune(force_dir=1)
            if not subg_slack_tuned or all(
                    i.min() >= fval for i in ls_result.fvals_new):
                # randomization as a last resort
                self._randomize_subg_slack(
                    f'no progress after line search (dx@dg={grad.dx_dg:.3g})')
                return xk, IterStatus.retry
        elif self.iters >= 2:
            if grad.dx_dg <= -subg_slack * (solver.subg_slack_incr * 2 - 1):
                # dx_dg seems large enough, try increasing eps
                self.logmsg(f'try increasing eps since dx@dg = {grad.dx_dg:.3g}'
                            ' is large enough')
                run_slack_tune(force_dir=1)
            elif (self.subg_slack_tune_cnt <=
                  self.iters * self.subg_slack_tune_prob * 1.2 and
                  self.rng.uniform() < self.subg_slack_tune_prob):
                # tune by chance
                run_slack_tune()

        fvals_new = np.array(ls_result.fvals_new)
        # use the min ecountered so far, which is not worse than the line
        # search termination condition
        idx0, idx1 = np.unravel_index(np.argmin(fvals_new), fvals_new.shape)
        assert fvals_new[idx0, idx1] < fval

        self.last_subg_slack = subg_slack
        self.subg_slack_is_from_random = False
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
                # we used the initial default value at this iteration; a better
                # estimation is now available from subg_slack_est_hist
                subg_slack = np.inf

            subg_slack = max(
                self.min_subg_slack,
                min(
                    subg_slack,
                    self.subg_slack_est_hist.max() * self.subg_slack_est_mul))

        # if fval_lb is good enough we can use this bound
        subg_slack = min(subg_slack, (fval - fval_lb) / 2)

        self.subg_slack = subg_slack

        return xk, IterStatus.succeeded
