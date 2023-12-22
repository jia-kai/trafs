from .shared import OptimizationResult, Optimizable, StronglyConvexOptimizable

import attrs
import numpy as np

import time
import typing

@attrs.define(kw_only=True)
class GradDescentSolver:
    """projected subgradient descent"""
    PROBLEM_CLASS = Optimizable

    max_iters: int

    lr_mul: typing.Optional[float] = None
    """learning rate multiplier; the schedule is lr_mul / sqrt(t + 1)"""

    verbose: bool = False

    def solve(self, obj: Optimizable) -> OptimizationResult:
        """solve the problem"""
        time_start = time.time()
        fval_hist = []
        iter_times = []
        xk = obj.x0.copy()

        best_fval = np.inf

        avg_x = xk.copy()

        lr_mul = self.lr_mul
        if lr_mul is None:
            if isinstance(obj, StronglyConvexOptimizable):
                param = obj.eval_cvx_params()
                lr_mul = param.R / param.L
            else:
                lr_mul = obj.pgd_default_lr

        for i in range(self.max_iters):
            fval, subgrad = obj.eval(xk, need_grad=True)
            if fval < best_fval:
                best_fval = fval
                best_xk = xk.copy()
            grad = subgrad.take_arbitrary()
            lr = lr_mul / np.sqrt(i + 1)
            xk = obj.proj(xk - lr * grad)
            avg_x = avg_x * (i / (i + 1)) + xk / (i + 1)
            avg_fval = obj.eval(avg_x)
            if avg_fval < best_fval:
                best_fval = avg_fval
                best_xk = avg_x.copy()
            fval_hist.append(min(fval, avg_fval))

            if self.verbose:
                print(f'{i}: {fval=:.3g} {avg_fval=:.3g}')

            iter_times.append(time.time() - time_start)

        if obj.eval(avg_x) < best_fval:
            xk = avg_x
        else:
            xk = best_xk
        return OptimizationResult(
            optimal=False,
            x=xk,
            fval=obj.eval(xk),
            fval_hist=np.array(fval_hist),
            iter_times=np.array(iter_times),
            iters=self.max_iters,
            ls_tot_iters=0,
            time=time.time() - time_start,
        )
