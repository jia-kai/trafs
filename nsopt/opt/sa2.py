from .shared import (OptimizationResult, KnownLipschitzOptimizable,
                     UnconstrainedOptimizable)
from ..utils import CPUTimer

import attrs
import numpy as np

@attrs.define(kw_only=True)
class SA2Solver:
    """Subgradient Method with Double Simple Averaging"""

    PROBLEM_CLASS = UnconstrainedOptimizable

    max_iters: int

    def solve(self, obj: UnconstrainedOptimizable) -> OptimizationResult:
        fval_hist = []
        iter_times = []
        xk = obj.x0.copy()

        if isinstance(obj, KnownLipschitzOptimizable):
            param = obj.eval_cvx_params()
            lr_mul = param.R / param.L
        else:
            lr_mul = obj.pgd_default_lr

        timer = CPUTimer()
        for i in range(self.max_iters):
            fval, subgrad = obj.eval(xk, need_grad=True)
            fval_hist.append(fval)
            grad = subgrad.take_arbitrary()

            if i == 0:
                avg_grad = grad
            else:
                avg_grad = avg_grad * (i / (i + 1)) + grad / (i + 1)

            eta = np.sqrt(i + 1) * lr_mul
            xkp = obj.x0 - eta * avg_grad
            xk = xk * ((i + 1) / (i + 2)) + xkp / (i + 2)
            iter_times.append(timer.elapsed())

        return OptimizationResult(
            optimal=False,
            x=xk,
            fval=obj.eval(xk),
            fval_hist=np.array(fval_hist),
            iter_times=np.array(iter_times),
            iters=self.max_iters,
            ls_tot_iters=0,
            time=timer.elapsed(),
        )
