"""the accelerated proximal gradient method"""
# see https://sci-hub.se/https://doi.org/10.1137/080716542

from .shared import OptimizationResult, ProximalGradOptimizable

import attrs
import numpy as np

import time

@attrs.define(kw_only=True)
class ProximalGradSolver:
    PROBLEM_CLASS = ProximalGradOptimizable

    max_iters: int

    init_L: float = 0.05
    """initial guess of Lipschitz constant"""

    L_growth: float = 1.5
    """Lipschitz constant growth factor"""

    use_fast: bool = True
    """whether to use the accelerated version in the fast ISTA paper"""

    def solve(self, obj: ProximalGradOptimizable) -> OptimizationResult:
        """solve the problem"""
        time_start = time.time()
        fval_hist = []
        iter_times = []
        xk = obj.x0.copy()
        L = self.init_L
        ls_tot_iters = 0

        if self.use_fast:
            tk = np.array(1, dtype=np.float128)
            prox_old = xk

        optimal = False
        for _ in range(self.max_iters):
            fval, grad = obj.prox_f(xk, need_grad=True)
            fval_hist.append(fval + obj.prox_g(xk))

            while True:
                xnew = obj.prox_minx(xk - grad / L, L)
                d = xnew - xk
                thresh = fval + d.dot(grad) + L/2 * d.dot(d) + obj.prox_g(xnew)
                if obj.eval(xnew) <= thresh:
                    break
                ls_tot_iters += 1
                L *= self.L_growth
                if not np.isfinite(L):
                    # optimal solution found
                    optimal = True
                    iter_times.append(time.time() - time_start)
                    break

            if optimal:
                break
            if self.use_fast:
                tnew = (1. + np.sqrt(1. + 4. * tk * tk)) / 2.
                prox_new = xnew
                xnew = xnew + ((tk - 1.) / tnew) * (xnew - prox_old)
                prox_old = prox_new
                tk = tnew
            xk = xnew
            iter_times.append(time.time() - time_start)

        assert ls_tot_iters > 0, 'initial L is too large'
        return OptimizationResult(
            optimal=optimal,
            x=xk,
            fval=obj.eval(xk),
            fval_hist=np.array(fval_hist),
            iter_times=np.array(iter_times),
            iters=self.max_iters,
            ls_tot_iters=ls_tot_iters,
            time=time.time() - time_start,
        )
