"""the bundle method"""

from .shared import OptimizationResult, UnconstrainedOptimizable, Optimizable
from ..prob.simplex import SimplexConstrainedOptimizable
from ..utils import CPUTimer

import attrs
import numpy as np

import abc
import uuid
import tempfile
import os
import re

# note: I used this method to call python from julia instead of directly
# calling MPBNGCInterface because I did not manage to pass a python callback to
# PyJulia.

JULIA_TEMPLATE = """
using MPBNGCInterface
using PyCall

pyimpl = pyimport("{pymodule}")
pyimpl_fn = pyimpl._g_callback_dict["{callback_name}"]
function fasg(n, x, mm, f, g)
    pyimpl_fn(n, x, mm, f, g)
end

function run(n, x)
    opt = BundleOptions("{callback_name}", {opts})
    prob = BundleProblem(n, fasg, x)
    return solveProblem(prob, opt)
end

function run_simplex(n, x)
    opt = BundleOptions("{callback_name}", {opts})
    lb = zeros(Float64, n)
    ub = ones(Float64, n)
    lbc = ones(Float64, 1)
    ubc = ones(Float64, 1)
    c = ones(Float64, n, 1)
    prob = BundleProblem(n, fasg, x,
        lb, ub, lbc, ubc, c)
    return solveProblem(prob, opt)
end
"""

MPBNGC_LOG_RE = re.compile(r'\s*Iter:\s*([0-9]+)\s*Nfun:\s*([0-9]+)\s')

_g_callback_dict = {}


class UnconstrainedOrSimplexOptimizableMeta(abc.ABCMeta):
    def __instancecheck__(self, obj):
        return isinstance(obj, (UnconstrainedOptimizable,
                                SimplexConstrainedOptimizable))
    def __subclasscheck__(self, cls):
        return issubclass(cls, (UnconstrainedOptimizable,
                                SimplexConstrainedOptimizable))


class UnconstrainedOrSimplexOptimizableBaseWithMeta(
        metaclass=UnconstrainedOrSimplexOptimizableMeta):
    pass


class UnconstrainedOrSimplexOptimizable(
        Optimizable, UnconstrainedOrSimplexOptimizableBaseWithMeta):
    pass

@attrs.define(kw_only=True)
class BundleSolver:
    """a proximal bundle method with the implementation described in
    http://napsu.karmitsa.fi/publications/pbncgc_report.pdf
    """

    PROBLEM_CLASS = UnconstrainedOrSimplexOptimizable

    max_iters: int

    verbose: bool = False
    """whether to print per-iteration information"""

    eps: float = 1e-4

    def solve(self,
              obj: UnconstrainedOrSimplexOptimizable) -> OptimizationResult:
        """solve the problem"""
        import julia.Main as jl
        opts = {
            'OPT_NITER': self.max_iters,
            'OPT_LMAX': 100,
            'OPT_NFASG': self.max_iters * 100,
            'OPT_JMAX': 50,
            'OPT_IPRINT': 3,
            'OPT_EPS': self.eps,
            'OPT_NOUT': 23, # match the file number in our modified MPBNGC
        }
        callback_name = uuid.uuid4().hex
        fval_hist = []
        iter_times = []
        def callback(n, x, mm, f, g):
            fval, subgrad = obj.eval(x, need_grad=True)
            fval_hist.append(fval)
            grad = subgrad.take_arbitrary()
            assert mm == 1 and f.size == 1 and g.size == n and n == obj.x0.size
            f[0] = fval
            g[:, 0] = grad
            iter_times.append(timer.elapsed())

        timer = CPUTimer()
        def work():
            nonlocal fval_hist, iter_times
            jl.eval(JULIA_TEMPLATE.format(
                opts=','.join(f'{k} => {v}' for k, v in opts.items()),
                pymodule=self.__module__,
                callback_name=callback_name,
            ))
            if isinstance(obj, SimplexConstrainedOptimizable):
                fn = jl.run_simplex
            else:
                fn = jl.run
            timer.reset()
            x, fval_lib, ierr, stats = fn(obj.x0.size, obj.x0)
            total_time = timer.elapsed()

            # parse the log to extract function values at each iteration
            fval_hist = np.array(fval_hist)
            iter_times = np.array(iter_times)
            sel_idx = []
            with open('bundleout.txt') as fin:
                for line in fin:
                    if self.verbose:
                        print(line, end='')
                    match = MPBNGC_LOG_RE.match(line)
                    assert match is not None, f'invalid log line: {line}'
                    assert int(match.group(1)) == len(sel_idx), (
                        f'invalid log line: {line};'
                        f' expected iter {len(sel_idx)}'
                    )
                    sel_idx.append(int(match.group(2)) - 1)
            sel_idx = np.array(sel_idx)
            assert len(sel_idx) == stats['NITER'] + 1, (
                f'invalid log file: {len(sel_idx)} iters,'
                f' expected {stats["NITER"] + 1}'
            )
            fval_hist = np.minimum.accumulate(fval_hist[sel_idx])
            iter_times = iter_times[sel_idx]
            x = obj.proj(x)
            fval = obj.eval(x)
            np.testing.assert_allclose(fval, fval_lib, atol=1e-5, rtol=1e-5)

            return OptimizationResult(
                optimal=ierr == 0,
                x=x,
                fval=fval,
                fval_hist=fval_hist,
                iter_times=iter_times,
                iters=stats['NITER'],
                ls_tot_iters=-1,
                time=total_time,
            )


        _g_callback_dict[callback_name] = callback
        oldcwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                return work()
            finally:
                os.chdir(oldcwd)
                del _g_callback_dict[callback_name]
