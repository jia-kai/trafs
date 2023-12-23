"""optimization methods"""

from .trafs import TRAFSSolver
from .proximal import ProximalGradSolver
from .gd import GradDescentSolver
from .bundle import BundleSolver
from .sa2 import SA2Solver

import argparse
import typing

if typing.TYPE_CHECKING:
    from .shared import OptimizationResult

class MethodFactory:
    METHODS_ALL = {
        'trafs': (TRAFSSolver,
                  lambda args: dict(
                      verbose=args.verbose, verbose_iters=1,
                      eps_term=args.eps_term,
                      max_iters=args.max_iters)
                  ),
        'ista': (ProximalGradSolver,
                 lambda args: dict(
                     max_iters=args.max_iters, use_fast=False)
                 ),
        'fista': (ProximalGradSolver,
                  lambda args: dict(
                      max_iters=args.max_iters, use_fast=True)
                  ),
        'gd': (GradDescentSolver,
               lambda args: dict(max_iters=args.max_iters,
                                 verbose=args.verbose)
               ),
        'bundle': (BundleSolver,
                   lambda args: dict(
                       max_iters=args.max_iters, verbose=args.verbose,
                       eps=args.eps_term)
                   ),
        'sa2': (SA2Solver,
                lambda args: dict(
                    max_iters=args.max_iters)
                ),
    }

    def __init__(self, problem_class: typing.Optional[type]):
        """:param problem_class: the class of the optimization problem, one of
        the classes defined in ``.shared`` to filter available methods"""
        if problem_class is None:
            self.methods = self.METHODS_ALL
        else:
            self.methods = {
                k: v for k, v in self.METHODS_ALL.items()
                if issubclass(problem_class, v[0].PROBLEM_CLASS)
            }

    def setup_parser(self, parser: argparse.ArgumentParser,
                     default_max_iters=5000):
        """add method arguments to parser"""
        parser.add_argument('--max-iters', type=int, default=default_max_iters)
        parser.add_argument('--eps-term', type=float, default=1e-6,
                            help='eps for termination')
        parser.add_argument('--supress', nargs='*', default=[],
                            choices=self.methods.keys(),
                            help='which methods to not run')
        parser.add_argument('--only', nargs='*', default=[],
                            choices=self.methods.keys(),
                            help='which methods to run')
        parser.add_argument('--verbose', action='store_true',
                            help='whether to output optimizer internals')

    def run_solvers(self, args, obj) -> dict[str, "OptimizationResult"]:
        """run all solvers; return a dict mapping method name to result"""
        results = {}
        for k, (cls, mkarg) in self.methods.items():
            if k in args.supress or (args.only and k not in args.only):
                continue
            if not isinstance(obj, cls.PROBLEM_CLASS):
                continue
            print(f'Running {k} on {obj} ...', flush=True)
            solver = cls(**mkarg(args))
            r = solver.solve(obj)
            results[k] = r
            print(f'{k}: f={r.fval:.5g} iters={r.iters} time={r.time:.3f}s'
                  f' optimal={r.optimal} ls={r.ls_tot_iters}',
                  flush=True)
        return results
