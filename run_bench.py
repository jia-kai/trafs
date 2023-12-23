#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nsopt import setup_threads
setup_threads()

from nsopt.opt import MethodFactory
from nsopt.opt.shared import Optimizable
import nsopt.prob

import attrs
import numpy as np

import argparse
import pickle
import typing

@attrs.frozen
class ProbSpec:
    seed: int
    """random seed"""

    idx: int
    """index of the instance in the benchmark"""

    bench_size: int
    """total number of instances of this problem in the benchmark"""

    @property
    def idx_ratio(self):
        return self.idx / (self.bench_size - 1)

    def make_rng(self, prob_name: str):
        s = tuple(map(ord, prob_name))
        return np.random.default_rng((self.seed, self.idx, self.bench_size) + s)

def int_interp(a: int, b: int, t: float) -> int:
    """uniform interpolation between integer values"""
    return int(round((b - a) * t + a))

def geo_interp(a: float, b: float, t: float) -> float:
    """Geometric interpolation between a and b
    :param t: interpolation parameter, 0 <= t <= 1
    """
    return float(a * (b / a) ** t)

class ProbMakers:
    @classmethod
    def SPL(cls, spec: ProbSpec):
        n = int_interp(10, 5000, spec.idx_ratio)
        return nsopt.prob.MaxOfAbs(n)

    @classmethod
    def DPL(cls, spec: ProbSpec):
        n = int_interp(10, 1200, spec.idx_ratio)
        return nsopt.prob.GeneralizedMXHILB(n)

    @classmethod
    def LLR(cls, spec: ProbSpec):
        rng = spec.make_rng('LLR')
        while True:
            m = int(rng.integers(8, 2049))
            n = int(rng.integers(8, 2049))
            if m * n <= 1024**2:
                break
        lam = geo_interp(1e-4, 1e-1, rng.uniform())
        prob, _ = nsopt.prob.LassoRegression.gen_random(m, n, lam, rng=rng)
        return prob

    @classmethod
    def LLC(cls, spec: ProbSpec):
        rng = spec.make_rng('LLC')
        while True:
            m = int(rng.integers(8, 2049))
            n = int(rng.integers(8, 2049))
            k = int(rng.integers(3, 11))
            if m * n * k <= 1024**2 * 5:
                break
        lam = geo_interp(1e-4, 1e-1, rng.uniform())
        prob, _ = nsopt.prob.LassoClassification.gen_random(m, n, k, lam,
                                                            rng=rng)
        return prob

    @classmethod
    def DG(cls, spec: ProbSpec):
        rng = spec.make_rng('DG')
        while True:
            m = int(rng.integers(8, 1025))
            n = int(rng.integers(8, 1025))
            if m * n <= 400 ** 2:
                break
        return nsopt.prob.DistanceGame.gen_random(n, m, rng=rng)

    @classmethod
    def all_makers(cls) -> dict[str, typing.Callable[[ProbSpec], Optimizable]]:
        return {name: getattr(cls, name) for name in dir(cls)
                if name.isupper() and callable(getattr(cls, name))}

def main():
    makers = ProbMakers.all_makers()
    parser = argparse.ArgumentParser(
        description='run benchmark experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='output file')
    parser.add_argument('-p', '--prob-name', type=str, required=True,
                        choices=makers.keys())
    parser.add_argument('-i', '--idx',  required=True, type=int,
                        help='index of the instance')
    parser.add_argument(
        '-s', '--bench-size', type=int, required=True,
        help='total number of instances in the benchmark of this problem')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    MethodFactory(None).setup_parser(parser, default_max_iters=50000)
    args = parser.parse_args()
    assert 0 <= args.idx < args.bench_size

    spec = ProbSpec(seed=args.seed, idx=args.idx, bench_size=args.bench_size)
    prob = makers[args.prob_name](spec)
    results = MethodFactory(type(prob)).run_solvers(args, prob)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
