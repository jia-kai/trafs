#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nsopt import setup_threads
setup_threads()

from nsopt.opt import MethodFactory
from nsopt.prob import LassoRegression, LassoClassification

import numpy as np
import matplotlib.pyplot as plt

import argparse

def main():
    factory = MethodFactory(LassoRegression)
    parser = argparse.ArgumentParser(
        description='Compare different optimization methods on Lasso',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--sparsity', type=float, default=0.95)
    parser.add_argument('-m', '--num-data', type=int, default=512)
    parser.add_argument('-n', '--num-ftr', type=int, default=1600)
    parser.add_argument('-c', '--num-class', type=int, default=5)
    parser.add_argument('--mode', choices=['reg', 'cls'], default='reg',
                        help='problem mode: regression or classification')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    factory.setup_parser(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.mode == 'reg':
        lasso, xtrue = LassoRegression.gen_random(
            args.num_data, args.num_ftr, lam=args.lam, sparsity=args.sparsity,
            rng=rng
        )
    else:
        assert args.mode == 'cls'
        lasso, xtrue = LassoClassification.gen_random(
            args.num_data, args.num_ftr, args.num_class, lam=args.lam,
            sparsity=args.sparsity,
            rng=rng
        )

    results = factory.run_solvers(args, lasso)
    print(f'xtrue: F={lasso.eval(xtrue):.3g} f={lasso.prox_f(xtrue):.3g}')

    plt.figure()
    ref = min(i.fval_hist.min() for i in results.values())
    ref = max(ref / 2, ref - args.eps_term)
    def plot(r, label):
        F = lasso.eval(r.x)
        f = lasso.prox_f(r.x)
        l1 = np.abs(r.x).sum()
        print(f'{label}: {F=:.3g} {f=:.3g} {l1=:.3g}')
        plt.plot(np.arange(len(r.fval_hist)), r.fval_hist - ref, label=label)
    for k, v in results.items():
        plot(v, k)
    plt.xlabel('Iters')
    plt.ylabel('Relative primal')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
