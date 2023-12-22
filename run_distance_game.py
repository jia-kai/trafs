#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nsopt import setup_threads
setup_threads()

from nsopt.opt import MethodFactory
from nsopt.prob import DistanceGame

import numpy as np
import matplotlib.pyplot as plt

import argparse

def main():
    factory = MethodFactory(DistanceGame)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-n', type=int, default=10,
                        help='problem dimension')
    parser.add_argument('-m', type=int, help='number of opponent actions')
    parser.add_argument('-k', type=int, help='L2 projection dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    factory.setup_parser(parser)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    prob = DistanceGame.gen_random(args.n, args.m, args.k, rng)

    results = factory.run_solvers(args, prob)

    ref = min(i.fval_hist.min() for i in results.values())
    if ref > 0:
        ref = max(ref / 2, ref - args.eps_term)
    else:
        ref = max(ref * 2, ref - args.eps_term)
    plt.figure()
    def plot(r, label):
        print(f'{label}: f={r.fval:.3g}')
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
