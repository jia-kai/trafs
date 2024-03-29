#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nsopt import setup_threads
setup_threads()

from nsopt.opt import MethodFactory
from nsopt.prob import (MaxOfAbs, MaxQ, MXHILB, ChainedLQ, ChainedCB3I,
                        ChainedCB3II)

import numpy as np
import matplotlib.pyplot as plt

import argparse

prob_map = {
    'mabs': MaxOfAbs,
    'mxhilb': MXHILB,
    'maxq': MaxQ,
    'clq': ChainedLQ,
    'ccb31': ChainedCB3I,
    'ccb32': ChainedCB3II,
}

def main():
    factory = MethodFactory(MaxOfAbs)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-p', '--problem', choices=prob_map.keys(),
                        default='mabs',
                        help='problem to run')
    parser.add_argument('-n', '--num-dim', type=int, default=10,
                        help='problem dimension')
    parser.add_argument('--rand-x0', action='store_true',
                        help='randomize x0')
    factory.setup_parser(parser)
    args = parser.parse_args()

    prob = prob_map[args.problem](args.num_dim)

    if args.rand_x0:
        x0 = np.random.standard_normal(args.num_dim)
        prob.x0[:] = x0

    results = factory.run_solvers(args, prob)
    opt = float(prob.get_optimal_value())

    plt.figure()
    def plot(r, label):
        print(f'{label}: f={r.fval:.3g}')
        plt.plot(np.arange(len(r.fval_hist)), r.fval_hist - opt, label=label)
    for k, v in results.items():
        plot(v, k)
    plt.xlabel('Iters')
    plt.ylabel('Primal gap')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
