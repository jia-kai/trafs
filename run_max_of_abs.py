#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nsopt import setup_threads
setup_threads()

from nsopt.opt import MethodFactory
from nsopt.prob import MaxOfAbs, GeneralizedMXHILB

import numpy as np
import matplotlib.pyplot as plt

import argparse

def main():
    factory = MethodFactory(MaxOfAbs)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-p', '--problem', choices=['mxhilb', 'mabs'],
                        default='mabs',
                        help='problem to run')
    parser.add_argument('-n', '--num-dim', type=int, default=10,
                        help='problem dimension')
    parser.add_argument('--rand-x0', action='store_true',
                        help='randomize x0')
    factory.setup_parser(parser)
    args = parser.parse_args()

    if args.problem == 'mxhilb':
        prob = GeneralizedMXHILB(args.num_dim)
    else:
        assert args.problem == 'mabs'
        prob = MaxOfAbs(args.num_dim)

    if args.rand_x0:
        x0 = np.random.standard_normal(args.num_dim)
        prob.x0[:] = x0

    results = factory.run_solvers(args, prob)

    plt.figure()
    def plot(r, label):
        print(f'{label}: f={r.fval:.3g}')
        plt.plot(np.arange(len(r.fval_hist)), r.fval_hist, label=label)
    for k, v in results.items():
        plot(v, k)
    plt.xlabel('Iters')
    plt.ylabel('Primal gap')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
