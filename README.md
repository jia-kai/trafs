# TRAFS: A Nonsmooth Convex Optimization Algorithm with $\mathcal{O}\left(\frac{1}{\epsilon}\right)$ Iteration Complexity

This repository is the implementation of the TRAFS algorithm, a nonsmooth convex
optimization algorithm that both provides better guarantees and converges
significantly faster in numerical experiments compared to previous methods. Read
our [paper](https://arxiv.org/abs/2311.06205) for more details.

## Setup

The program has only been tested on Linux.


First run `git submodule update --init --recursive` to initialize the
submodules. Make sure there is a recent C++ compiler, and `gfortran` is
available. Install julia to the system. Run `uv sync` to setup the python env.
Use `./setup-deps.sh` to compile and install julia dependencies.

## Usage

Use `./run_bench.sh` to reproduce all the numerical experiments reported in the
paper. After the script finishes, use `./make_bench_table.py` to generate the
LaTex tables summarizing the results.

Use `run_*.py` to run individual experiment.

All the optimization algorithms are in the [opt](nsopt/opt) module. The
benchmark problems are in the [prob](nsopt/prob) module. The code should be
self-explanatory.
