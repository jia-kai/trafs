#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from run_bench import ProbMakers
from nsopt.opt.shared import OptimizationResult

import attrs
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy.stats.mstats import gmean

import argparse
import pickle
import itertools
import typing
from pathlib import Path

@attrs.define
class Number:
    """a number object with latex formatting"""
    val: typing.Optional[float]
    bold: bool = False
    percent: bool = False

    def __lt__(self, other):
        assert isinstance(other, Number), other
        if self.val is None:
            return False
        if other.val is None:
            return True
        return self.val < other.val

    def __le__(self, other):
        assert isinstance(other, Number), other
        if self.val is None:
            return False
        if other.val is None:
            return True
        return self.val <= other.val

    def __repr__(self):
        if self.val is None:
            return '-'
        if self.percent:
            r = rf'{self.val*100:.0f}\%'
        else:
            r = self.fmt(self.val)
        if self.bold:
            r = r'\textbf{' + r + '}'
        return r

    @classmethod
    def fmt(cls, num: float) -> str:
        """format a real number to latex string"""
        if abs(num) < 5e-3:
            if abs(num) == 0:
                return '0'
            s = f'{num:.1e}'
            b, e = s.split('e')
            b = float(b)    # remove trailing zeros
            e = int(e)      # remove leading zeros
            if abs(b - 1) < 1e-9:
                s = fr'\num{{e{e}}}'
            else:
                s = fr'\num{{{b}e{e}}}'
            return s
        return f'{num:.2f}'

    @classmethod
    def bold_extreme(cls, nums: typing.Iterable['Number'], r=min):
        v = r(*nums)
        for i in nums:
            if repr(i) == repr(v):
                i.bold = True


class BenchmarkSummarizer:
    results: dict[str, list[dict[str, OptimizationResult]]]

    eps_split = [1e-3, 1e-6]

    METHOD_DISP_NAMES = {
        'trafs': r'\tras{}',
        'bundle': 'Bundle',
        'gd': 'GD',
        'sa2': r'\SAtwo{}',
        'ista': 'ISTA',
        'fista': 'FISTA',
    }
    """name and order of methods to display"""

    def __init__(self, bench_dir: Path):
        results = {}
        nr_cases = 0
        for prob in ProbMakers.all_makers().keys():
            print(f'Loading {prob}...')
            paths = list(bench_dir.glob(f'{prob}.*.pkl'))
            if nr_cases == 0:
                nr_cases = len(paths)
            else:
                assert nr_cases == len(paths)
            prob_results = []
            for i in range(nr_cases):
                with (bench_dir / f'{prob}.{i:02d}.pkl').open('rb') as f:
                    r = pickle.load(f)
                    assert isinstance(r, dict)
                    for k, v in r.items():
                        if k == 'optimal':
                            continue
                        assert isinstance(v, OptimizationResult)
                        v.fval_hist[-1] = v.fval
                        v.iter_times[-1] = v.time
                    prob_results.append(r)
            results[prob] = prob_results
        self.results = results

    def make_bench_table(self) -> pd.DataFrame:
        row_headers = []
        col_headers = (
            list(itertools.product(
                [fr'$\epsilon={Number.fmt(i)}$'
                 for i in self.eps_split],
                [r'Iter\tnote{a}', r'Time\tnote{a}', r'Solved\tnote{b}']
            )) +
            list(itertools.product(
                ['Termination'],
                [r'Iter\tnote{c}', r'Time\tnote{c}', r'Error\tnote{d}']
            ))
        )
        table = []

        for prob_i, prob in enumerate(itertools.chain(
                ProbMakers.all_makers().keys(),
                ['All'])):
            results: list[dict[str, OptimizationResult]]
            methods: list[str]
            if prob != 'All':
                results = self.results[prob]
                methods = [i for i in self.METHOD_DISP_NAMES.keys()
                           if i in results[0]]
            else:
                results = list(itertools.chain(*self.results.values()))
                available = set(results[0].keys())
                for r in self.results.values():
                    available &= set(r[0].keys())
                methods = [i for i in self.METHOD_DISP_NAMES.keys()
                           if i in available]

            row_headers.extend(itertools.product(
                [prob],
                [self.METHOD_DISP_NAMES[i] for i in methods]
            ))

            def reduce_opt(r: dict[str, OptimizationResult]) -> float:
                r1 = min(v.fval for k, v in r.items() if k != 'optimal')
                r2 = r['optimal']
                if r2 is None:
                    assert prob_i > 5
                    return r1
                assert r1 >= r2
                return r2

            fopts = np.array([reduce_opt(r) for r in results], dtype=np.float64)

            sub_table = np.empty((len(methods), len(col_headers)), dtype=object)

            for eps_i, eps in enumerate(self.eps_split):
                self._fill_eps_perf(sub_table, prob, eps_i, eps, results,
                                    fopts, methods)

            col = len(self.eps_split) * 3
            for meth_i, meth_name in enumerate(methods):
                meth_iter = []
                meth_time = []
                meth_fval = []
                for r in results:
                    mr = r[meth_name]
                    meth_iter.append(mr.iters)
                    meth_time.append(mr.time)
                    meth_fval.append(mr.fval)

                meth_fval = np.array(meth_fval, dtype=np.float64)
                meth_gap = (meth_fval - fopts) / (1 + np.abs(fopts))
                sub_table[meth_i, col] = '{:.0f}'.format(np.mean(meth_iter))
                sub_table[meth_i, col + 1] = Number(np.mean(meth_time))
                sub_table[meth_i, col + 2] = Number(self._gmean(
                    meth_gap, shift=1e-6))

            Number.bold_extreme(sub_table[:, -1])

            table.append(sub_table)

        table = np.vstack(table)
        df = pd.DataFrame(
            table, index=pd.MultiIndex.from_tuples(row_headers),
            columns=pd.MultiIndex.from_tuples(col_headers))
        return df

    def _fill_eps_perf(
            self, sub_table: npt.NDArray[np.object_],
            prob: str, eps_i: int, eps: float,
            results: list[dict[str, OptimizationResult]],
            fopts: npt.NDArray, methods: list[str]):

        thresh = (1 + np.abs(fopts)) * eps + fopts

        solve_iter = []
        solve_time = []
        solve_mask = []
        for method in methods:
            si = []
            st = []
            sm = []
            for ri, r in enumerate(results):
                hist = r[method].fval_hist
                idx = np.argmax(hist <= thresh[ri])
                if hist[idx] <= thresh[ri]:
                    si.append(idx)
                    st.append(r[method].iter_times[idx])
                    sm.append(True)
                else:
                    si.append(2**30)
                    st.append(2**30)
                    sm.append(False)

            sm = np.array(sm, dtype=bool)
            if method == 'trafs' and not np.all(sm):
                fail_i = np.where(~sm)
                print(f'{prob} with {eps=:g}: TRAFS failed at {fail_i}')
            solve_iter.append(si)
            solve_time.append(st)
            solve_mask.append(sm)

        solve_iter = np.array(solve_iter, dtype=np.int32)
        solve_time = np.array(solve_time, dtype=np.float64)
        solve_mask = np.array(solve_mask, dtype=bool)

        min_iter = solve_iter.min(axis=0)
        min_time = solve_time.min(axis=0)
        for meth_i in range(len(methods)):
            meth_mask = solve_mask[meth_i]
            if not np.any(meth_mask):
                sub_table[meth_i, eps_i * 3] = Number(None)
                sub_table[meth_i, eps_i * 3 + 1] = Number(None)
                sub_table[meth_i, eps_i * 3 + 2] = Number(0, percent=True)
                continue
            meth_iter = solve_iter[meth_i][meth_mask]
            meth_time = solve_time[meth_i][meth_mask]
            sub_table[meth_i, eps_i * 3] = Number(self._gmean(
                meth_iter / min_iter[meth_mask]))
            sub_table[meth_i, eps_i * 3 + 1] = Number(self._gmean(
                meth_time / min_time[meth_mask]))
            sub_table[meth_i, eps_i * 3 + 2] = Number(
                np.mean(meth_mask.astype(np.int32)), percent=True)

        for i in range(3):
            Number.bold_extreme(sub_table[:, eps_i*3+i],
                                r=max if i == 2 else min)

    def _gmean(self, val, *, shift: float=0) -> float:
        return gmean(val + shift) - shift


class LatexTableProc:
    """process latex table string generated by pandas"""
    _table: str
    _lines: list[str]

    def __init__(self, table):
        self._table = table
        self._lines = None

    def as_str(self) -> str:
        if self._table is None:
            self._table = '\n'.join(self._lines)
            self._lines = None
        return self._table

    def as_lines(self) -> list[str]:
        if self._lines is None:
            self._lines = self._table.split('\n')
            self._table = None
        return self._lines

    def merge_multirow(self, col: int) -> typing.Self:
        """use multirow at the given column to merge equal values; must be used
        after :meth:`fix_cline`"""
        lines = self.as_lines()
        table = []
        table_lidx = []
        insert_after = {}
        for i, line in enumerate(lines):
            if i > 2 and ' & ' in line:
                table.append(line.split(' & '))
                table_lidx.append(i)
        table_np = np.empty((len(table), len(table[0])), dtype=object)
        for i, j in zip(table_np, table):
            i[:] = j
        table = table_np
        s = 0
        while s < len(table):
            t = s
            val = table[s, col]
            while t < len(table) and table[t, col] == val:
                t += 1
            if t > s + 1:
                table[s, col] = r'\multirow[c]{%d}{*}{%s}' % (t - s, val)
                table[s+1:t, col] = ' '
                insert_after[table_lidx[s] - 1] = insert_after[
                    table_lidx[t - 1]] = r'\cline{%d-%d}' % (col + 1, col + 1)
            s = t
        for i, j in zip(table_lidx, table):
            lines[i] = ' & '.join(j)
            if (a := insert_after.get(i)):
                lines[i] += '\n' + a
        self.as_str()
        return self

    def remove_table_last_cline(self) -> typing.Self:
        """remove the last cline above bottomrule in a talbe"""
        lines = self.as_lines()
        def find_by_mark(mark):
            for i, j in enumerate(lines):
                if j.strip() == mark:
                    assert lines[i - 1].strip().startswith(r'\cline'), (
                        mark, lines[i - 1])
                    del lines[i - 1]
                    return True
            return False
        if not (find_by_mark(r'\bottomrule') or find_by_mark(r'\end{tabular}')):
            raise RuntimeError('last cline not found')
        return self

    def fix_cline(self) -> typing.Self:
        """replace cline with cmidrule, and remove consecutive clines"""
        lines = self.as_lines()
        for i, line in enumerate(lines):
            if line.startswith(r'\cline'):
                parts = line.split(' ')
                for p in parts:
                    assert p.startswith(r'\cline')
                lines[i] = parts[0].replace(r'\cline', r'\cmidrule(lr)')
            else:
                assert r'\cline' not in line, repr(line)
        return self

    def add_multirow_header(self, nrow: int, *headers: str) -> typing.Self:
        """add header for the index columns; if ``nrow==1``, it will be a normal
        cell instead of a multirow header"""
        lines = self.as_lines()
        assert lines[0].startswith(r'\begin')
        if lines[1].startswith(r'\toprule'):
            s = 2
        else:
            s = 1
        for i in range(nrow):
            cols = lines[s + i].split('&')
            for j in range(len(headers)):
                assert cols[j].isspace(), (s, i, j, cols[:len(headers)])

        cols = lines[s].split('&')
        for i, j in enumerate(headers):
            assert cols[i].isspace()
            if nrow == 1:
                cols[i] = j
            else:
                cols[i] = r' \multirow[c]{%d}{*}{%s} ' % (nrow, j)
        lines[s] = ' & '.join(cols)

        return self

    def replace_multicol(self, placeholder: str, content: str,
                         ncol: int) -> typing.Self:
        """replace ``placeholder`` with a multicol with given content and size"""
        lines = self.as_lines()
        for l_i, line in enumerate(lines):
            if placeholder in line:
                cols = line.split(' & ')
                for c_i, c in enumerate(cols):
                    if c.strip() == placeholder:
                        d = r'\multicolumn{%d}{c}{%s}' % (ncol, content)
                        assert cols[-1].strip().endswith(r'\\')
                        assert c_i + ncol <= len(cols)
                        is_end = c_i + ncol == len(cols)
                        for i in cols[c_i+1:c_i+ncol]:
                            if i.strip() and not (
                                i is cols[-1] and i.strip() == r'\\'):
                                raise RuntimeError(f'nonempty col: {i}')
                        cols[c_i:c_i + ncol] = [d]
                        if is_end:
                            cols[-1] += r' \\'
                        lines[l_i] = ' & '.join(cols)
                        return self
        raise RuntimeError(f'placeholder {placeholder} not found')

    def set_multirow_width(self, nrow: int, width: str) -> typing.Self:
        """set the width of multirow commands that span the given number of
        rows"""
        table = self.as_str()
        cmd = r'\multirow[c]{%d}' % nrow
        src = cmd + '{*}'
        dst = cmd + ('{%s}' % width)
        self._table = table.replace(src, dst)
        return self

    def remove_header_line(self) -> typing.Self:
        """remove the header line (i.e., the first line)"""
        lines = self.as_lines()
        l = lines[2].strip()
        assert l.startswith('&') and l.endswith(r'\\'), l
        del lines[2]
        assert lines[2].strip() == r'\midrule'
        del lines[2]
        return self

    def insert_line_begin(self, line: int, content: str) -> typing.Self:
        """insert at the beginning of given line"""
        lines = self.as_lines()
        lines[line] = content + lines[line]
        return self

def main():
    parser = argparse.ArgumentParser(
        description='Summarize benchmark results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bench-dir', default='bench-out',
                        help='directory to load benchmark results')
    parser.add_argument('-o', '--output', required=True,
                        help='output latex file')
    parser.add_argument('--print', action='store_true',
                        help='print the table to stdout')
    args = parser.parse_args()

    summarizer = BenchmarkSummarizer(Path(args.bench_dir))
    df = summarizer.make_bench_table()
    if args.print:
        print(df)

    latex = df.style.to_latex(
        hrules=True, clines='skip-last;data',
        multicol_align='c', column_format='ll' + 'r'*df.shape[1],
    )
    latex = (LatexTableProc(latex)
             .remove_table_last_cline()
             .fix_cline()
             .add_multirow_header(2, 'Problem', 'Method')
             .as_str())

    with open(args.output, 'w') as fout:
        fout.write(latex)

if __name__ == '__main__':
    main()
