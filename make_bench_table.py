#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from run_bench import ProbMakers
from nsopt.opt.shared import OptimizationResult

import attrs
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    percent_prec: int = 0
    prefer_sci: bool = False
    tnote: typing.Optional[str] = None

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
            r = rf'{{:.{self.percent_prec}f}}\%'.format(self.val * 100)
        else:
            r = self.fmt(self.val, self.prefer_sci)
        if self.bold:
            r = r'\textbf{' + r + '}'
        if self.tnote is not None:
            r += r'\tnote{' + self.tnote + '}'
        return r

    @classmethod
    def fmt(cls, num: float, prefer_sci: bool = False) -> str:
        """format a real number to latex string"""
        if abs(num) < 5e-3 or (prefer_sci and abs(num) > 1e3):
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
    def bold_extreme(cls, nums: typing.Iterable['Number'], reduction=min):
        rv = repr(reduction(*nums))
        for i in nums:
            if repr(i) == rv:
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

    trafs_speedup: list[float]
    """speedup of TRAFS over bundle for different eps values; filled by
    :meth:`make_bench_table`"""

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

        for prob in itertools.chain(ProbMakers.all_makers().keys(),
                                    ['All']):
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

            opt_is_tight: bool = True
            def reduce_opt(r: dict[str, OptimizationResult]) -> float:
                nonlocal opt_is_tight
                r1 = min(v.fval for k, v in r.items() if k != 'optimal')
                r2 = r['optimal']
                if r2 is None:
                    opt_is_tight = False
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
            for i in sub_table[:, -1]:
                i.prefer_sci = True

            if prob == 'All':
                for i in sub_table:
                    for j in range(len(self.eps_split)):
                        i[j*3+2].percent_prec = 1

                self._compute_speedup(results, fopts)

            if not opt_is_tight:
                for i in sub_table[:, -1]:
                    i.tnote = r'*'

            table.append(sub_table)

        table = np.vstack(table)
        df = pd.DataFrame(
            table, index=pd.MultiIndex.from_tuples(row_headers),
            columns=pd.MultiIndex.from_tuples(col_headers))
        return df

    def _compute_speedup(self, results: list[dict[str, OptimizationResult]],
                         fopts: npt.NDArray) -> None:

        speedup = []

        meth0 = 'trafs'
        meth1 = 'bundle'
        for eps in self.eps_split:
            thresh = (1 + np.abs(fopts)) * eps + fopts
            items = []
            for ri, r in enumerate(results):
                thresh_i = thresh[ri]

                def get_time(rm: OptimizationResult):
                    hist = rm.fval_hist
                    idx = np.argmax(hist <= thresh_i)
                    assert hist[idx] <= thresh_i
                    return rm.iter_times[idx]

                if max(r[meth0].fval, r[meth1].fval) <= thresh_i:
                    items.append(get_time(r[meth1]) / get_time(r[meth0]))
            speedup.append(self._gmean(items))
        self.trafs_speedup = speedup

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
                                reduction=max if i == 2 else min)

    def _gmean(self, val, *, shift: float=0) -> float:
        val = np.ascontiguousarray(val, dtype=np.float64)
        if np.all(val == 0):
            return 0.0
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

def write_defs(summarizer: BenchmarkSummarizer, df: pd.DataFrame, fout):
    def no_bold(x):
        b = x.bold
        x.bold = False
        ret = repr(x)
        x.bold = b
        return ret

    def wd(k, v):
        if isinstance(v, Number):
            v = no_bold(v)
        print(rf'\newcommand{{\{k}}}{{{v}}}', file=fout)

    trafs, gd, bundle = (
        BenchmarkSummarizer.METHOD_DISP_NAMES[i]
        for i in ['trafs', 'gd', 'bundle'])

    col_rate = df.columns[5]

    for i, s in enumerate(summarizer.trafs_speedup):
        wd(f'trasSpeedup{chr(ord("A")+i)}', f'{s:.1f}')

    wd('trasSolveRateCmp', '{:.1f}'.format(
        df.loc[('All', trafs), col_rate].val /
        df.loc[('All', bundle), col_rate].val
    ))
    wd('trasSolveRate', df.loc[('All', trafs), col_rate])
    wd('gdSolveRate', df.loc[('All', gd), col_rate])
    wd('bundleSolveRate', df.loc[('All', bundle), col_rate])

def plot_result(df: pd.DataFrame, eps_i, outfile: str):
    meth2id = [(v, i) for i, v in
               enumerate(BenchmarkSummarizer.METHOD_DISP_NAMES.values())]

    plot_x = []
    plot_y = []
    plot_meth = []

    nan_bars_idx = []
    bar_group_width = 0.9
    bar_width_with_sep = bar_group_width / len(meth2id)

    for prob_i, prob in enumerate(ProbMakers.all_makers().keys()):
        for meth, meth_i in meth2id:
            try:
                data = df.loc[(prob, meth)]
            except KeyError:
                continue
            val = data.iloc[eps_i * 3].val
            plot_x.append(prob_i + meth_i * bar_width_with_sep)
            plot_meth.append(meth_i)
            if val is None:
                nan_bars_idx.append(len(plot_y))
                plot_y.append(0)
            else:
                plot_y.append(val)

    fig, ax = plt.subplots(figsize=(9 * 0.7, 6 * 0.7))
    color_palette = ['#E63946', '#457B9D', '#F4A261',
                     '#2A9D8F', '#A8DADC',
                     '#1D3557']

    plot_x = np.array(plot_x)
    plot_y = np.array(plot_y)
    nan_bars_idx = np.array(nan_bars_idx)
    plot_meth = np.array(plot_meth)
    plot_y[nan_bars_idx] = plot_y.max() * 2
    label_map = {
        r'\tras{}': 'TRAFS (ours)',
        r'\SAtwo{}': r'$\text{SA}_2$',
    }
    mpl.rcParams['hatch.linewidth'] = .6
    hatch_color = '#F1FAEE'
    hatches = np.array([''] * len(plot_x), dtype=object)
    hatches[nan_bars_idx] = 'xx'
    for i in range(len(meth2id)):
        mask = plot_meth == i
        ax.bar(
            plot_x[mask], plot_y[mask], width=bar_width_with_sep * .9,
            color=color_palette[i],
            hatch=hatches[mask],
            edgecolor=hatch_color, linewidth=0,
        )
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xticks(np.arange(len(ProbMakers.all_makers())) +
                  bar_group_width / 2)
    ax.set_xticklabels(ProbMakers.all_makers().keys())
    ax.set_xlabel('Problem class (50 instances in each class)')
    eps_exp = [-3, -6]
    assert BenchmarkSummarizer.eps_split[eps_i] == 10**eps_exp[eps_i]
    ax.set_ylabel(r'Normalized mean solving time to reach $\epsilon \leq 10^{'
                  + str(eps_exp[eps_i]) + '}$')
    handles = [
        mpatches.Patch(
            color=color_palette[i], label=label_map.get(label, label))
        for label, i in meth2id
    ]
    handles.append(mpatches.Patch(
        facecolor='white', hatch='xxx', label='Failed to solve',
        edgecolor='black', linewidth=0
    ))
    ax.legend(loc='upper right', fancybox=True, framealpha=0.5,
              handles=handles)
    ax.set_yscale('log')
    ax.set_ylim(0.2, plot_y.max())
    fig.tight_layout()
    fig.savefig(outfile, metadata={'CreationDate': None},
                dpi=300, bbox_inches='tight')

def write_to_latex(summarizer: BenchmarkSummarizer,
                   df: pd.DataFrame, outfile: str):
    latex = df.style.to_latex(
        hrules=True, clines='skip-last;data',
        multicol_align='c', column_format='ll' + 'r'*df.shape[1],
    )
    latex = (LatexTableProc(latex)
             .remove_table_last_cline()
             .fix_cline()
             .add_multirow_header(2, 'Problem', 'Method')
             .as_str())

    outp = Path(outfile)
    with outp.open('w') as fout:
        fout.write(latex)

    with outp.with_stem(outp.stem + '-defs').open('w') as fout:
        write_defs(summarizer, df, fout)

def main():
    parser = argparse.ArgumentParser(
        description='Summarize benchmark results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bench-dir', default='bench-out',
                        help='directory to load benchmark results')
    parser.add_argument('-t', '--latex', help='output to latex file')
    parser.add_argument('-p', '--plot',
                        help='plot the table to files; two files are generated '
                        'with the given prefix')
    parser.add_argument('--plot-ft', help='file type for plot', default='pdf')
    parser.add_argument('--print', action='store_true',
                        help='print the table to stdout')
    args = parser.parse_args()

    summarizer = BenchmarkSummarizer(Path(args.bench_dir))
    df = summarizer.make_bench_table()
    if args.print:
        print(df)

    if args.plot:
        for i in range(len(summarizer.eps_split)):
            plot_result(df, i, f'{args.plot}-{i}.{args.plot_ft}')

    if args.latex:
        write_to_latex(summarizer, df, args.latex)

if __name__ == '__main__':
    main()
