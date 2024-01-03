# cython: language_level=3

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

import typing

cimport numpy as np
cimport cython

ctypedef np.float64_t f64_t
ctypedef np.int32_t i32_t

cdef f64_t sign(f64_t x) noexcept:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

cdef class SparseMatMaker:
    cdef np.ndarray data_buf, row_ind_buf, col_ind_buf
    cdef f64_t[:] data
    cdef np.int32_t[:] row_ind, col_ind
    cdef int nr_data

    def __init__(self, max_nr_data):
        self.data_buf = np.empty(max_nr_data, dtype=np.float64)
        self.row_ind_buf = np.empty(max_nr_data, dtype=np.int32)
        self.col_ind_buf = np.empty(max_nr_data, dtype=np.int32)
        self.data = self.data_buf
        self.row_ind = self.row_ind_buf
        self.col_ind = self.col_ind_buf
        self.nr_data = 0

    cdef add(self, np.int32_t row, np.int32_t col, f64_t val) noexcept:
        assert self.nr_data < self.data_buf.shape[0]
        self.data[self.nr_data] = val
        self.row_ind[self.nr_data] = row
        self.col_ind[self.nr_data] = col
        self.nr_data += 1

    def get(self, nrow, ncol) -> csc_matrix:
        return csc_matrix(
            (self.data_buf[:self.nr_data],
             (self.row_ind_buf[:self.nr_data], self.col_ind_buf[:self.nr_data])),
            shape=(nrow, ncol),
            dtype=np.float64,
        )


@cython.boundscheck(False)
@cython.wraparound(False)
def max_of_abs_subd(
    f64_t slack, f64_t fval,
    f64_t x0, np.ndarray[f64_t, ndim=1] abs1, np.ndarray[f64_t, ndim=1] abs1_inp,
) -> typing.Optional[csc_matrix]:
    """compute the vertices for the subdifferential convex hull of the max of
        abs function
    :return: vertex matrix, shape (n, num_vtx); None if origin is included in
        the convex hull
    """
    cdef f64_t slack0, abs0 = abs(x0)
    cdef f64_t d0 = fval - abs0
    cdef int n = abs1.shape[0] + 1

    cdef SparseMatMaker ret_G
    cdef int nr_col = 0

    ret_G = SparseMatMaker(n * 2)

    if d0 <= slack:
        slack0 = slack - d0
        if abs0 <= slack0 / 2:
            return

        ret_G.add(0, nr_col, sign(x0))
        nr_col += 1

    cdef int i
    cdef f64_t di, slacki, gi, gprev
    for i in range(1, n):
        di = fval - abs1[i - 1]
        if di <= slack:
            slacki = slack - di
            if abs1[i - 1] <= slacki / 2:
                return
            gi = sign(abs1_inp[i - 1])
            ret_G.add(i, nr_col, gi)
            ret_G.add(i - 1, nr_col, gi * (-2))
            nr_col += 1

    return ret_G.get(n, nr_col)

@cython.boundscheck(False)
@cython.wraparound(False)
def mxhilb_comp_batch(np.ndarray[f64_t, ndim=1] recips,
                      np.ndarray[f64_t, ndim=2] x):
    """compute the components of the MXHILB function for a batch of inputs
    :return: (dim, batch) array; the function value is `abs(ret).max(axis=0)``
    """
    cdef int batch, dim
    batch = x.shape[0]
    dim = x.shape[1]

    cdef np.ndarray[f64_t, ndim=2] comp = np.empty(
            (dim, batch), dtype=np.float64)

    cdef int i
    xT = x.T
    for i in range(dim):
        np.matmul(recips[i:i+dim], xT, out=comp[i])

    return comp


@cython.boundscheck(False)
@cython.wraparound(False)
def mxhilb_subd(
        f64_t slack, f64_t fval, np.ndarray[f64_t, ndim=1] recips,
        np.ndarray[f64_t, ndim=1] comp) -> typing.Optional[npt.NDArray]:
    """compute the functional subdifferential vertices of the MXHILB function;
    return None if the origin is included in the subdifferential"""
    cdef int dim = comp.shape[0]
    cdef list cols = []
    cdef int i
    cdef f64_t fi, di, epsi

    for i in range(dim):
        fi = abs(comp[i])
        di = fval - fi
        if di <= slack:
            epsi = slack - di
            if fi <= epsi / 2:
                return None

            c = recips[i:i+dim]
            if comp[i] < 0:
                c = -c
            cols.append(c)

    return np.ascontiguousarray(np.array(cols).T)


@cython.boundscheck(False)
@cython.wraparound(False)
def chained_lq_subd(
        f64_t slack,
        np.ndarray[f64_t, ndim=1] g_base,
        np.ndarray[f64_t, ndim=1] x,
        np.ndarray[f64_t, ndim=1] comp) -> csc_matrix:

    assert x.shape[0] == comp.shape[0] + 1
    cdef np.ndarray[long, ndim=1] sidx = np.argsort(np.abs(comp))
    cdef SparseMatMaker ret = SparseMatMaker(x.shape[0] * 2)
    cdef f64_t comp_i_abs
    cdef int i, p, ncol = 0

    for i in range(comp.shape[0]):
        p = sidx[i]
        if slack >= 0:
            comp_i_abs = abs(comp[p])
            if slack >= comp_i_abs:
                slack -= comp_i_abs
                ret.add(p, ncol, 2 * x[p])
                ret.add(p + 1, ncol, 2 * x[p + 1])
                ncol += 1
                continue
            slack = -1
        if comp[p] > 0:
            g_base[p] += 2 * x[p]
            g_base[p + 1] += 2 * x[p + 1]

    return ret.get(x.shape[0], ncol)
