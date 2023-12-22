# cython: language_level=3

import numpy as np
from scipy.sparse import csc_matrix

cimport numpy as np
cimport cython

import typing

ctypedef np.float64_t f64_t

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
        self.data[self.nr_data] = val
        self.row_ind[self.nr_data] = row
        self.col_ind[self.nr_data] = col
        self.nr_data += 1
        assert self.nr_data <= self.data_buf.shape[0]

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
