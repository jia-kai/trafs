# cython: language_level=3

from .utils import SumOfCvxHullDesc

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, coo_matrix

import typing

cimport numpy as np
cimport cython

ctypedef np.float64_t f64_t
ctypedef np.int32_t i32_t

cdef extern from "kernels.h":
    f64_t sign(f64_t x) noexcept

    cdef cppclass cn2 "2":
        pass
    cdef cppclass cn3 "3":
        pass

    void compute_sum_of_max_subd_mask_impl "compute_sum_of_max_subd_mask" [N] (
        f64_t tot_slack, np.uint8_t* mask, const f64_t* comp,
        cython.size_t comp_size) noexcept


cdef class SparseMatMaker:
    cdef np.ndarray data_buf, row_ind_buf, col_ind_buf
    cdef f64_t[:] data
    cdef np.int32_t[:] row_ind, col_ind
    cdef int nr_data, have_duplicate

    def __init__(self, max_nr_data, have_duplicate=False):
        """
        :param max_nr_data: the maximum number of data entries
        :param have_duplicate: whether the matrix may have duplicate entries
        """
        self.data_buf = np.empty(max_nr_data, dtype=np.float64)
        self.row_ind_buf = np.empty(max_nr_data, dtype=np.int32)
        self.col_ind_buf = np.empty(max_nr_data, dtype=np.int32)
        self.data = self.data_buf
        self.row_ind = self.row_ind_buf
        self.col_ind = self.col_ind_buf
        self.nr_data = 0
        self.have_duplicate = have_duplicate

    cdef add(self, np.int32_t row, np.int32_t col, f64_t val):
        assert self.nr_data < self.data_buf.shape[0]
        self.data[self.nr_data] = val
        self.row_ind[self.nr_data] = row
        self.col_ind[self.nr_data] = col
        self.nr_data += 1

    cdef get(self, nrow, ncol):
        if self.have_duplicate:
            maker = coo_matrix
        else:
            maker = csc_matrix

        mat = maker(
            (self.data_buf[:self.nr_data], (self.row_ind_buf[:self.nr_data],
                                            self.col_ind_buf[:self.nr_data])),
            shape=(nrow, ncol),
            dtype=np.float64
        )
        if self.have_duplicate:
            return mat.tocsc()
        else:
            return mat

def sum_of_max_subd_mask(
        f64_t slack, np.ndarray[f64_t, ndim=2] comp) -> npt.NDArray:
    """compute the mask of the activated components of the sum of max function,
        defined as ``f = comp.max(axis=1).sum()`.
    :param slack: the slack of the subdifferential
    :return: the mask for the subdifferential
    """
    comp = np.ascontiguousarray(comp)
    cdef np.ndarray[np.uint8_t, ndim=1] mask = np.empty(
            comp.shape[0], dtype=np.uint8)
    if comp.shape[1] == 2:
        compute_sum_of_max_subd_mask_impl[cn2](
            slack, &mask[0], &comp[0, 0], comp.shape[0])
    elif comp.shape[1] == 3:
        compute_sum_of_max_subd_mask_impl[cn3](
            slack, &mask[0], &comp[0, 0], comp.shape[0])
    else:
        raise NotImplementedError(f'unsupported comp num: {comp.shape[1]}')
    return mask


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

cdef class SumOfMaxSparse2Builder:
    """convex hull desc builder for sum of max functions with sparse gradients
    that have at most 2 components
    """

    cdef np.ndarray bias_npy, ui_npy
    cdef f64_t* bias
    cdef i32_t* ui
    cdef SparseMatMaker G

    cdef int have_other_comp, aux_used, nr_col, nr_aux, dim, max_nr_vtx
    cdef int other_idx0, other_idx1
    cdef f64_t other_val0, other_val1

    def __init__(self, int dim, int sum_size, int max_size):
        cdef np.ndarray[f64_t, ndim=1] bias_npy
        cdef np.ndarray[i32_t, ndim=1] ui_npy
        cdef int max_nr_vtx = sum_size * max_size

        bias_npy = np.zeros(dim, dtype=np.float64)
        ui_npy = np.empty(max_nr_vtx, dtype=np.int32)

        self.bias_npy = bias_npy
        self.ui_npy = ui_npy
        self.bias = &bias_npy[0]
        self.ui = &ui_npy[0]
        self.G = SparseMatMaker(sum_size * max_size * max_size * 2,
                                have_duplicate=True)
        self.have_other_comp = 0
        self.aux_used = 0
        self.nr_col = 0
        self.nr_aux = 0
        self.dim = dim
        self.max_nr_vtx = max_nr_vtx

    cdef add(self, int idx0, f64_t val0, int idx1, f64_t val1):
        cdef int c
        if self.have_other_comp:
            c = self.nr_col
            self.nr_col += 1
            assert c < self.max_nr_vtx
            self.G.add(self.other_idx0, c, -self.other_val0)
            self.G.add(self.other_idx1, c, -self.other_val1)
            self.G.add(idx0, c, val0)
            self.G.add(idx1, c, val1)
            self.ui[c] = self.nr_aux
            self.aux_used = 1
        else:
            self.bias[idx0] += val0
            self.bias[idx1] += val1
            self.have_other_comp = 1
            self.other_idx0 = idx0
            self.other_idx1 = idx1
            self.other_val0 = val0
            self.other_val1 = val1

    cdef commit_summand(self):
        if self.have_other_comp:
            self.have_other_comp = 0

        if self.aux_used:
            self.nr_aux += 1
            self.aux_used = 0

    cdef get(self):
        return SumOfCvxHullDesc(
            bias=self.bias_npy,
            G=self.G.get(self.dim, self.nr_col),
            ui=self.ui_npy[:self.nr_col],
            nr_hull=self.nr_aux,
        )


@cython.boundscheck(False)
@cython.wraparound(False)
def chained_lq_subd(
        f64_t slack,
        np.ndarray[f64_t, ndim=1] x,
        np.ndarray[f64_t, ndim=2] comp) -> SumOfCvxHullDesc:

    cdef np.ndarray[np.uint8_t, ndim=1] mask = sum_of_max_subd_mask(slack, comp)
    cdef SumOfMaxSparse2Builder builder = SumOfMaxSparse2Builder(
            x.shape[0], comp.shape[0], 2)
    cdef int i
    for i in range(comp.shape[0]):
        if mask[i] & 1:
            builder.add(i, -1, i + 1, -1)
        if mask[i] & 2:
            builder.add(i, -1 + 2 * x[i], i + 1, -1 + 2 * x[i + 1])
        builder.commit_summand()
    return builder.get()


@cython.boundscheck(False)
@cython.wraparound(False)
def chained_cb3_I_subd(
        f64_t slack,
        np.ndarray[f64_t, ndim=1] x,
        np.ndarray[f64_t, ndim=2] comp) -> SumOfCvxHullDesc:

    cdef np.ndarray[np.uint8_t, ndim=1] mask = sum_of_max_subd_mask(slack, comp)
    cdef SumOfMaxSparse2Builder builder = SumOfMaxSparse2Builder(
            x.shape[0], comp.shape[0], 3)
    cdef f64_t xi0, xi1
    cdef int i, mi
    for i in range(comp.shape[0]):
        xi0 = x[i]
        xi1 = x[i + 1]
        mi = mask[i]
        if mi & 1:
            builder.add(i, 4 * xi0 * xi0 * xi0, i + 1, 2 * xi1)
        if mi & 2:
            builder.add(i, -2 * (2 - xi0), i + 1, -2 * (2 - xi1))
        if mi & 4:
            builder.add(i, -comp[i, 2], i + 1, comp[i, 2])
        builder.commit_summand()
    return builder.get()


@cython.wraparound(False)
def distance_game_subd(
    np.ndarray[f64_t, ndim=1] comp_slack,
    np.ndarray[f64_t, ndim=2] g_kl,
    np.ndarray[f64_t, ndim=2] A,
    np.ndarray[f64_t, ndim=1] Ax,
    np.ndarray[f64_t, ndim=3] B,
    np.ndarray[f64_t, ndim=2] BtBx,
    np.ndarray[f64_t, ndim=1] Bx_norm) -> tuple[
        npt.NDArray, tuple[npt.NDArray, npt.NDArray]]:

    cdef int i, ncomp = comp_slack.shape[0], xdim = A.shape[1]
    assert g_kl.shape[0] == ncomp and g_kl.shape[1] == xdim
    assert A.shape[0] == ncomp and A.shape[1] == xdim
    assert Ax.shape[0] == ncomp
    assert B.shape[0] == ncomp and B.shape[2] == xdim
    assert BtBx.shape[0] == ncomp and BtBx.shape[1] == xdim
    assert Bx_norm.shape[0] == ncomp

    cdef f64_t slack
    cdef np.ndarray[f64_t, ndim=2] g0, g1
    cdef np.ndarray[f64_t, ndim=3] h1
    cdef np.ndarray[f64_t, ndim=1] tmp
    g0 = np.empty((ncomp * 2, xdim), dtype=np.float64)
    g1 = np.empty((ncomp * 2, xdim), dtype=np.float64)
    h1 = np.empty((ncomp * 2, B.shape[1], xdim), dtype=np.float64)

    cdef int g0p = 0
    cdef int g1p = 0

    for i in range(len(comp_slack)):
        slack = comp_slack[i]
        if Bx_norm[i] < slack / 2:
            if abs(Ax[i]) + Bx_norm[i] < slack / 2:
                g1[g1p] = g_kl[i] - A[i]
                g1[g1p + 1] = g_kl[i] + A[i]
                h1[g1p] = h1[g1p + 1] = B[i]
                g1p += 2
            else:
                g1[g1p] = g_kl[i] + A[i] * sign(Ax[i])
                h1[g1p] = B[i]
                g1p += 1
        else:
            tmp = BtBx[i] / max(Bx_norm[i], 1e-10)
            tmp += g_kl[i]
            if abs(Ax[i]) < slack / 2:
                g0[g0p] = tmp - A[i]
                g0[g0p + 1] = tmp + A[i]
                g0p += 2
            else:
                g0[g0p] = tmp + A[i] * sign(Ax[i])
                g0p += 1

    return g0[:g0p], (g1[:g1p], h1[:g1p])


@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_multi_cvx_hull_max_sum(
        int nr_hull,
        np.ndarray[f64_t, ndim=1] vtx_values,
        np.ndarray[i32_t, ndim=1] hull_ids) -> float:
    assert vtx_values.shape[0] == hull_ids.shape[0]
    cdef np.ndarray[f64_t, ndim=1] maxv = np.zeros(nr_hull, dtype=np.float64)
    cdef int i, hid
    for i in range(vtx_values.shape[0]):
        hid = hull_ids[i]
        assert 0 <= hid < nr_hull
        maxv[hid] = max(maxv[hid], vtx_values[i])
    return maxv.sum()
