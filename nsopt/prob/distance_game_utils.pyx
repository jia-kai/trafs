# cython: language_level=3

import numpy as np

cimport numpy as np
cimport cython

ctypedef np.float64_t f64_t

cdef f64_t sign(f64_t x) noexcept:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

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
