# cython: language_level=3

import numpy as np
import numpy.typing as npt

import typing

cimport numpy as np
cimport cython

ctypedef np.float64_t f64_t
ctypedef np.int32_t i32_t

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
