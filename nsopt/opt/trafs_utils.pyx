# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t dtype_t

cdef class RotationBuffer:
    cdef int size, pos
    cdef np.ndarray arr_buf
    cdef dtype_t* arr

    def __init__(self, int size):
        self.size = size
        assert size >= 2
        self.arr_buf = np.empty(size, dtype=np.float64)
        self.arr = <dtype_t*>self.arr_buf.data
        self.reset()

    def put(self, dtype_t val):
        self.arr[self.pos] = val
        self.pos = (self.pos + 1) % self.size

    def max(self):
        return self.arr_buf.max()

    def reset(self):
        self.pos = 0
        self.arr_buf.fill(-1)

    def __imul__(self, dtype_t val):
        self.arr_buf *= val
        return self
