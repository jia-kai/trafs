from ..opt.shared import Optimizable

import numpy as np
import numpy.typing as npt

def projection_simplex(v: npt.NDArray, z: float=1) -> npt.NDArray:
    """projection of v onto the simplex"""
    # see https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    # see also https://arxiv.org/pdf/1309.1541.pdf
    assert v.ndim == 1
    n_features = v.shape[0]
    u = np.ascontiguousarray(np.sort(v)[::-1])
    cssv = np.cumsum(u) - z
    ind = np.arange(1, n_features + 1, dtype=v.dtype) 
    cond = u * ind > cssv 
    rho = np.count_nonzero(cond)
    assert rho > 0 and cond[rho - 1]
    lam = cssv[rho - 1] / float(rho)
    w = v - lam
    w = np.maximum(w, 0, out=w)
    return w

class SimplexConstrainedOptimizable(Optimizable):
    def proj(self, x: npt.NDArray) -> npt.NDArray:
        return projection_simplex(x, 1.0)
