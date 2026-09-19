import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

from .utils import SumOfCvxHullDesc

type FloatArray = npt.NDArray[np.float64]
type Int32Array = npt.NDArray[np.int32]
type UInt8Array = npt.NDArray[np.uint8]

def sum_of_max_subd_mask(slack: float, comp: FloatArray) -> UInt8Array: ...
def l1_reg_subd(
    slack: float, lam: float, g0: FloatArray, x: FloatArray, pen: FloatArray
) -> FloatArray: ...
def max_of_abs_subd(
    slack: float,
    fval: float,
    x0: float,
    abs1: FloatArray,
    abs1_inp: FloatArray,
) -> csc_matrix | None: ...
def mxhilb_comp_batch(recips: FloatArray, x: FloatArray) -> FloatArray: ...
def mxhilb_subd(
    slack: float, fval: float, recips: FloatArray, comp: FloatArray
) -> FloatArray | None: ...
def chained_lq_subd(
    slack: float, x: FloatArray, comp: FloatArray
) -> SumOfCvxHullDesc: ...
def chained_cb3_I_subd(
    slack: float, x: FloatArray, comp: FloatArray
) -> SumOfCvxHullDesc: ...
def distance_game_subd(
    comp_slack: FloatArray,
    g_kl: FloatArray,
    A: FloatArray,
    Ax: FloatArray,
    B: FloatArray,
    BtBx: FloatArray,
    Bx_norm: FloatArray,
) -> tuple[FloatArray, tuple[FloatArray, FloatArray]]: ...
def reduce_multi_cvx_hull_max_sum(
    nr_hull: int, vtx_values: FloatArray, hull_ids: Int32Array
) -> float: ...
