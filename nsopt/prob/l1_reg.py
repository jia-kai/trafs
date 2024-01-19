from ..opt.shared import (
    ProximalGradOptimizable, KnownLipschitzOptimizable, LipschitzConstants)
from .utils import make_stable_rng, UnconstrainedFuncSubDiffHelper
from ..utils import setup_pyx_import

import attr
import numpy as np
import numpy.typing as npt
import scipy.special as sps

import typing

with setup_pyx_import():
    from .kernels import l1_reg_subd

class L1RegularizedOptimizable(ProximalGradOptimizable):
    """min f(x) + lam ||x||_1 where f is smooth"""

    lam: float

    @attr.frozen
    class SubDiff(ProximalGradOptimizable.SubDiff):
        _helper = UnconstrainedFuncSubDiffHelper()

        x0: npt.NDArray
        """the point where the subgradient is evaluated"""

        g0: npt.NDArray
        """the gradient of f at x0"""

        pen: npt.NDArray
        """the penalty term at x0 (i.e., lam * abs(x0))"""

        lam: float
        """the regularization parameter"""

        def reduce_trafs(
                self,
                subg_slack: float, df_lb_thresh: float, norm_bound: float,
                state: dict):
            gc = l1_reg_subd(subg_slack, self.lam, self.g0, self.x0, self.pen)
            return self._helper.reduce_with_min_grad(
                gc, df_lb_thresh, norm_bound)

        def take_arbitrary(self):
            return self.g0 + np.sign(self.x0) * self.lam

    def __init__(self, lam: float):
        self.lam = float(lam)

    def eval(self, x: npt.NDArray, *, need_grad: bool=False) -> typing.Union[
            float, tuple[float, "LassoRegression.SubDiff"]]:
        pen = self.lam * np.abs(x)
        if not need_grad:
            return self.prox_f(x, need_grad=False) + pen.sum()
        else:
            f, grad = self.prox_f(x, need_grad=True)
            return f + pen.sum(), self.SubDiff(x, grad, pen, self.lam)

    def eval_batch(self, x: npt.NDArray) -> npt.NDArray:
        pen = self.lam * np.linalg.norm(x, axis=1, ord=1)
        return self.prox_f_batch(x) + pen

    def prox_g(self, x: npt.NDArray):
        return self.lam * np.linalg.norm(x, ord=1)

    def prox_minx(self, y: npt.NDArray, L: float) -> npt.NDArray:
        t = self.lam / L
        return np.sign(y) * np.maximum(np.abs(y) - t, 0.)


class LassoRegression(L1RegularizedOptimizable, KnownLipschitzOptimizable):
    """min 1/2m ||Ax - b||^2 + lam ||x||_1"""
    A: npt.NDArray
    b: npt.NDArray

    def __init__(self, A: npt.NDArray, b: npt.NDArray, lam: float, x0=None):
        super().__init__(lam)
        assert A.ndim == 2, A.shape
        m, n = A.shape
        assert b.shape == (m, ), b.shape
        if x0 is None:
            x0 = np.zeros(n)
        else:
            assert x0.shape == (n,), x0.shape
        self.A = A
        self.b = b
        self.x0 = x0

    def prox_f(self, x: npt.NDArray, *, need_grad: bool = False):
        A = self.A
        m, _ = A.shape
        Ax = A @ x
        res = Ax - self.b
        f = (1 / (m*2)) * np.dot(res, res)
        if not need_grad:
            return f
        else:
            return f, (1 / m) * (A.T @ res)

    def prox_f_batch(self, x: npt.NDArray) -> npt.NDArray:
        assert x.ndim == 2
        A = self.A
        m, _ = A.shape
        AxT = x @ A.T  # (batch, m)
        res = AxT - self.b[np.newaxis]
        res2 = np.square(res, out=res)
        return (1 / (m*2)) * np.sum(res2, axis=1)

    def eval_cvx_params(self) -> LipschitzConstants:
        A = self.A
        m, n = A.shape
        R = np.sqrt(n)  # a rough estimation
        eig = np.linalg.eigvalsh(A.T @ A) / m
        assert eig[0] <= eig[-1]
        L = eig[-1] * R
        alpha = np.abs(eig[0])
        beta = eig[-1]
        return LipschitzConstants(float(self.eval(self.x0)), R, L, alpha, beta)

    def __repr__(self):
        return (f'L1Reg(m={self.A.shape[0]},'
                f' n={self.A.shape[1]}, lam={self.lam:.2g})')

    @classmethod
    def gen_random(cls, m: int, n: int, lam: float,
                   sparsity=0.95, noise=0.05,
                   rng: typing.Optional[np.random.Generator] = None
                   ) -> tuple["LassoRegression", npt.NDArray]:
        """
        Generate a random Lasso problem
        :return: (problem, xtrue)
        """
        if rng is None:
            rng = make_stable_rng(cls)
        A = rng.standard_normal((m, n))
        xtrue = rng.standard_normal(n)
        xtrue *= rng.uniform(size=n) >= sparsity
        b = A @ xtrue
        noise_s = noise * np.abs(b).mean()
        b += rng.normal(scale=noise_s, size=m)
        return cls(A, b, lam), xtrue


class LassoClassification(L1RegularizedOptimizable):
    """multi-class classification with L1 regularization and cross-entropy
    loss

    Loss = -1/m (log sum exp (A_i x) - (A_i x)_{y_i}) + lam ||x||_1
    """
    A: npt.NDArray
    b: npt.NDArray
    nr_class: int

    _m_arange: npt.NDArray

    def __init__(self, A: npt.NDArray, b: npt.NDArray, lam: float, x0=None):
        super().__init__(lam)
        assert A.ndim == 2, A.shape
        m, n = A.shape
        assert b.shape == (m, ), b.shape
        assert b.dtype == np.int32, b.dtype
        nr_class = b.max() + 1
        self.nr_class = nr_class
        x_size = n * nr_class
        if x0 is None:
            x0 = np.zeros(x_size)
        else:
            assert x0.shape == (x_size,), (x0.shape, (n, nr_class))
        self.A = A
        self.b = b
        self.x0 = x0
        self._m_arange = np.arange(m)

    def prox_f(self, x: npt.NDArray, *, need_grad: bool = False):
        A = self.A
        m, n = A.shape
        Ax = A @ x.reshape(n, self.nr_class)    # (m, nr_class)
        ce = (1/m) * (sps.logsumexp(Ax, axis=1) -
                       Ax[self._m_arange, self.b]).sum()
        if need_grad:
            # d[i, j, k] = Ax[i, k] - Ax[i, j]
            d = Ax[:, np.newaxis, :] - Ax[:, :, np.newaxis]
            # g0 is diff(loss, Ax)
            g0 = (1 / m) / np.exp(d).sum(axis=2)    # (m, nr_class)
            g0[self._m_arange, self.b] -= 1 / m
            grad = A.T @ g0
            return ce, grad.flatten()
        return ce

    def prox_f_batch(self, x: npt.NDArray) -> npt.NDArray:
        batch, _ = x.shape
        A = self.A
        m, n = A.shape
        x = x.reshape(batch, n, self.nr_class)
        # Ax = np.einsum('mn, bnc -> bmc', A, x) # too slow
        xt = x.transpose((1, 0, 2)).reshape(n, batch * self.nr_class)
        Ax = (A @ xt).reshape(m, batch, self.nr_class)
        Ax_b = Ax[self._m_arange, :, self.b]    # (m, batch)
        ce = (1/m) * (sps.logsumexp(Ax, axis=2) - Ax_b).sum(axis=0)
        return ce

    def __repr__(self):
        return (f'L1Cls(m={self.A.shape[0]},'
                f' n={self.A.shape[1]}, k={self.nr_class}, lam={self.lam:.2g})')

    @classmethod
    def gen_random(cls, m: int, n: int, nr_class: int, lam: float,
                   sparsity=0.95, noise=0.05,
                   rng: typing.Optional[np.random.Generator] = None
                   ) -> tuple["LassoClassification", npt.NDArray]:
        """:return: (problem, xtrue)"""
        if rng is None:
            rng = make_stable_rng(cls)
        xtrue = np.empty((n, nr_class), dtype=np.float64)
        xtrue[:, 0] = rng.standard_normal(n) * (rng.uniform(size=n) < sparsity)
        min_angle_cos = np.cos(np.pi / nr_class)
        for i in range(1, nr_class):
            # ensure that they have enough pairwise distance
            while True:
                xi = rng.standard_normal(n) * (rng.uniform(size=n) < sparsity)
                len_xi = np.linalg.norm(xi)
                for j in range(i):
                    len_xj = np.linalg.norm(xtrue[:, j])
                    if np.abs(np.dot(xi, xtrue[:, j])) > (
                            min_angle_cos * len_xi * len_xj):
                        break
                else:
                    xtrue[:, i] = xi
                    break

        coeff = rng.exponential(scale=1, size=(m, nr_class))
        b = rng.integers(nr_class, size=m, dtype=np.int32)
        coeff[np.arange(m), b] = 0
        coeff *= 0.1 / coeff.sum(axis=1, keepdims=True)
        coeff[np.arange(m), b] = 1
        A = coeff @ xtrue.T
        noise_s = noise * np.abs(A).mean()
        A += rng.normal(scale=noise_s, size=A.shape)
        return cls(A, b, lam), xtrue
