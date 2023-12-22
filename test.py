import scipy.optimize as spo
import numpy as np

from nsopt.opt.shared import (
    Optimizable, ProximalGradOptimizable, UnconstrainedOptimizable)
from nsopt.prob.utils import make_stable_rng
from nsopt.prob import (LassoRegression, LassoClassification, MaxOfAbs,
                        GeneralizedMXHILB, DistanceGame)

import unittest
import itertools

class TestCaseWithRng(unittest.TestCase):
    def setUp(self):
        self.rng = make_stable_rng(type(self))

class BatchEvalTest(TestCaseWithRng):
    def check(self, opt: Optimizable, batch_size=55):
        x0 = opt.x0
        xb = x0[np.newaxis, :] + self.rng.standard_normal(
            (batch_size, *x0.shape))
        xb[0] = x0
        fvals = []
        for i in range(batch_size):
            xb[i] = opt.proj(xb[i])
            fi0 = opt.eval(xb[i])
            fi1 = opt.eval(xb[i], need_grad=True)[0]
            self.assertEqual(fi0, fi1)
            fvals.append(fi0)
        fb = opt.eval_batch(xb)
        np.testing.assert_allclose(fvals, fb)

    def test_lasso_regression(self):
        lasso, _ = LassoRegression.gen_random(10, 20, .3, rng=self.rng)
        self.check(lasso)

    def test_lasso_classification(self):
        lasso, xt = LassoClassification.gen_random(10, 20, 5, .3, rng=self.rng)
        ce = lasso.prox_f(xt)
        self.assertGreater(ce, 0)
        self.assertLess(ce, 1)
        self.check(lasso)

    def test_max_of_abs(self):
        moa = MaxOfAbs(100)
        self.check(moa)

    def test_generalized_mxhilb(self):
        mxh = GeneralizedMXHILB(100)
        self.check(mxh)

    def test_distance_game(self):
        g = DistanceGame.gen_random(50)
        self.check(g)


class NumericalGradTest(TestCaseWithRng):
    @classmethod
    def check_prox(cls, opt: ProximalGradOptimizable):
        x = opt.x0
        _, grad = opt.prox_f(x, need_grad=True)
        grad_num = spo.approx_fprime(x, opt.prox_f)
        np.testing.assert_allclose(grad, grad_num, atol=1e-5, rtol=1e-5)

    def check_numerical_subdiff(self, opt: Optimizable, x0, *,
                                is_unconstrained=True):
        _, subg = opt.eval(x0, need_grad=True)
        g0 = -subg.reduce_trafs(1e-9, 0, 1, {}).dx
        g1 = subg.take_arbitrary()
        if is_unconstrained:
            self.assertIsInstance(opt, UnconstrainedOptimizable)
            np.testing.assert_allclose(np.linalg.norm(g0), 1, atol=1e-8)
            g0 *= np.linalg.norm(g1)
            np.testing.assert_allclose(g0, g1)
        else:
            self.assertLessEqual(np.linalg.norm(g0), 1 + 1e-5)
        gnum = spo.approx_fprime(x0, opt.eval)
        np.testing.assert_allclose(g1, gnum, atol=1e-5, rtol=1e-5)

    def test_lasso_regression(self):
        lasso, _ = LassoRegression.gen_random(30, 20, .5, rng=self.rng)
        self.check_prox(lasso)

    def test_lasso_classification(self):
        lasso, _ = LassoClassification.gen_random(30, 40, 5, .3, rng=self.rng)
        self.check_prox(lasso)

    def test_max_of_abs(self):
        moa = MaxOfAbs(100)

        x0 = self.rng.standard_normal(moa.x0.shape)
        x0[23] += 5
        self.check_numerical_subdiff(moa, x0)

        x0 = self.rng.standard_normal(moa.x0.shape)
        x0[0] += 5
        self.check_numerical_subdiff(moa, x0)

    def test_generalized_mxhilb(self):
        for n in range(95, 105):
            mxh = GeneralizedMXHILB(n)
            x0 = self.rng.standard_normal(mxh.x0.shape)
            self.check_numerical_subdiff(mxh, x0)

    def test_distance_game(self):
        for n in itertools.chain([5], range(95, 105)):
            g = DistanceGame.gen_random(n)
            x0 = self.rng.uniform(size=n) + .05
            x0 /= x0.sum()
            self.check_numerical_subdiff(g, x0, is_unconstrained=False)


class TestL1SubDiff(TestCaseWithRng):
    n = 10
    slack = 1.2
    lam = 0.7

    def _check_subg(self, pen, idx):
        n = self.n
        lam = self.lam
        g0 = self.rng.standard_normal(n)
        x0 = pen / lam * np.sign(self.rng.standard_normal(n))

        g_low = g0 + lam * np.sign(x0)
        g_high = g_low.copy()
        g_low[idx] = g0[idx] - lam
        g_high[idx] = g0[idx] + lam

        g_low_get, g_high_get = LassoRegression.SubDiff(
            x0, g0, pen, lam)._get_subgrad(self.slack, 100)
        np.testing.assert_allclose(g_low, g_low_get)
        np.testing.assert_allclose(g_high, g_high_get)

    def test_meth1(self):
        n = self.n
        slack = self.slack
        thresh = slack / (2 * np.sqrt(n)) - 1e-4
        idx = [1, 3, 5, 7, 8]
        pen = self.rng.uniform(slack / 2 + 1e-4, slack, n)
        pen[idx] = self.rng.uniform(thresh * 0.99, thresh, len(idx))
        self._check_subg(pen, idx)

    def test_meth2(self):
        n = self.n
        slack = self.slack
        thresh = slack / (2 * np.sqrt(n)) + 1e-4
        idx = [4, 6]
        pen = self.rng.uniform(slack / 2 + 1e-4, slack, n)
        pen[idx] = self.rng.uniform(thresh, thresh * 1.01, len(idx))
        self._check_subg(pen, idx)
