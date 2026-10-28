# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
import h5py
import numpy as np
import torch
from professor import utils
from professor.torch_models import Generator
from professor.layers import AlphaLinear

try:  # the trainer needs mpi4py and an MPI library
    from professor.mltrainer import CompleteDatasetOneFileSims, CompleteDatasetDivideScaling, sensitivity_rms
except Exception as e:  # pragma: no cover
    MLTRAINER_IMPORT_ERROR = repr(e)
else:
    MLTRAINER_IMPORT_ERROR = ""


class TestForwardJacobianVectorProduct(unittest.TestCase):

    def test_matches_finite_differences(self):
        """a Generator's Jacobian-vector product matches a central difference; the primal is the plain output"""
        torch.manual_seed(0)
        model = Generator(5, 32, 3, min_features=4, max_features=16, last_layer=AlphaLinear(3), act_fun="SiLU")
        model = model.double().eval()
        x = torch.randn(2, 5, 1, 1, dtype=torch.float64)
        t = torch.randn_like(x)
        pred, d_pred = utils.forward_jacobian_vector_product(model, x, t)
        h = 1e-6
        fd = (model(x + h * t) - model(x - h * t)) / (2 * h)
        self.assertTrue(torch.allclose(pred, model(x)))
        self.assertTrue(torch.allclose(d_pred, fd, rtol=1e-5, atol=1e-8))

    def test_backward_through_jvp(self):
        """the derivative loss has gradients with respect to the weights"""
        torch.manual_seed(0)
        model = Generator(3, 16, 2, min_features=4, max_features=8, last_layer=AlphaLinear(2))
        x = torch.randn(4, 3, 1, 1)
        _, d_pred = utils.forward_jacobian_vector_product(model, x, torch.randn_like(x))
        torch.mean(d_pred**2).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))


class TestSobolevLoss(unittest.TestCase):

    torch.manual_seed(1)
    sensitivities = torch.randn(3, 2, 2, 8, 4)
    u = torch.randn(3, 2)
    exact = torch.einsum("nk,nkcyx->ncyx", u, sensitivities)

    def test_exact_derivative(self):
        self.assertEqual(utils.sobolev_loss(self.exact, self.sensitivities, self.u).item(), 0.0)

    def test_value(self):
        """mean squared error of the directional derivative, divided by the number of inputs"""
        d = torch.randn_like(self.exact)
        ref = torch.mean((self.exact + d - self.exact) ** 2) / 2
        self.assertTrue(torch.allclose(utils.sobolev_loss(self.exact + d, self.sensitivities, self.u), ref))

    def test_non_finite_samples_skipped(self):
        d = torch.randn_like(self.exact)
        sensitivities = self.sensitivities.clone()
        sensitivities[1, 0, 1, 2, 3] = float("nan")
        sensitivities[2, 1, 0, 0, 0] = float("inf")
        loss = utils.sobolev_loss(self.exact + d, sensitivities, self.u)
        self.assertTrue(torch.allclose(loss, torch.mean(d[0] ** 2) / 2))
        all_bad = torch.full_like(sensitivities, float("nan"))
        self.assertEqual(utils.sobolev_loss(d, all_bad, self.u).item(), 0.0)

    def test_3d_fields(self):
        """3D sensitivities (n, k, channels, y, x, z) work the same way"""
        sensitivities = torch.randn(2, 3, 1, 4, 4, 2)
        u = torch.randn(2, 3)
        exact = torch.einsum("nk,nkcyxz->ncyxz", u, sensitivities)
        d = torch.randn_like(exact)
        self.assertEqual(utils.sobolev_loss(exact, sensitivities, u).item(), 0.0)
        self.assertTrue(torch.allclose(utils.sobolev_loss(exact + d, sensitivities, u), torch.mean(d**2) / 3))


class FakeComm:
    def allreduce(self, x, op=None):
        return x


@unittest.skipIf(MLTRAINER_IMPORT_ERROR, f"professor.mltrainer: {MLTRAINER_IMPORT_ERROR}")
class TestSensitivityDatasets(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.inputs = rng.normal(size=(4, 3)).astype(np.float32)
        self.fields = rng.normal(size=(4, 2, 8, 4)).astype(np.float32)
        self.jac = rng.normal(size=(4, 2, 2, 8, 4)).astype(np.float32)
        self.jac[3, 0, 0, 0, 0] = np.nan
        with h5py.File(os.path.join(self.tmp.name, "sim.h5"), "w") as f:
            f["inputs"] = self.inputs
            f["fields"] = self.fields
            f["jacobian"] = self.jac
        self.filelist = np.array([("sim.h5", i) for i in range(4)], dtype=[("f", "U16"), ("i", int)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_key(self):
        ds = CompleteDatasetOneFileSims(self.filelist, 2, self.tmp.name)
        self.assertEqual(len(ds[1]), 2)
        self.assertEqual((ds.n_input, ds.pixels_y, ds.pixels_x), (3, 8, 4))

    def test_one_file_sims(self):
        ds = CompleteDatasetOneFileSims(self.filelist, 2, self.tmp.name, sensitivity_key="jacobian")
        x, y, j = ds[1]
        self.assertEqual(x.shape, (3, 1, 1))
        self.assertTrue(np.array_equal(j.numpy(), self.jac[1]))

    def test_divide_scaling(self):
        """inputs divided by the scaling make the sensitivities multiplied by it"""
        with h5py.File(os.path.join(self.tmp.name, "one.h5"), "w") as f:
            f["inputs"] = self.inputs[1]
            f["fields"] = self.fields[1]
            f["jacobian"] = self.jac[1]
        scaling = torch.tensor([2.0, 4.0, 8.0]).view(-1, 1, 1)
        ds = CompleteDatasetDivideScaling(np.array(["one.h5"]), 2, self.tmp.name, scaling, sensitivity_key="jacobian")
        x, y, j = ds[0]
        self.assertTrue(np.allclose(x.numpy().ravel(), self.inputs[1] / [2, 4, 8]))
        self.assertTrue(np.allclose(j.numpy(), self.jac[1] * np.array([2.0, 4.0]).reshape(2, 1, 1, 1)))

    def test_sensitivity_rms(self):
        """per-input rms over the finite samples"""
        ds = CompleteDatasetOneFileSims(self.filelist, 2, self.tmp.name, sensitivity_key="jacobian")
        rms = sensitivity_rms(ds, np.arange(4), FakeComm())
        ref = np.sqrt(np.mean(self.jac[:3].astype(np.float64) ** 2, axis=(0, 2, 3, 4)))
        self.assertTrue(np.allclose(rms, ref))


if __name__ == "__main__":
    unittest.main()
