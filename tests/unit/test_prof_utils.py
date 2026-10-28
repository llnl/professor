# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import torch
import torch.nn as nn
from professor import utils


class TestConsolidatedLoss(unittest.TestCase):
    # Create random tensors for testing
    torch.manual_seed(42)  # For reproducibility
    pred = torch.rand(10, 5)
    target = torch.rand(10, 5)

    y = torch.rand(5, 10, 512, 512)
    yhat = torch.rand(5, 10, 512, 512)

    # Standard PyTorch loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    def test_l1_loss(self):
        # Get L1 loss using our consolidated function
        our_losses = utils.consolidated_loss(self.pred, self.target, loss_types=["l1"])
        our_l1 = our_losses["l1"]

        # Get L1 loss using PyTorch's function
        torch_l1 = self.l1_loss(self.pred, self.target)

        # Check if they're the same
        self.assertTrue(torch.allclose(our_l1, torch_l1), f"L1 loss mismatch: Ours = {our_l1}, PyTorch = {torch_l1}")

        # Verify our manual implementation matches PyTorch's
        manual_l1 = torch.mean(torch.abs(self.pred - self.target))
        self.assertTrue(
            torch.allclose(our_l1, manual_l1), f"L1 loss manual check failed: Ours = {our_l1}, Manual = {manual_l1}"
        )

    def test_l1_loss_image(self):
        # Get L1 loss using our consolidated function
        our_losses = utils.consolidated_loss(self.yhat, self.y, loss_types=["l1"])
        our_l1 = our_losses["l1"]

        # Get L1 loss using PyTorch's function
        torch_l1 = self.l1_loss(self.yhat, self.y)

        # Check if they're the same
        self.assertTrue(torch.allclose(our_l1, torch_l1), f"L1 loss mismatch: Ours = {our_l1}, PyTorch = {torch_l1}")

        # Verify our manual implementation matches PyTorch's
        manual_l1 = torch.mean(torch.abs(self.y - self.yhat))
        self.assertTrue(
            torch.allclose(our_l1, manual_l1), f"L1 loss manual check failed: Ours = {our_l1}, Manual = {manual_l1}"
        )

    def test_sum_per_channel(self):
        # Get L1 loss using our consolidated function
        our_losses = utils.consolidated_loss(
            self.yhat,
            self.y,
            loss_types=["sum-per-channel"],
        )
        our_sum_per_channel = our_losses["sum-per-channel"]

        # Verify our manual implementation matches PyTorch's
        diff = self.yhat - self.y
        abs_diff = torch.abs(diff)
        sum_per_channel = torch.sum(abs_diff, (-2, -1)).sum(0)
        self.assertTrue(
            torch.allclose(our_sum_per_channel, sum_per_channel),
            f"sum_per_channel check failed: Ours = {our_sum_per_channel}, Manual = {sum_per_channel}",
        )

    def test_l2_loss(self):
        # Get L2 loss using our consolidated function
        our_losses = utils.consolidated_loss(self.pred, self.target, loss_types=["l2"])
        our_l2 = our_losses["l2"]

        # Get L2 loss using PyTorch's function
        torch_l2 = self.mse_loss(self.pred, self.target)

        # Check if they're the same
        self.assertTrue(torch.allclose(our_l2, torch_l2), f"L2 loss mismatch: Ours = {our_l2}, PyTorch = {torch_l2}")

        # Verify our manual implementation matches PyTorch's
        diff = self.pred - self.target
        manual_l2 = torch.mean(diff * diff)
        self.assertTrue(
            torch.allclose(our_l2, manual_l2), f"L2 loss manual check failed: Ours = {our_l2}, Manual = {manual_l2}"
        )

    def test_l2_loss_images(self):
        # Get L2 loss using our consolidated function
        our_losses = utils.consolidated_loss(self.yhat, self.y, loss_types=["l2"])
        our_l2 = our_losses["l2"]

        # Get L2 loss using PyTorch's function
        torch_l2 = self.mse_loss(self.yhat, self.y)

        # Check if they're the same
        self.assertTrue(torch.allclose(our_l2, torch_l2), f"L2 loss mismatch: Ours = {our_l2}, PyTorch = {torch_l2}")

        # Verify our manual implementation matches PyTorch's
        diff = self.yhat - self.y
        manual_l2 = torch.mean(diff * diff)
        self.assertTrue(
            torch.allclose(our_l2, manual_l2), f"L2 loss manual check failed: Ours = {our_l2}, Manual = {manual_l2}"
        )

    def test_l2_sum(self):
        # Get L2 loss using our consolidated function
        our_losses = utils.consolidated_loss(self.yhat, self.y, loss_types=["l2-sum"])
        our_l2 = our_losses["l2-sum"]

        diff = self.yhat - self.y
        l2_sum = torch.sum(torch.square(diff))

        # Check if they're the same
        self.assertTrue(torch.allclose(our_l2, l2_sum), f"L2 loss mismatch: Ours = {our_l2}, PyTorch = {l2_sum}")

    def test_l1_sum(self):
        # Get L1 loss using our consolidated function
        our_losses = utils.consolidated_loss(self.yhat, self.y, loss_types=["l1-sum"])
        our_l1 = our_losses["l1-sum"]

        diff = self.yhat - self.y
        l1_sum = torch.sum(torch.abs(diff))

        # Check if they're the same
        self.assertTrue(torch.allclose(our_l1, l1_sum), f"L1 loss mismatch: Ours = {our_l1}, PyTorch = {l1_sum}")

    def test_linf_loss(self):
        # Get L-infinity loss using our consolidated function
        our_losses = utils.consolidated_loss(self.pred, self.target, loss_types=["linf"])
        our_linf = our_losses["linf"]

        # Calculate L-infinity loss manually
        manual_linf = torch.max(torch.abs(self.pred - self.target))

        # Check if they're the same
        self.assertTrue(
            torch.allclose(our_linf, manual_linf),
            f"L-infinity loss mismatch: Ours = {our_linf}, Manual = {manual_linf}",
        )

    def test_weighted_sum(self):
        # Define weights
        weights = [0.3, 0.5, 0.2]

        # Get weighted sum of losses using our consolidated function
        weighted_sum = utils.consolidated_loss(
            self.pred,
            self.target,
            weights=weights,
        )
        assert isinstance(weighted_sum, torch.Tensor)

        # Calculate individual losses and weighted sum manually
        individual_losses = utils.consolidated_loss(
            self.pred,
            self.target,
        )

        assert isinstance(individual_losses, dict)
        manual_weighted_sum = (
            weights[0] * individual_losses["l1"]
            + weights[1] * individual_losses["l2"]
            + weights[2] * individual_losses["linf"]
        )

        # Check if they're the same
        self.assertTrue(
            torch.allclose(weighted_sum, manual_weighted_sum),
            f"Weighted sum mismatch: Function = {weighted_sum}, Manual = {manual_weighted_sum}",
        )


class TestFieldLoss(unittest.TestCase):

    torch.manual_seed(42)
    pred = torch.randn(2, 3, 64, 32)
    target = torch.randn(2, 3, 64, 32)

    def test_default_is_pointwise(self):
        """the default and zero weights give plain mean absolute / squared error"""
        l1 = utils.field_loss(self.pred, self.target)
        l2 = utils.field_loss(self.pred, self.target, "l2", 0.0, 0.0)
        self.assertTrue(torch.allclose(l1, nn.L1Loss()(self.pred, self.target)))
        self.assertTrue(torch.allclose(l2, nn.MSELoss()(self.pred, self.target)))

    def test_reference_formula(self):
        """unit weights give the L1 + difference + pyramid loss of the Rivus Rayleigh-Taylor study"""
        e = self.pred - self.target
        ref = torch.mean(torch.abs(e))
        ref = ref + torch.mean(torch.abs(e[:, :, 1:] - e[:, :, :-1]))
        ref = ref + torch.mean(torch.abs(e[:, :, :, 1:] - e[:, :, :, :-1]))
        pyr, g = 0.0, e
        for lev in range(5):
            down = nn.functional.avg_pool2d(g, 2)
            up = nn.functional.interpolate(down, scale_factor=2, mode="nearest")
            pyr = pyr + 2.0**-lev * torch.mean(torch.abs(g - up))
            g = down
        ref = ref + pyr + 2.0**-5 * torch.mean(torch.abs(g))
        self.assertTrue(torch.allclose(utils.field_loss(self.pred, self.target, "l1", 1.0, 1.0, 5), ref))

    def test_constant_error(self):
        """a constant error has no pixel differences and no band-pass content"""
        for shape in [(1, 2, 16, 8), (1, 2, 8, 4, 4)]:
            e = torch.full(shape, -0.5)
            self.assertEqual(utils.neighbour_l1(e).item(), 0.0)
            self.assertAlmostEqual(utils.pyramid_l1(e, 2).item(), 0.25 * 0.5)

    def test_pyramid_small_and_odd_images(self):
        """sides shorter than 2^levels use fewer levels; odd sides are allowed"""
        e = torch.randn(2, 1, 5, 3)
        self.assertTrue(torch.allclose(utils.pyramid_l1(e, 5), utils.pyramid_l1(e, 1)))
        self.assertTrue(torch.isfinite(utils.field_loss(e, torch.zeros_like(e), "l1", 1.0, 1.0)))

    def test_3d_fields(self):
        """3D fields use differences along z too and a 2x2x2 average pyramid"""
        e = torch.randn(2, 3, 8, 4, 6)
        ref = sum(torch.mean(torch.abs(torch.diff(e, dim=d))) for d in (2, 3, 4))
        self.assertTrue(torch.allclose(utils.neighbour_l1(e), ref))
        down = nn.functional.avg_pool3d(e, 2)
        band = e - down.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
        ref = torch.mean(torch.abs(band)) + 0.5 * torch.mean(torch.abs(down))
        self.assertTrue(torch.allclose(utils.pyramid_l1(e, 1), ref))
        self.assertTrue(torch.isfinite(utils.field_loss(e, torch.zeros_like(e), "l1", 1.0, 1.0)))
        with self.assertRaises(ValueError):
            utils.pyramid_l1(torch.randn(2, 3, 8))


if __name__ == "__main__":
    unittest.main()
