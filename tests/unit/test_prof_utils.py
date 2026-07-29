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


class TestZeroHeavyLoss(unittest.TestCase):
    def test_weights_positive_and_negative_signal_pixels(self):
        pred = torch.tensor([0.0, 0.0, 0.0, 0.0])
        target = torch.tensor([0.0, 5e-5, 2.0, -3.0])
        loss = utils.ZeroHeavyLoss(threshold=1e-4, nonzero_weight=10.0)

        actual = loss(pred, target)
        weighted_error = torch.tensor([0.0, 5e-5, 20.0, 30.0])
        expected = torch.sum(weighted_error) / torch.tensor(22.0)

        self.assertTrue(
            torch.allclose(actual, expected),
            f"Zero-heavy loss mismatch: Function = {actual}, Manual = {expected}",
        )

    def test_nonzero_signal_has_larger_gradient(self):
        pred = torch.tensor([0.0, 0.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        loss = utils.ZeroHeavyLoss(threshold=1e-4, nonzero_weight=10.0)

        actual = loss(pred, target)
        actual.backward()

        self.assertEqual(pred.grad[0].item(), 0.0)
        self.assertLess(pred.grad[1].item(), 0.0)
        self.assertTrue(torch.allclose(pred.grad[1], torch.tensor(-10.0 / 11.0)))

    def test_sum_and_none_reductions(self):
        pred = torch.tensor([1.0, 1.0])
        target = torch.tensor([0.0, -2.0])

        sum_loss = utils.ZeroHeavyLoss(threshold=1e-4, nonzero_weight=5.0, reduction="sum")
        none_loss = utils.ZeroHeavyLoss(threshold=1e-4, nonzero_weight=5.0, reduction="none")

        self.assertTrue(torch.allclose(sum_loss(pred, target), torch.tensor(16.0)))
        self.assertTrue(torch.allclose(none_loss(pred, target), torch.tensor([1.0, 15.0])))


if __name__ == "__main__":
    unittest.main()
