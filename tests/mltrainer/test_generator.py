# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# import os
import unittest
import numpy as np
import torch
from professor.torch_models import Generator
from torchinfo import summary  # noqa F:401

# set random seed
np.random.seed(1231231)
torch.manual_seed(1231231)
# # set os threads to 1...
# os.environ["OMP_NUM_THREADS"] = "1"
# # set to 1 thread
# torch.set_num_threads(1)


class TestEverything(unittest.TestCase):

    def test_model_output(self):
        im_sizes = [128, 256, 512, 1024]
        kernels = [1, 2, 4, 8]
        X = torch.rand(2, 4, 1, 1)  # batch size  # random inputs
        for kernel in kernels:
            for im_size in im_sizes:

                model = Generator(4, im_size, 1, y_kernel=kernel, x_kernel=kernel)
                Y = model(X)
                # summary(model, input_size=(1, 4, 1, 1))
                Y_shape = Y.shape
                self.assertTrue(Y_shape[-1] == im_size)
                self.assertTrue(Y_shape[-2] == im_size)

    def test_rectangles(self):
        im_sizes = [256, 512]
        kernel_x = [1, 2, 4]
        kernel_y = [1, 2, 4]
        X = torch.rand(2, 2, 1, 1)  # batch size  # random inputs
        for xker in kernel_x:
            for yker in kernel_y:
                for im_size in im_sizes:
                    max_kern = max(xker, yker)
                    min_kern = min(xker, yker)
                    min_im_size = im_size * min_kern // max_kern
                    model = Generator(2, im_size, 1, y_kernel=yker, x_kernel=xker)
                    Y = model(X)
                    # summary(model, input_size=(1, 2, 1, 1))
                    _, _, a, b = Y.shape
                    self.assertTrue(min(a, b) == min_im_size)
                    self.assertTrue(max(a, b) == im_size)

    def test_activation_functions_init(self):
        im_size = 64
        X = torch.rand(10, 4, 1, 1)  # batch size  # random inputs
        model = Generator(4, im_size, 1)
        activation_functions = model.activation_functions
        for act_fun in activation_functions:
            # print(act_fun)
            model = Generator(4, im_size, 1, act_fun=act_fun)
            # print(model)
            # summary(model, input_size=(1, 4, 1, 1))
            _ = model(X)


if __name__ == "__main__":
    unittest.main()
