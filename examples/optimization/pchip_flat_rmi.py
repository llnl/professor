# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from time import time
import numpy as np
import torch
import professor.vela
from scipy.optimize import fmin_l_bfgs_b
# import pyDOE
import matplotlib.pyplot as plt

cfg = 'pchip.yaml'
config = professor.vela.config.Config(cfg)
model = professor.vela.model.PyTorchModel(config)
device = model._get_checkpoint_device(0)
# get the pytorch model reference
model = model.models[0].ref

bounds = np.ones((4, 2))
bounds = bounds * 0.25
bounds[:, 0] *= -1

x = torch.zeros(1, 6, 1, 1).float()
x = x.to(device)
x[0, -1, 0, 0] = 13.0
x[0, -2, 0, 0] = 0.2


Softmax = torch.nn.Softmax(dim=0)
Softmin = torch.nn.Softmin(dim=0)

with torch.no_grad():
    y_hat = model(x).to('cpu')[0, 0]
    thresh = torch.where(y_hat >= 2.0, 1.0, 0.0)
    idx_values = torch.argmax(thresh, axis=1)
    # .float()
    # softmax = Softmax(idx_values)
    # softmin = Softmin(idx_values)
    norm = idx_values.max() - idx_values.min()


def my_objective(x_try, return_grad=False, plot=False):
    with torch.no_grad():
        x = torch.zeros(1, 6, 1, 1).float()
        x = x.to(device)
        x[0, -1, 0, 0] = 13.0
        x[0, -2, 0, 0] = 0.2
        x[0, :4, 0, 0] = torch.from_numpy(x_try)
        # x = x.requires_grad_()
        # print(x)
        # model.zero_grad()
        # x.zero_grad()
        y_hat = model(x).to('cpu')[0, 0]
        # print(y_hat)
        one = torch.ones(1)#.requires_grad_()
        zero = torch.zeros(1)#.requires_grad_()
        thresh = torch.where(y_hat >= 2.0, one, zero)
        threshf = torch.fliplr(thresh)
        # print('Thresh', thresh)
        idx_values = torch.argmax(threshf, axis=1)
        # print(idx_values)
        # print('idx values', idx_values)
        # norm =  torch.mean((idx_values.max() - idx_values) + (idx_values - idx_values.min()))
        norm =  idx_values.max() - idx_values.min()
        if return_grad:
            norm.backward()
            grads = x.grad[0, :4, 0, 0]
            return grads.detach().numpy()
        if plot:
            plt.figure()
            plt.title(f'Norm: {norm} x: {x_try}')
            plt.plot(idx_values.detach())
            plt.figure()
            plt.title(f'Norm: {norm} x: {x_try}')
            plt.imshow(threshf)
            plt.show()
            plt.close()
            plt.close()
        return norm.item()


xtry = np.zeros(4)
obj = my_objective(xtry, plot=False)
print(obj)
n_runs = 100
x_sol = np.zeros((n_runs, 4))
x_0s = np.zeros_like(x_sol)
f_sol = np.zeros(n_runs)
b = bounds[:, 1]
a = bounds[:, 0]

for i in range(n_runs):
    x0 = np.random.random(4)
    x0 = (b-a) * x0 + a
    t0 = time()
    res = fmin_l_bfgs_b(my_objective, x0, epsilon=1e-1, approx_grad=True,
                        bounds=bounds, factr=10.0)
    t1 = time()
    print(f'RunTime: {t1-t0} X: {res[0]} f(X): {res[1]} {res[2]}')
    x_sol[i] = res[0]
    x_0s[i] = x0
    f_sol[i] = res[1]
np.save('x_sol_flat.npy', x_sol)
np.save('x0_flat.npy', x_0s)
np.save('f_sol_flat.npy', f_sol)
