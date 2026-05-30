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

# storage for logging objective calls
run_histories = []          # list of np.array, one per run
current_run_values = None   # will be a Python list during each run

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
    global current_run_values  # use the per-run buffer
    x = torch.zeros(1, 6, 1, 1).float()
    x = x.to(device)
    x[0, -1, 0, 0] = 13.0
    x[0, -2, 0, 0] = 0.2
    x[0, :4, 0, 0] = torch.from_numpy(x_try)
    if return_grad:
        x.requires_grad()
    with torch.no_grad():
        # x = x.requires_grad_()
        # print(x)
        # model.zero_grad()
        # x.zero_grad()
        y_hat = model(x).to('cpu')[0, 0]
        # print(y_hat)
        one = torch.ones(1)#.requires_grad_()
        zero = torch.zeros(1)#.requires_grad_()
        thresh = torch.where(y_hat >= 1.95, one, zero)
        threshf = torch.fliplr(thresh)
        # print('Thresh', thresh)
        idx_values = torch.argmax(threshf, axis=1)
        # print(idx_values)
        # print('idx values', idx_values)
        # norm =  torch.mean((idx_values.max() - idx_values) + (idx_values - idx_values.min()))
        norm =  -1.0 * (idx_values.max() - idx_values.min())
        # record the objective value for this function call
        if current_run_values is not None:
            current_run_values.append(norm.item())
            if norm.item() < -300:
                print('WTF!!!')
                print(x_try, norm.item(), print(norm))
                print('WTF!!!')
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
            plt.savefig('temp.png')
            plt.close()
        return norm.item()


xtry = np.zeros(4)
xtry[0] = -.25
xtry[1] = .25
xtry[2] = -.25
xtry[3] = .25
obj = my_objective(xtry, plot=False)
print(obj)
n_runs = 100
x_sol = np.zeros((n_runs, 4))
x_0s = np.zeros_like(x_sol)
f_sol = np.zeros(n_runs)
b = bounds[:, 1]
a = bounds[:, 0]

for i in range(n_runs):
    # reset per-run log
    current_run_values = []
    x0 = np.random.random(4)
    x0 = (b-a) * x0 + a
    t0 = time()
    res = fmin_l_bfgs_b(
        my_objective,
        x0,
        epsilon=1e-2,  # only used to approx gradient
        approx_grad=True,
        bounds=bounds,
        factr=10.0,
    )
    t1 = time()
    # finalize this run's history
    run_histories.append(np.array(current_run_values, dtype=float))
    # print(f'RunTime: {t1-t0} X: {res[0]} f(X): {res[1]} {res[2]}')
    print(len(run_histories[-1]), min(run_histories[-1]), f'X: {res[0]} f(X): {res[1]} {res[2]}')
    x_sol[i] = res[0]
    x_0s[i] = x0
    f_sol[i] = res[1]
np.save('x_sol_max.npy', x_sol)
np.save('x0_max.npy', x_0s)
np.save('f_sol_max.npy', f_sol)

# === Post-process function call histories ===
# Compute running max of objective per run and pad to common length
max_len = max(len(h) for h in run_histories)
mins = [x.min() for x in run_histories]
n_runs = len(run_histories)

# padded array: shape (n_runs, max_len)
# fill with NaN so you can easily mask unused entries if needed
# min per row, ignoring NaNs
# min_per_run = np.nanmin(original_min_obj_2d, axis=1)  # shape (n_runs,)

# padded_min_obj = np.repeat(min_per_run[:, None], max_len, axis=1)
padded_min_obj = np.full((n_runs, max_len), np.nan)
# row_mins = np.array([min_obj[:max_lens[i]].min() for i in range(n_runs)])
# padded_min_obj = np.repeat(row_mins[:, None], max_len, axis=1)
for i, h in enumerate(run_histories):
    # running min for this run
    running_min = np.ones(max_len)
    for j, k in enumerate(running_min):
        running_min[j] = np.min(h[:j+1])
    # running_min = np.where(h > h.min(), h, h.min())
    padded_min_obj[i] = running_min
    # padded_min_obj[i, len(running_min):] = np.ones(max_len-len(running_min))*h.min()
    # running_min = np.minimum.accumulate(h)
    # padded_min_obj[i, :len(running_min)] = running_min

# # save histories and padded running maxima
# np.save('run_histories.npy', run_histories, allow_pickle=True)
# np.save('padded_max_obj.npy', padded_max_obj)

# === Plot: max objective vs function calls for each run ===
plt.figure()
for i in range(n_runs):
    calls = np.arange(max_len)
    if i == 0:
        plt.plot(
            calls,
            -1*padded_min_obj[i],
            alpha=0.1,
            color='gray',
            label='L-BFGS-B'
        )
    else:
        plt.plot(
            calls,
            -1*padded_min_obj[i],
            alpha=0.1,
            color='gray',
        )
# Compute mean and std across runs
mean_obj = np.mean(padded_min_obj, axis=0)
std_obj = np.std(padded_min_obj, axis=0)

ai_run_history = 1.0*np.array(
    [-256, -253, -188, -184, -163, -133, -14, -11, -11, -36,
     -277, -277, -268, -257, -255, -235, -237, -225, -229, -220, -170, -146,
     -274, -274, -273, -273, -273, -271, -271, -270, -269, -265, -264,]
)
best_ai_run_history = np.zeros_like(ai_run_history)
for i, x in enumerate(ai_run_history):
    best_ai_run_history[i] = np.min(ai_run_history[:i+1])

plt.plot(np.arange(len(best_ai_run_history)), -1*best_ai_run_history, '-', label='o3')

# # Plot mean curve
# plt.plot(
#     calls,
#     mean_obj,
#     color='blue',
#     linewidth=2,
#     label='Mean'
# )

# # Plot shaded std band
# plt.fill_between(
#     calls,
#     mean_obj - 2*(std_obj),
#     mean_obj + 2*(std_obj),
#     color='blue',
#     alpha=0.3,
#     label='±2 std'
# )


plt.xlabel('Function calls')
plt.ylabel('Best objective value')
# plt.title('')
plt.grid(True)
# Limit x-axis from 0 to 50
plt.xlim(0, 50)
plt.legend()
plt.tight_layout()
plt.savefig('maxrmi_objective_vs_calls.png', dpi=150)
plt.close()

plt.figure()
for i in range(n_runs):
    calls = np.arange(len(run_histories[i]))
    if i == 0:
        plt.plot(
            calls,
            run_histories[i],
            'o',
            alpha=0.1,
            color='gray',
            label='L-BFGS-B'
        )
    else:
        plt.plot(
            calls,
            run_histories[i],
            'o',
            alpha=0.1,
            color='gray',
        )
plt.xlabel('Function calls')
plt.ylabel('Objective value')
plt.plot(np.arange(len(ai_run_history)), ai_run_history, 'o', label='o3')
plt.grid(True)
# Limit x-axis from 0 to 50
plt.xlim(0, 50)
plt.legend()
plt.tight_layout()
plt.savefig('objective_vs_calls.png', dpi=150)
plt.close()
