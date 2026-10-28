# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import io
import math
import os
import re
import tempfile
import unittest
import argparse
import h5py
import numpy as np
from professor import mltrainer


def write_dataset(path: str, n_files: int = 4, n_samples: int = 16) -> None:
    """type 1 dataset of 3 analytic fields on 48 x 16 pixels with inputs (a, b, c, t) and their
    exact sensitivities d fields / d (a, b, c) under the key 'jacobian'"""
    rng = np.random.default_rng(0)
    y, x = np.meshgrid(np.linspace(0, 1, 48), np.linspace(0, 1, 16), indexing="ij")
    rows = []
    for i in range(n_files):
        inputs = rng.uniform(0.5, 1.5, size=(n_samples, 4))
        fields = np.zeros((n_samples, 3, 48, 16))
        jac = np.zeros((n_samples, 3, 3, 48, 16))
        for j, (a, b, c, t) in enumerate(inputs):
            fields[j, 0] = a * np.sin(math.pi * y) + b * x
            fields[j, 1] = b * c * np.cos(math.pi * x)
            fields[j, 2] = np.exp(-c) * y * t
            jac[j, 0, 0] = np.sin(math.pi * y)
            jac[j, 1, 0] = x
            jac[j, 1, 1] = c * np.cos(math.pi * x)
            jac[j, 2, 1] = b * np.cos(math.pi * x)
            jac[j, 2, 2] = -np.exp(-c) * y * t
        name = f"sim_{i:03d}.h5"
        with h5py.File(f"{path}/{name}", "w") as f:
            f["inputs"] = inputs.astype(np.float32)
            f["fields"] = fields.astype(np.float32)
            f["jacobian"] = jac.astype(np.float32)
        rows += [f"{name} {j}" for j in range(n_samples)]
    with open(f"{path}/filelist.txt", "w") as f:
        f.write("\n".join(rows) + "\n")


class TestEverything(unittest.TestCase):

    def test_prof_train_sobolev(self):
        """train with the Sobolev term on a synthetic dataset with exact sensitivities"""
        with tempfile.TemporaryDirectory() as dataset, tempfile.TemporaryDirectory() as run_directory:
            write_dataset(dataset)
            args = argparse.Namespace(
                batch_size=8,
                lr=1e-4,
                num_epochs=3,
                seed=1231,
                n_checkpoint=1,
                loss_target='l1',
                grad_weight=0.0,
                pyr_weight=0.0,
                pyr_levels=5,
                sobolev_weight=1.0,
                sobolev_key="jacobian",
                max_feature=32,
                min_feature=4,
                restart_model="",
                keys="f0,f1,f2",
                dataset_path=dataset,
                dataset_type=1,
                dataset_file='filelist.txt',
                y_kernel=3,
                x_kernel=1,
                z_kernel=1,
                upscale_type="transpose",
                residual=False,
                intermediate_channels=3,
                generator_type="legacy",
                act_fun="SiLU",
                dataloader_workers=2,
                divide_input_scale="1,1,1,1",
                n_sims=1000,
                run_directory=run_directory,
                profile=False,
                profile_memory=False,
                compile=False,
                vis_config="",
            )
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                mltrainer.main(args)
            print(log.getvalue())

            self.assertIn("sobolev: 1.0 * derivative mse, 'jacobian' rms per input", log.getvalue())
            sobolev = [float(v) for v in re.findall(r"^\d{4} sobolev (\S+)$", log.getvalue(), re.MULTILINE)]
            self.assertEqual(len(sobolev), 3)
            self.assertTrue(all(math.isfinite(v) and v > 0 for v in sobolev))
            self.assertTrue(os.path.exists(f"{run_directory}/0002.pt"))


if __name__ == "__main__":
    unittest.main()
