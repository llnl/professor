# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import io
import os
import tempfile
import unittest
import argparse
import torch
from professor import mltrainer
from tests.mltrainer_integration_paths import dataset_path


class TestEverything(unittest.TestCase):

    def test_prof_train_rt_data_field_loss(self):
        """train on the RT data with the pixel-difference and pyramid loss terms switched on"""
        with tempfile.TemporaryDirectory() as run_directory:
            args = argparse.Namespace(
                batch_size=10,
                lr=1e-4,
                num_epochs=2,
                seed=1231,
                n_checkpoint=1,
                loss_target='l1',
                grad_weight=1.0,
                pyr_weight=1.0,
                pyr_levels=5,
                max_feature=32,
                min_feature=4,
                restart_model="",
                keys="density,velocity_x,velocity_y,pressure,energy,materials",
                dataset_path=dataset_path('rayleigh-taylor-single'),
                dataset_type=0,
                dataset_file='filelist.txt',
                y_kernel=3,
                x_kernel=1,
                z_kernel=1,
                upscale_type="transpose",
                residual=False,
                intermediate_channels=3,
                generator_type="legacy",
                act_fun="ReLU",
                dataloader_workers=6,
                divide_input_scale="1,1,1,1",
                n_sims=100,
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

            self.assertIn("loss: l1 + 1.0 * pixel differences + 1.0 * pyramid (5 levels)", log.getvalue())
            checkpoint = f"{run_directory}/0001.pt"
            self.assertTrue(os.path.exists(f"{run_directory}/0000.pt") and os.path.exists(checkpoint))
            state = torch.load(checkpoint, map_location="cpu")["model"]
            self.assertTrue(all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()))


if __name__ == "__main__":
    unittest.main()
