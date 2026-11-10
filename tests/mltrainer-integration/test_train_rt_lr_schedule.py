# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
import argparse
import torch
from professor import mltrainer
from tests.mltrainer_integration_paths import dataset_path


def rt_args(run_directory, restart_model=""):
    return argparse.Namespace(
        batch_size=10,
        lr=3e-4,
        num_epochs=2,
        seed=1231,
        n_checkpoint=1,
        loss_target='l1',
        max_feature=32,
        min_feature=4,
        restart_model=restart_model,
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
        lr_scaling="none",
        lr_schedule="cosine",
        warmup_steps=5,
        lr_min=0.01,
        clip_grad=1.0,
    )


class TestEverything(unittest.TestCase):

    def test_prof_train_rt_data_lr_schedule(self):
        """train with warm-up, cosine decay and gradient clipping, then restart from the first epoch"""
        # 101 samples in batches of 10 on one GPU
        steps_per_epoch = 11
        with tempfile.TemporaryDirectory() as run_directory:
            mltrainer.main(rt_args(run_directory))

            first = torch.load(f"{run_directory}/0000.pt", map_location="cpu")
            last = torch.load(f"{run_directory}/0001.pt", map_location="cpu")
            self.assertEqual(first["scheduler"]["last_epoch"], steps_per_epoch)
            self.assertEqual(last["scheduler"]["last_epoch"], 2 * steps_per_epoch)
            # the learning rate is not scaled, and has decayed to lr_min * lr at the end of the run
            self.assertAlmostEqual(last["scheduler"]["base_lrs"][0], 3e-4)
            self.assertAlmostEqual(last["optimizer"]["param_groups"][0]["lr"], 3e-6)

            # a restart continues the schedule where the checkpoint left it
            restart_directory = f"{run_directory}/restart"
            os.mkdir(restart_directory)
            mltrainer.main(rt_args(restart_directory, restart_model=f"{run_directory}/0000.pt"))
            restarted = torch.load(f"{restart_directory}/0000.pt", map_location="cpu")
            self.assertEqual(restarted["scheduler"]["last_epoch"], 2 * steps_per_epoch)
            self.assertAlmostEqual(
                restarted["optimizer"]["param_groups"][0]["lr"], last["optimizer"]["param_groups"][0]["lr"]
            )


if __name__ == "__main__":
    unittest.main()
