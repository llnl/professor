# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import os
import shutil
import tempfile
import unittest
import torch
from professor import mltrainer
from professor.__main__ import build_parser
from tests.mltrainer_integration_paths import dataset_path


def rt_args(run_directory, restart_model="", num_epochs=2, schedule=True):
    parser = build_parser()
    parser.set_defaults(
        profile=False,
        profile_memory=False,
        compile=False,
    )
    return parser.parse_args(
        [
            "--batch_size",
            "10",
            "--lr",
            "3e-4",
            "--num_epochs",
            str(num_epochs),
            "--seed",
            "1231",
            "--n_checkpoint",
            "1",
            "--loss_target",
            "l1",
            "--max_feature",
            "32",
            "--min_feature",
            "4",
            "--restart_model",
            restart_model,
            "--keys",
            "density,velocity_x,velocity_y,pressure,energy,materials",
            "--dataset_path",
            dataset_path('rayleigh-taylor-single'),
            "--dataset_type",
            "0",
            "--dataset_file",
            "filelist.txt",
            "--y_kernel",
            "3",
            "--x_kernel",
            "1",
            "--z_kernel",
            "1",
            "--upscale-type",
            "transpose",
            "--intermediate-channels",
            "3",
            "--generator-type",
            "legacy",
            "--act-fun",
            "ReLU",
            "--dataloader_workers",
            "6",
            "--divide_input_scale",
            "1,1,1,1",
            "--n_sims",
            "100",
            "--run_directory",
            run_directory,
            "--vis-config",
            "",
            "--lr_scaling",
            "none",
            "--clip_grad",
            "1.0",
        ]
        + (["--lr_schedule", "cosine", "--warmup_steps", "5", "--lr_min", "0.01"] if schedule else [])
    )


def cosine_lr(step, total, warmup=5, lr=3e-4, lr_min=0.01):
    """learning rate of optimizer step `step` of the schedule of rt_args, after the warm-up"""
    progress = min(1.0, (step - warmup) / (total - 1 - warmup))
    return lr * (lr_min + (1 - lr_min) * 0.5 * (1 + math.cos(math.pi * progress)))


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
            self.assertEqual(last["lr_schedule"]["lr_schedule"], "cosine")
            self.assertEqual(last["lr_schedule"]["total_steps"], 2 * steps_per_epoch)
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

    def test_prof_train_rt_data_lr_schedule_longer_restart(self):
        """a restart without the schedule arguments and with more epochs decays on from the checkpoint"""
        steps_per_epoch = 11
        with tempfile.TemporaryDirectory() as run_directory:
            first_directory = f"{run_directory}/first"
            os.mkdir(first_directory)
            mltrainer.main(rt_args(first_directory))
            first = torch.load(f"{first_directory}/0000.pt", map_location="cpu")
            lr_at_restart = first["optimizer"]["param_groups"][0]["lr"]
            self.assertAlmostEqual(lr_at_restart, cosine_lr(steps_per_epoch, 2 * steps_per_epoch))

            # continue from the first epoch in a run directory, for 3 epochs instead of 2
            restart_directory = f"{run_directory}/restart"
            os.mkdir(restart_directory)
            shutil.copy(f"{first_directory}/0000.pt", restart_directory)
            mltrainer.main(rt_args(restart_directory, num_epochs=3, schedule=False))
            saved = torch.load(f"{restart_directory}/0002.pt", map_location="cpu")["lr_schedule"]
            self.assertEqual(saved["lr_schedule"], "cosine")
            self.assertEqual(saved["warmup_steps"], 5)
            self.assertEqual(saved["total_steps"], 3 * steps_per_epoch)
            self.assertEqual(saved["decay_start"], steps_per_epoch)
            self.assertAlmostEqual(3e-4 * saved["decay_from"], lr_at_restart)
            last = torch.load(f"{restart_directory}/0002.pt", map_location="cpu")
            self.assertEqual(last["scheduler"]["last_epoch"], 3 * steps_per_epoch)
            self.assertAlmostEqual(last["optimizer"]["param_groups"][0]["lr"], 3e-6)

            # a different schedule than the checkpoint's is an error
            different = rt_args(restart_directory, num_epochs=4)
            different.lr_min = 0.1
            with self.assertRaisesRegex(ValueError, "--lr_min 0.1 differs from 0.01"):
                mltrainer.main(different)


if __name__ == "__main__":
    unittest.main()
