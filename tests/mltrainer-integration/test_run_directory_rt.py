# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import unittest

from torchinfo import summary  # noqa: F401

from professor import mltrainer
from professor.__main__ import build_parser
from tests.mltrainer_integration_paths import dataset_path

mypath = "/tmp/test-run"


def build_test_args(num_epochs, run_directory):
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
            "1e-4",
            "--num_epochs",
            str(num_epochs),
            "--seed",
            "1231",
            "--n_checkpoint",
            "25",
            "--loss_target",
            "l1",
            "--max_feature",
            "32",
            "--min_feature",
            "8",
            "--restart_model",
            "",
            "--keys",
            "density,velocity_x,velocity_y,pressure,energy,materials",
            "--dataset_path",
            dataset_path("rayleigh-taylor-single"),
            "--dataset_type",
            "0",
            "--dataset_file",
            "filelist.txt",
            "--y_kernel",
            "3",
            "--x_kernel",
            "1",
            "--dataloader_workers",
            "6",
            "--divide_input_scale",
            "1,1,1,1",
            "--n_sims",
            "60",
            "--run_directory",
            run_directory,
            "--vis-config",
            "",
        ]
    )


class TestEverything(unittest.TestCase):
    def test_prof_train_rt_run_directory(self):
        if os.path.exists(mypath):
            shutil.rmtree(mypath)
        os.mkdir(mypath)

        args = build_test_args(1, mypath)
        mltrainer.main(args)
        # Starting restart from run_directory
        # also test if the path has extra / at end of it
        args = build_test_args(2, mypath + "/")
        mltrainer.main(args)


if __name__ == "__main__":
    unittest.main()
