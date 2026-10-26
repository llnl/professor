# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# import os
import unittest

from professor import mltrainer
from professor.__main__ import build_parser
from tests.mltrainer_integration_paths import dataset_path


class TestEverything(unittest.TestCase):
    def test_prof_train_ds_data(self):
        parser = build_parser()
        parser.set_defaults(
            profile=False,
            profile_memory=False,
            compile=False,
        )
        args = parser.parse_args(
            [
                "--batch_size",
                "10",
                "--lr",
                "1e-4",
                "--num_epochs",
                "1",
                "--seed",
                "1231",
                "--n_checkpoint",
                "25",
                "--loss_target",
                "l1",
                "--max_feature",
                "128",
                "--min_feature",
                "8",
                "--restart_model",
                "",
                "--keys",
                "density,velocity_x,velocity_y",
                "--dataset_path",
                dataset_path("double_sinewave"),
                "--dataset_type",
                "1",
                "--dataset_file",
                "filelist.txt",
                "--y_kernel",
                "4",
                "--x_kernel",
                "4",
                "--dataloader_workers",
                "6",
                "--divide_input_scale",
                "1,1,1,1",
                "--n_sims",
                "100",
                "--run_directory",
                "",
                "--vis-config",
                "",
            ]
        )
        mltrainer.main(args)


if __name__ == "__main__":
    unittest.main()
