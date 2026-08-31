# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# import os
import unittest
import argparse
from professor import mltrainer
from tests.mltrainer_integration_paths import dataset_path
from torchinfo import summary  # noqa F:401


class TestEverything(unittest.TestCase):

    def test_prof_train_rt_data(self):
        args = argparse.Namespace(
            batch_size=10,
            lr=1e-4,
            num_epochs=1,
            seed=1231,
            n_checkpoint=25,
            loss_target='l1',
            max_feature=32,
            min_feature=4,
            restart_model="",
            keys="density,velocity_x,velocity_y,pressure,energy,materials",
            dataset_path=dataset_path('rayleigh-taylor-single'),
            dataset_type=0,
            dataset_file='filelist.txt',
            y_kernel=3,
            x_kernel=1,
            dataloader_workers=6,
            divide_input_scale="1,1,1,1",
            n_sims=100,
            run_directory="",
            profile=False,
            profile_memory=False,
            compile=False,
            vis_config="",
        )
        mltrainer.main(args)


if __name__ == "__main__":
    unittest.main()
