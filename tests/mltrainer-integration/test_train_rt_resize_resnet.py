# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
import torch
from omegaconf import OmegaConf
from professor import mltrainer
from professor.__main__ import build_parser
from professor.vela.config import FullSchema, Keys
from professor.vela.utils import get_object_reference, initialize_function_calls_in_parameters
from tests.mltrainer_integration_paths import dataset_path


class TestEverything(unittest.TestCase):

    def test_prof_train_rt_data_resize_resnet(self):
        """train the 2D-resize-resnet generator on the RT data, then rebuild it from the vis config"""
        with tempfile.TemporaryDirectory() as run_directory:
            vis_config = f"{run_directory}/vis_config.yaml"
            parser = build_parser()
            parser.set_defaults(
                batch_size=10,
                lr=1e-4,
                num_epochs=2,
                seed=1231,
                n_checkpoint=1,
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
                z_kernel=1,
                upscale_type="transpose",
                residual=False,
                intermediate_channels=3,
                generator_type="2D-resize-resnet",
                act_fun="SiLU",
                res_blocks="0,0,0,0,1,1,2,2",
                no_coord_channels=False,
                no_batch_norm=True,
                dataloader_workers=6,
                divide_input_scale="1,1,1,1",
                n_sims=100,
                run_directory=run_directory,
                profile=False,
                profile_memory=False,
                compile=False,
                vis_config=vis_config,
            )
            args = parser.parse_args([])
            mltrainer.main(args)

            checkpoint = f"{run_directory}/0001.pt"
            self.assertTrue(os.path.exists(f"{run_directory}/0000.pt") and os.path.exists(checkpoint))
            conf = FullSchema(OmegaConf.to_container(OmegaConf.load(vis_config), resolve=True)).val
            pytorch = conf[Keys.MODEL.value][Keys.PYTORCH.value]
            self.assertEqual(pytorch["checkpoints"][0], checkpoint)
            invocation = pytorch[Keys.INVOCATION.value]
            model = get_object_reference(invocation[Keys.TARGET.value])(
                **initialize_function_calls_in_parameters(invocation[Keys.PARAMS.value])
            )
            model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
            with torch.no_grad():
                output = model.eval()(torch.rand(2, 4, 1, 1))
            self.assertEqual(tuple(output.shape), (2, 6, 768, 256))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
