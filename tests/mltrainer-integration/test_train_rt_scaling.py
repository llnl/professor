# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
import argparse
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from professor import mltrainer
from professor.vela.config import FullSchema, Keys
from professor.vela.utils import get_object_reference, initialize_function_calls_in_parameters
from tests.mltrainer_integration_paths import dataset_path


def rt_args(run_directory, vis_config, restart_model=""):
    return argparse.Namespace(
        batch_size=10,
        lr=1e-4,
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
        vis_config=vis_config,
        field_scaling="std",
        input_scaling="minmax",
        log_inputs="0,1,2",
        scaling_samples=50,
    )


class TestEverything(unittest.TestCase):

    def test_prof_train_rt_data_scaling(self):
        """train with field and input scaling, rebuild the model from the vis config, and restart"""
        with tempfile.TemporaryDirectory() as run_directory:
            vis_config = f"{run_directory}/vis_config.yaml"
            mltrainer.main(rt_args(run_directory, vis_config))

            checkpoint = f"{run_directory}/0001.pt"
            state = torch.load(checkpoint, map_location="cpu")["model"]
            field_scale = state["last_layer.scale"].flatten()
            input_offset = state["first_layer.offset"].flatten()
            input_scale = state["first_layer.scale"].flatten()
            self.assertTrue(torch.all(field_scale > 0))
            self.assertEqual(state["first_layer.log_mask"].flatten().tolist(), [True, True, True, False])

            # the input scaling maps the inputs of the training samples to [-1, 1]
            path = dataset_path('rayleigh-taylor-single')
            files = np.genfromtxt(f"{path}/filelist.txt", dtype=None, encoding="UTF-8")[:101]
            inputs = []
            for i in np.unique(np.linspace(0, len(files) - 1, 50).round().astype(int)):
                with h5py.File(f"{path}/{files[i]}", "r") as f:
                    inputs.append(f["inputs"][:])
            inputs = np.array(inputs, dtype=np.float64)
            inputs[:, :3] = np.log(inputs[:, :3])
            scaled = (inputs - input_offset.double().numpy()) / input_scale.double().numpy()
            self.assertTrue(np.allclose(scaled.min(axis=0), -1.0, atol=1e-5))
            self.assertTrue(np.allclose(scaled.max(axis=0), 1.0, atol=1e-5))

            # prof-vis rebuilds the same model, which takes raw inputs and returns fields in dataset units
            conf = FullSchema(OmegaConf.to_container(OmegaConf.load(vis_config), resolve=True)).val
            pytorch = conf[Keys.MODEL.value][Keys.PYTORCH.value]
            invocation = pytorch[Keys.INVOCATION.value]
            model = get_object_reference(invocation[Keys.TARGET.value])(
                **initialize_function_calls_in_parameters(invocation[Keys.PARAMS.value])
            )
            model.load_state_dict(state)
            sliders = conf[Keys.GUI.value][Keys.NAPARI.value][Keys.SLIDERS.value]
            raw = torch.tensor([[sliders[f"parameter_{ii}"]["initial_value"] for ii in range(4)]]).view(1, 4, 1, 1)
            self.assertTrue(np.isclose(sliders["parameter_0"]["lower_bound"], np.exp(inputs[:, 0].min()), rtol=1e-5))
            self.assertTrue(np.isclose(sliders["parameter_3"]["upper_bound"], inputs[:, 3].max(), rtol=1e-5))
            with torch.no_grad():
                output = model.eval()(raw)
            self.assertEqual(tuple(output.shape), (1, 6, 768, 256))
            self.assertTrue(torch.isfinite(output).all())

            # a restart keeps the scaling of the checkpoint
            restart_directory = f"{run_directory}/restart"
            os.mkdir(restart_directory)
            mltrainer.main(rt_args(restart_directory, "", restart_model=checkpoint))
            restarted = torch.load(f"{restart_directory}/0001.pt", map_location="cpu")["model"]
            for key in ["last_layer.offset", "last_layer.scale", "first_layer.offset", "first_layer.scale"]:
                self.assertTrue(torch.equal(restarted[key], state[key]))


if __name__ == "__main__":
    unittest.main()
