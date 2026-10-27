# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

import professor

GENERATOR_ACTIVATION_FUNCTIONS = (
    "ReLU",
    "Tanh",
    "Softplus",
    "SoftSign",
    "Mish",
    "SiLU",
    "GELU",
    "CELU",
    "LeakyReLU",
    "ELU",
)

GENERATOR_TYPES = (
    "legacy",
    "2D",
    "2D-resize-resnet",
    "3D-triplane",
    "3D-voxel",
    # "3D-spectral"
)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="prof-trainer",
        description=f"Train full-field machine learning models in regression!\n{professor._name}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Mini-batch size on each GPU",
    )
    parser.add_argument(
        "--batch_multiplier",
        default=1,
        type=int,
        help="Number of mini-batches to accumulate before updating model weights.",
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Learning rate used by the optimizer. Note effective learning"
        " rate is applied based on the total number of GPUs and batch size.",
    )
    parser.add_argument(
        "--lr_scaling",
        default="linear",
        choices=("linear", "none"),
        help="'linear' multiplies --lr by batch_multiplier * number of GPUs * batch_size;"
        " 'none' uses --lr as the learning rate. Default is 'linear'.",
    )
    parser.add_argument(
        "--lr_schedule",
        default="constant",
        choices=("constant", "cosine"),
        help="Learning rate schedule over the optimizer steps of the run, after the warm-up:"
        " 'constant', or 'cosine' decay to lr_min * lr at the last step. Default is 'constant'.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Number of optimizer steps of linear learning rate warm-up from 0. Default is 0.",
    )
    parser.add_argument(
        "--lr_min",
        default=0.01,
        type=float,
        help="Final learning rate of the cosine schedule, as a fraction of the learning rate. Default is 0.01.",
    )
    parser.add_argument(
        "--clip_grad",
        default=0.0,
        type=float,
        help="Clip the global gradient norm to this value before each optimizer step. Default is 0, no clipping.",
    )
    parser.add_argument("--num_epochs", default=100, type=int, help="The total number of training epochs.")
    parser.add_argument(
        "--seed",
        default=1231,
        type=int,
        help="Does not do much, as two exact training script will return different ML Models.",
    )
    parser.add_argument(
        "--n_checkpoint",
        default=25,
        type=int,
        help="The frequency to checkpoint the model weights to disk.",
    )
    parser.add_argument(
        "--loss_target",
        default="l1",
        type=str,
        help="Either 'l1' for mean absolute error, or 'l2' for mean squared error. Default is l1",
    )
    parser.add_argument(
        "--max_feature",
        default=512,
        type=int,
        help="Maximum number of channels in the intermediate state of the model.",
    )
    parser.add_argument(
        "--min_feature",
        default=32,
        type=int,
        help="Minimum number of channels in the intermediate state of the model.",
    )
    parser.add_argument(
        "--restart_model",
        default="",
        type=str,
        help="The absolute file path to a previous model checkpoint '.pt' file.",
    )
    parser.add_argument(
        "--keys",
        default="density,velocity_x,velocity_y,pressure,energy,materials",
        type=str,
        help="Comma separated list of fields to fit model to. "
        "The number of entries must mach the number of fields in your dataset. "
        "The names here are only used to write out tensorboard training logs.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="The absolute path to the dataset.",
    )
    parser.add_argument(
        "--dataset_type",
        type=int,
        default=0,
        help="Type 0, 1, or 2 dataset.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="filelist.txt",
        help="The plaintext filelist of all of the hdf5 files. This should be located in the `datset_path`.",
    )
    parser.add_argument(
        "--x_kernel",
        type=int,
        default=1,
        help="Number of x pixels in the first representation of the model.",
    )
    parser.add_argument(
        "--y_kernel",
        type=int,
        default=3,
        help="Number of y pixels in the first representation of the model.",
    )
    parser.add_argument(
        "--z_kernel",
        type=int,
        default=1,
        help="Number of z pixels in the first representation of the model.",
    )
    parser.add_argument(
        "--upscale-type",
        type=str,
        default="transpose",
        help="Data upscaling method (transpose, nearest, linear)",
    )
    parser.add_argument("--residual", action="store_true", help="Use residual layers in the model.")
    parser.add_argument(
        "--res-blocks",
        type=str,
        default="1",
        help="2D-resize-resnet generator: residual blocks per stage, one count for all stages or a comma"
        " separated count per stage, coarse to fine.",
    )
    parser.add_argument(
        "--no-coord-channels",
        action="store_true",
        help="2D-resize-resnet generator: do not append (y, x) coordinate channels to the stage inputs.",
    )
    parser.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Build the generator without batch normalization.",
    )
    parser.add_argument(
        "--intermediate-channels",
        type=int,
        default=3,
        help="Number of intermediate channels to use prior to 3D triplane reconstruction",
    )
    parser.add_argument(
        "--generator-type",
        type=str,
        default="legacy",
        choices=GENERATOR_TYPES,
        help="Generator type",
    )
    parser.add_argument(
        "--act-fun",
        type=str,
        default="ReLU",
        choices=GENERATOR_ACTIVATION_FUNCTIONS,
        help="Activation function to use in the generator.",
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=6,
        help="The number of threads to use on each rank to stream the data from disk to RAM.",
    )
    parser.add_argument(
        "--divide_input_scale",
        default="1,1,1,1",
        type=str,
        help="Comma separated list to divide each input by. Only used on type 2 dataset",
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        default=1000,
        help="Number of image arrays in your dataset. Use a huge number to "
        " use all data. Use a smaller for quick epoch debugging.",
    )
    parser.add_argument(
        "--run_directory",
        type=str,
        default="",
        help="Directory to save checkpoints and tensorboard results. Will"
        " automatically check this folder if existing checkpoints exist,"
        " and if so restart from the latest checkpoint.",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode.")
    parser.add_argument(
        "--profile_memory", action="store_true", help="Requires --profile to be set, then enable memory profiling mode."
    )
    parser.add_argument("--compile", action="store_true", help="Whether to use torch.compile to optimize the module.")
    parser.add_argument("--vis-config", type=str, default="", help="Build a template visualization config file.")
    return parser


def prof_trainer() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from professor.mltrainer import main

    main(args)


if __name__ == "__main__":
    prof_trainer()
