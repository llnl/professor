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
        "--field_scaling",
        default="none",
        choices=("none", "std"),
        help="Scale each field to order 1 inside the model. 'std' subtracts the mean and divides by the"
        " standard deviation of each field in the training data. The model output and the logged errors"
        " stay in the units of the dataset; the training loss is on the scaled fields. Default is 'none'.",
    )
    parser.add_argument(
        "--field_offset",
        default="",
        type=str,
        help="Comma separated per-field offsets for the field scaling, (field - offset) / scale."
        " Turns on field scaling and replaces the computed means. Use --field_offset=-1,... when the"
        " first value is negative.",
    )
    parser.add_argument(
        "--field_scale",
        default="",
        type=str,
        help="Comma separated per-field scales for the field scaling. Turns on field scaling and"
        " replaces the computed standard deviations.",
    )
    parser.add_argument(
        "--input_scaling",
        default="none",
        choices=("none", "minmax"),
        help="Scale each input to order 1 inside the model. 'minmax' maps the range of each input in"
        " the training data (after --log_inputs) to [-1, 1]. Default is 'none'.",
    )
    parser.add_argument(
        "--input_offset",
        default="",
        type=str,
        help="Comma separated per-input offsets for the input scaling, (input - offset) / scale."
        " Turns on input scaling and replaces the computed centers. Use --input_offset=-1,... when the"
        " first value is negative.",
    )
    parser.add_argument(
        "--input_scale",
        default="",
        type=str,
        help="Comma separated per-input scales for the input scaling. Turns on input scaling and"
        " replaces the computed half ranges.",
    )
    parser.add_argument(
        "--log_inputs",
        default="",
        type=str,
        help="Comma separated indices of (positive) inputs to take the log of before the input scaling.",
    )
    parser.add_argument(
        "--scaling_samples",
        default=1024,
        type=int,
        help="Number of training samples to estimate the field and input scaling from. Default is 1024.",
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
