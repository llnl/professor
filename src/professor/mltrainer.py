# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import multiprocessing
import argparse
import socket
import contextlib
import numpy as np
from numpy.typing import NDArray
import h5py
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import professor
from professor.utils import (
    consolidated_loss,
    field_loss,
    forward_jacobian_vector_product,
    last_checkpoint,
    sobolev_loss,
)
from professor.layers import AlphaLinear
from professor.torch_models import build_generator
from professor.vela.config import build_template_config, update_config_checkpoint
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from time import time, sleep
from torchinfo import summary
from typing import cast, Tuple, Any


class CompleteDataset(Dataset[Any]):
    """One sample per hdf5 file. With sensitivity_key set, items also hold the field sensitivities
    f[sensitivity_key], shape (k, *fields.shape): d fields / d inputs for the first k inputs."""

    def __init__(
        self,
        filelist: NDArray[np.str_],
        n_channels: int,
        path: str,
        sensitivity_key: str = "",
    ):
        self._length: int = len(filelist)
        self._path: str = path
        self.pixels_y: int = 0
        self.pixels_x: int = 0
        self.pixels_z: int = 0

        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.sensitivity_key: str = sensitivity_key
        self.__getsizes__(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        myfile = f"{self._path}/{self.filelist[i]}"
        try:
            with h5py.File(myfile, "r") as f:
                # calling nan_to_num does seem to impact performance
                inputs = cast(h5py.Group, f["inputs"])[:]
                data = cast(h5py.Group, f["fields"])[:]
                sensitivities = cast(h5py.Group, f[self.sensitivity_key])[:] if self.sensitivity_key else None
            inputs = torch.from_numpy(inputs)
            data = torch.from_numpy(data)
            if sensitivities is not None:
                return inputs.view(-1, 1, 1), data, torch.from_numpy(sensitivities)
            return inputs.view(-1, 1, 1), data
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)

    def __getsizes__(self, i: int) -> None:
        x, y = self.__getitem__(i)[:2]
        self.n_input = x.shape[0]
        self.pixels_y = y.shape[1]
        self.pixels_x = y.shape[2]
        if len(y.shape) == 4:
            self.pixels_z = y.shape[3]


class CompleteDatasetDivideScaling(CompleteDataset):
    """Type 0 with the inputs divided by scaling; sensitivities are then multiplied by it, so they
    stay derivatives with respect to the model inputs."""

    def __init__(
        self,
        filelist: NDArray[np.str_],
        n_channels: int,
        path: str,
        scaling: torch.Tensor,
        sensitivity_key: str = "",
    ) -> None:
        self._length: int = len(filelist)
        self._path: str = path
        self.pixels_y: int = 0
        self.pixels_x: int = 0
        self.pixels_z: int = 0
        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.scaling: torch.Tensor = scaling
        self.sensitivity_key: str = sensitivity_key
        self.__getsizes__(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        myfile = f"{self._path}/{self.filelist[i]}"
        try:
            with h5py.File(myfile, "r") as f:
                # calling nan_to_num does seem to impact performance
                inputs = cast(h5py.Group, f["inputs"])[:]
                data = cast(h5py.Group, f["fields"])[:]
                sensitivities = cast(h5py.Group, f[self.sensitivity_key])[:] if self.sensitivity_key else None
            inputs = torch.from_numpy(inputs)
            data = torch.from_numpy(data)
            if sensitivities is not None:
                sensitivities = torch.from_numpy(sensitivities)
                scale = self.scaling.flatten()[: sensitivities.shape[0]].to(sensitivities.dtype)
                sensitivities = sensitivities * scale.view(-1, *([1] * (sensitivities.dim() - 1)))
                return inputs.view(-1, 1, 1) / self.scaling, data, sensitivities
            return inputs.view(-1, 1, 1) / self.scaling, data
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)


class CompleteDatasetOneFileSims(Dataset[Any]):
    """Many samples per hdf5 file, indexed by the filelist. With sensitivity_key set, items also hold
    f[sensitivity_key][idx], shape (k, *fields.shape): d fields / d inputs for the first k inputs."""

    def __init__(
        self,
        filelist: NDArray[np.str_],
        n_channels: int,
        path: str,
        sensitivity_key: str = "",
    ):
        self._length: int = len(filelist)
        self._path: str = path
        self.pixels_y: int = 0
        self.pixels_x: int = 0
        self.pixels_z: int = 0
        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.sensitivity_key: str = sensitivity_key
        self.__getsizes__(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        file = self.filelist[i][0]
        idx = self.filelist[i][1]
        myfile = f"{self._path}/{file}"
        try:
            with h5py.File(myfile, "r") as f:
                # calling nan_to_num does seem to impact performance
                # inputs = np.nan_to_num(f['inputs'][idx])
                # data = np.nan_to_num(f['fields'][idx])
                inputs = cast(h5py.Group, f["inputs"])[idx]
                data = cast(h5py.Group, f["fields"])[idx]
                sensitivities = cast(h5py.Group, f[self.sensitivity_key])[idx] if self.sensitivity_key else None
            inputs = torch.from_numpy(inputs)
            data = torch.from_numpy(data)
            if sensitivities is not None:
                return inputs.view(-1, 1, 1), data, torch.from_numpy(sensitivities)
            return inputs.view(-1, 1, 1), data
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)

    def __getsizes__(self, i: int) -> None:
        x, y = self.__getitem__(i)[:2]
        self.n_input = x.shape[0]
        self.pixels_y = y.shape[1]
        self.pixels_x = y.shape[2]
        if len(y.shape) == 4:
            self.pixels_z = y.shape[3]


def sensitivity_rms(dataset: Dataset[Any], indices: NDArray[np.int_], comm: Any) -> NDArray[np.float64]:
    """Rms over pixels and channels of each input's sensitivities (item [2]) of a dataset, from the
    samples at indices on each rank, summed over all ranks. Non-finite samples are skipped."""
    sums = np.zeros(0)
    for i in indices:
        sensitivities = dataset[int(i)][2].double()
        if sums.size == 0:
            sums = np.zeros(2 * sensitivities.shape[0])
        if torch.isfinite(sensitivities).all():
            k = sensitivities.shape[0]
            sums[:k] += torch.sum(torch.square(sensitivities.flatten(1)), 1).numpy()
            sums[k:] += sensitivities[0].numel()
    sums = comm.allreduce(sums)  # the default op is SUM; no mpi4py import, so the unit tests run without MPI
    k = sums.size // 2
    return np.sqrt(sums[:k] / np.maximum(sums[k:], 1))


class CompleteDatasetRandom(Dataset[Any]):
    def __init__(
        self,
        filelist: NDArray[np.str_],
        n_channels: int,
        path: str,
    ):
        self._length: int = len(filelist)
        self._path: str = path
        self.pixels_y: int = 0
        self.pixels_x: int = 0
        self.pixels_z: int = 0
        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.__getsizes__(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        myfile = f"{self._path}/{self.filelist[i]}"
        try:
            return torch.rand((6, 1, 1)), torch.rand((7, 512, 1664))
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)

    def __getsizes__(self, i: int) -> None:
        x, y = self.__getitem__(i)[:2]
        self.n_input = x.shape[0]
        self.pixels_y = y.shape[1]
        self.pixels_x = y.shape[2]
        if len(y.shape) == 4:
            self.pixels_z = y.shape[3]


def main(args: argparse.Namespace) -> None:
    # On some ROCm + MPI stacks (e.g. MI355X with MPICH), the HIP runtime cannot initialize
    # *after* MPI_Init (which mpi4py triggers on import) -- torch.cuda then reports no
    # GPUs. Initialize the GPU runtime before importing mpi4py. Both happen here rather than
    # at module import, because spawned/forkserver DataLoader workers re-import this module
    # and only read data: they should create neither a GPU context nor an MPI process.
    if torch.cuda.is_available():
        torch.cuda.init()
    from mpi4py import MPI

    # Initialize MPI
    COMM = MPI.COMM_WORLD
    rank = COMM.rank
    size = COMM.size

    # set envs
    print(f"[rank{rank}]: [professor.__version__]: {professor.__version__}")
    if rank == 0:
        hostname = socket.gethostname()
        print(professor._name)
    else:
        hostname = None

    hostname = COMM.bcast(hostname, root=0)
    print(f"[rank{rank}]: done hostname")

    os.environ["MASTER_ADDR"] = hostname
    os.environ.setdefault("MASTER_PORT", "23456")

    # wait a bit to make sure all nodes are running....
    sleep(3)

    COMM.Barrier()

    # equivalent to MPI init.
    if torch.distributed.is_initialized():
        print("Process group is already initialized.")
    else:
        print("Process group is not yet initialized.")
        torch.distributed.init_process_group(
            "nccl",
            init_method="env://",
            world_size=size,
            rank=rank,
        )

    # lookup number of ranks in the job, and our rank
    size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    assert size == COMM.Get_size()
    assert rank == COMM.rank

    # compute our local rank on the node and select a corresponding gpu,
    # this assumes we started exactly one rank per gpu on the node
    ngpus_per_node = torch.cuda.device_count()
    local_rank = rank % ngpus_per_node
    if torch.cuda.is_available():
        print(f"[rank{rank}]: ngpus_per_node {ngpus_per_node} local_rank {local_rank}")  # noqa E501
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        print(f"[rank{rank}]: Torch cuda device {torch.cuda.current_device()}")
    else:
        raise RuntimeError("No GPUs found!")

    # --- experimental performance toggles (env-gated, default off) ---
    _prof_bench = os.environ.get("PROF_CUDNN_BENCHMARK", "0") == "1"
    _prof_chlast = os.environ.get("PROF_CHANNELS_LAST", "0") == "1"
    # bf16 only: fp16 would need a gradient scaler, bf16 has the exponent range of fp32
    _prof_amp = os.environ.get("PROF_AMP", "").lower()  # "" or "bf16"
    if _prof_bench:
        torch.backends.cudnn.benchmark = True
    if _prof_amp not in ("", "bf16"):
        raise ValueError(f"PROF_AMP={_prof_amp!r} is not supported; use bf16 or leave it unset")
    _amp_dtype = torch.bfloat16 if _prof_amp == "bf16" else None
    if rank == 0:
        print(f"[perf] cudnn.benchmark={_prof_bench} channels_last={_prof_chlast} amp={_prof_amp or 'off'}")

    batch_size = args.batch_size
    batch_multiplier = getattr(args, "batch_multiplier", 1)
    if batch_multiplier < 1:
        raise ValueError("batch_multiplier must be a positive integer.")
    lr = args.lr
    num_epochs = args.num_epochs
    seed_number = args.seed
    n_checkpoint = args.n_checkpoint
    loss_target = args.loss_target
    restart_model = args.restart_model
    keys = args.keys.split(",")  # list of keys to
    n_models = len(keys)
    epoch_number = 0

    # learning rate scaling
    # batch_multiplier * ngpu * bs * base_lr
    lr = batch_multiplier * size * batch_size * lr
    if rank == 0:
        print("New effective lr=", lr)

    # Writer will output to pwd/runs/ directory by default
    mycomment = f"GEN_NGPU_{size}_LR_{lr}_Batch_{batch_size}_Loss_{loss_target}_MaxFeat_{args.max_feature}_MinFeat_{args.min_feature}_keys_{args.keys}"  # noqa
    if args.run_directory:
        print(f"[rank{rank}]: Using run directory of {args.run_directory}")
        # look for latest checkpoint file in that directory
        checkpoint_file, last_epoch = last_checkpoint(args.run_directory)
        if checkpoint_file:
            restart_model = checkpoint_file
            epoch_number = last_epoch + 1
            print(f"[rank{rank}]: Found previous checkpoint {checkpoint_file}")
            print(f"[rank{rank}]: Starting from epoch number {epoch_number}")
        writer = SummaryWriter(log_dir=args.run_directory, comment="")
        output_dir = writer.log_dir
    else:
        if rank == 0:
            writer = SummaryWriter(comment=mycomment)
            output_dir = writer.log_dir
        else:
            writer = None
            output_dir = None

    if len(restart_model) == 0:
        restart = False
    else:
        restart = True

    if rank == 0:
        if output_dir is not None:
            # check on the root rank whether the dir exisits
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            # save the args as a file in the output dir
            with open(f"{output_dir}/args.txt", "w") as f:
                my_str = str(args)
                my_str = my_str.replace(", ", ",\n")
                my_str += f"\nWorld Size: {size}\n"
                print(my_str)
                f.write(my_str)

    n_channels = n_models

    np.random.seed(seed_number + rank)

    n_sims = args.n_sims
    path = args.dataset_path

    output = "fields"

    filelist = np.genfromtxt(
        f"{path}/{args.dataset_file}",
        dtype=None,
        encoding="UTF-8",
    )

    filelist = filelist[: n_sims + 1]

    filelist_train = filelist
    if len(filelist) == 0:
        raise ValueError("Dataset filelist is empty.")
    filelist_val = np.array([filelist[min(50, len(filelist) - 1)]])

    scales = [float(x) for x in args.divide_input_scale.split(",")]
    scaling = torch.tensor(scales).float().view(-1, 1, 1)

    # the derivative (Sobolev) loss term needs the training samples' sensitivities
    sobolev_weight = getattr(args, "sobolev_weight", 0.0)
    use_sobolev = sobolev_weight > 0
    if use_sobolev and (args.dataset_type == 3 or args.compile):
        raise ValueError("--sobolev_weight > 0 needs dataset type 0, 1 or 2 and no --compile")
    sensitivity_key = getattr(args, "sobolev_key", "jacobian") if use_sobolev else ""

    if args.dataset_type == 0:
        TrainDataset = CompleteDataset(
            filelist_train,
            n_channels=n_channels,
            path=path,
            sensitivity_key=sensitivity_key,
        )
    elif args.dataset_type == 1:
        TrainDataset = CompleteDatasetOneFileSims(
            filelist_train,
            n_channels=n_channels,
            path=path,
            sensitivity_key=sensitivity_key,
        )
    elif args.dataset_type == 2:
        TrainDataset = CompleteDatasetDivideScaling(
            filelist_train,
            n_channels=n_channels,
            path=path,
            scaling=scaling,
            sensitivity_key=sensitivity_key,
        )
        print(f"[rank{rank}]: using this scaling: {scaling[:, 0, 0]}")
    elif args.dataset_type == 3:
        TrainDataset = CompleteDatasetRandom(
            filelist_train,
            n_channels=n_channels,
            path=path,
        )
    else:
        raise ValueError(f"Unrecognized dataset type: {args.dataset_type}")
    n_pixels_y = TrainDataset.pixels_y
    n_pixels_x = TrainDataset.pixels_x
    n_pixels_z = TrainDataset.pixels_z
    n_input = TrainDataset.n_input

    if args.dataset_type == 0:
        ValDataset = CompleteDataset(
            filelist_val,
            n_channels=n_channels,
            path=path,
        )
    elif args.dataset_type == 1:
        ValDataset = CompleteDatasetOneFileSims(
            filelist_val,
            n_channels=n_channels,
            path=path,
        )
    elif args.dataset_type == 2:
        ValDataset = CompleteDatasetDivideScaling(
            filelist_val,
            n_channels=n_channels,
            path=path,
            scaling=scaling,
        )
    elif args.dataset_type == 3:
        ValDataset = CompleteDatasetRandom(
            filelist_val,
            n_channels=n_channels,
            path=path,
        )
    else:
        raise Exception(f"Unrecognized dataset type: {args.dataset_type}")

    print(f"[Rank{rank}] finished setting up data loaders")
    COMM.Barrier()

    n_train = TrainDataset.__len__()

    if use_sobolev:
        _, sample_fields, sample_sensitivities = TrainDataset[0]
        n_sensitivity_inputs = sample_sensitivities.shape[0]
        if sample_sensitivities.shape[1:] != sample_fields.shape or n_sensitivity_inputs > n_input:
            raise ValueError(
                f"'{sensitivity_key}' has shape {tuple(sample_sensitivities.shape)} per sample; expected"
                f" (k, {', '.join(str(n) for n in sample_fields.shape)}) with k <= {n_input} inputs"
            )
        # tangents u_i ~ N(0, 1 / rms_i^2) weight every input's derivative error alike
        rms_samples = np.linspace(0, n_train - 1, 64 * size).astype(int)[rank::size]
        sensitivity_rms_per_input = sensitivity_rms(TrainDataset, rms_samples, COMM)
        direction_scale = torch.tensor(
            np.divide(
                1.0,
                sensitivity_rms_per_input,
                out=np.zeros_like(sensitivity_rms_per_input),
                where=sensitivity_rms_per_input > 0,
            ),
            dtype=torch.float32,
        ).to(device)
        if rank == 0:
            print(
                f"sobolev: {sobolev_weight} * derivative mse,"
                f" '{sensitivity_key}' rms per input {sensitivity_rms_per_input}"
            )

    n_pixels = max(n_pixels_y, n_pixels_x)
    n_dims = 2
    if n_pixels_z > 1:
        n_dims = 3
        n_pixels = max(n_pixels, n_pixels_z)

    finalLayer = AlphaLinear(n_channels=n_channels, n_dims=n_dims)

    # options added after the first integration tests; getattr keeps older argument namespaces working
    use_batch_norm = not getattr(args, "no_batch_norm", False)
    res_blocks = [int(b) for b in str(getattr(args, "res_blocks", "1")).split(",")]
    coord_channels = not getattr(args, "no_coord_channels", False)

    # Select generator strategy
    # Note: last_bias is not wired into the user argument parser
    model = build_generator(
        args.generator_type,
        input_size=n_input,
        im_size=n_pixels,
        num_channels=n_models,
        intermediate_channels=args.intermediate_channels,
        max_features=args.max_feature,
        min_features=args.min_feature,
        first_layer=nn.Identity(),
        last_layer=finalLayer,
        x_kernel=args.x_kernel,
        y_kernel=args.y_kernel,
        z_kernel=args.z_kernel,
        act_fun=args.act_fun,
        upscale_type=args.upscale_type,
        residual=args.residual,
        use_batch_norm=use_batch_norm,
        res_blocks=res_blocks[0] if len(res_blocks) == 1 else res_blocks,
        coord_channels=coord_channels,
        y_pixels=n_pixels_y,
        x_pixels=n_pixels_x,
    )

    if rank == 0:
        model_stats = summary(model, input_size=(batch_size, n_input, 1, 1), depth=7, device="cpu")
        with open(f"{output_dir}/model.info", "wb") as f:
            f.write(str(model_stats).encode("utf-8"))

    # broadcast parameters & optimizer state from root rank
    model.to(device)  # move to GPU
    memory_format = torch.channels_last if n_dims == 2 else torch.channels_last_3d
    if _prof_chlast:
        model = model.to(memory_format=memory_format)

    if restart:
        # remap storage from GPU 0 to local GPU
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        model.load_state_dict(torch.load(restart_model, map_location=map_location)["model"])

    print("Checkpoint and model have been loaded!")

    if args.compile:
        model = torch.compile(model)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    print(f"found {n_train} training samples")

    # Note: The default multiprocessing context can produce unexpected errors with CUDA
    # "spawn" is the safest option that is available on all systems, but is not as fast
    # "forkserver" is more efficient, but is only available on certain linux/unix systems
    multiprocessing_context = "spawn"
    if "forkserver" in multiprocessing.get_all_start_methods():
        multiprocessing_context = "forkserver"

    train_sampler = DistributedSampler(TrainDataset, shuffle=True)
    train_loader = DataLoader(
        TrainDataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.dataloader_workers,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=1,
        drop_last=os.environ.get("PROF_DROP_LAST", "0") == "1",
    )

    val_loader = torch.utils.data.DataLoader(
        ValDataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Build optimizers
    if os.environ.get("PROF_PLAIN_ADAM", "0") == "1":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=lr,
        )

    # Note: complex parameters must be handled in a separate step with the ZeroRedundancyOptimizer
    #       this option needs additional testing, so defer it to a future PR
    # real_params = [p for p in model.parameters() if p.dtype == torch.float32]
    # complex_params = [p for p in model.parameters() if p.dtype == torch.complex64]
    # optimizer = ZeroRedundancyOptimizer(
    #     real_params,
    #     optimizer_class=torch.optim.Adam,
    #     lr=lr,
    # )
    # optimizer_complex = None
    # if len(complex_params):
    #     optimizer_complex = ZeroRedundancyOptimizer(
    #         complex_params,
    #         optimizer_class=torch.optim.Adam,
    #         lr=lr,
    #     )

    if restart:
        # load optimizer state
        optimizer.load_state_dict(torch.load(restart_model)["optimizer"])
        # if optimizer_complex is not None:
        #     optimizer_complex.load_state_dict(torch.load(restart_model)["optimizer_complex"])
        print(f"[Rank{rank}] loaded previous optimizer state")

    if args.vis_config and rank == 0:
        # TODO: Estimate these values from the training data
        model_name = "Professor Model"
        model_params = [(f"parameter_{ii}", -1.0, 1.0) for ii in range(n_input)]

        build_template_config(
            config_fname=args.vis_config,
            model_name=model_name,
            checkpoint_path=f"{output_dir}/0000.pt",
            n_channels=n_channels,
            intermediate_channels=args.intermediate_channels,
            n_inputs=n_input,
            x_pixels=n_pixels_x,
            y_pixels=n_pixels_y,
            z_pixels=n_pixels_z,
            min_features=args.min_feature,
            max_features=args.max_feature,
            x_kernel=args.x_kernel,
            y_kernel=args.y_kernel,
            z_kernel=args.z_kernel,
            input_parameters=model_params,
            fields=keys,
            generator_type=args.generator_type,
            act_fun=args.act_fun,
            upscale_type=args.upscale_type,
            residual=args.residual,
            use_batch_norm=use_batch_norm,
            res_blocks=res_blocks,
            coord_channels=coord_channels,
        )

    # try to free up gpu memory
    torch.cuda.empty_cache()

    grad_weight = getattr(args, "grad_weight", 0.0)
    pyr_weight = getattr(args, "pyr_weight", 0.0)
    pyr_levels = getattr(args, "pyr_levels", 5)

    def MyLoss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return field_loss(output, target, loss_target, grad_weight, pyr_weight, pyr_levels)

    def ForwardLoss(
        data: torch.Tensor, target: torch.Tensor, sensitivities: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """model output, total loss and the Sobolev term (0 if off) for one training batch;
        the datasets return the sensitivities only when the Sobolev term is on"""
        if sensitivities is None:
            output = model(data)
            return output, MyLoss(output, target), torch.zeros((), device=device)
        sensitivities = sensitivities.to(device)
        direction = torch.randn(data.shape[0], n_sensitivity_inputs, device=device) * direction_scale
        tangent = torch.zeros_like(data)
        tangent[:, :n_sensitivity_inputs, 0, 0] = direction
        # forward-mode AD skips autocast (fp32 dual input against bf16 weights), so this step runs in fp32
        with torch.autocast("cuda", enabled=False):
            output, output_derivative = forward_jacobian_vector_product(model, data, tangent)
        loss_sobolev = sobolev_loss(output_derivative, sensitivities, direction)
        return output, MyLoss(output, target) + sobolev_weight * loss_sobolev, loss_sobolev.detach()

    if rank == 0:
        print(
            f"loss: {loss_target} + {grad_weight} * pixel differences + {pyr_weight} * pyramid ({pyr_levels} levels)"
        )

    n_train_loader = len(train_loader)
    n_train_data = size * batch_size * n_train_loader * n_channels * n_pixels_y * n_pixels_x  # noqa E:501
    print(f"Number of steps in one epoch: {n_train_loader}")
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=args.profile_memory,
        )
        if args.profile
        else contextlib.nullcontext()
    ) as prof:
        for epoch in range(epoch_number, num_epochs):
            with record_function("epoch_setup"):
                train_sampler.set_epoch(epoch)
                t0 = time()
                test_loss = 0.0
                test_mse = 0.0
                test_l8 = 0.0
                train_con = 0.0
                test_con = 0.0
                # Training metrics are accumulated on-GPU during the epoch and
                # copied to the host exactly once (after the step loop). This
                # avoids a device->host sync on every step, which otherwise
                # stalls the pipeline. Values are identical to the per-step sum.
                train_loss_gpu = torch.zeros((), device=device)
                train_mse_gpu = torch.zeros((), device=device)
                train_l8_gpu = torch.zeros((), device=device)
                training_losses_gpu = torch.zeros(n_models, device=device)
                train_sobolev_gpu = torch.zeros((), device=device)
            with record_function("epoch_iteration"):
                optimizer.zero_grad(set_to_none=True)
                # if optimizer_complex is not None:
                #     optimizer_complex.zero_grad(set_to_none=True)

                for batch_idx, (data, target, *sensitivities) in enumerate(train_loader):
                    with record_function("optimizer_step"):
                        should_update_weights = (batch_idx + 1) % batch_multiplier == 0
                        if batch_idx + 1 == n_train_loader:
                            should_update_weights = True

                        group_start = (batch_idx // batch_multiplier) * batch_multiplier
                        group_end = min(group_start + batch_multiplier, n_train_loader)
                        accumulation_group_size = group_end - group_start

                        data = data.to(device)
                        target = target.to(device)
                        if _prof_chlast:
                            data = data.to(memory_format=memory_format)
                            target = target.to(memory_format=memory_format)

                        sync_context = contextlib.nullcontext()
                        if not should_update_weights:
                            sync_context = model.no_sync()

                        with sync_context:
                            with torch.autocast("cuda", dtype=_amp_dtype, enabled=_amp_dtype is not None):
                                output, loss, loss_sobolev = ForwardLoss(data, target, *sensitivities)
                            if use_sobolev:
                                train_sobolev_gpu += loss_sobolev
                            scaled_loss = loss / accumulation_group_size
                            scaled_loss.backward()

                        if should_update_weights:
                            optimizer.step()
                            # if optimizer_complex is not None:
                            #     optimizer_complex.step()

                            optimizer.zero_grad(set_to_none=True)
                            # if optimizer_complex is not None:
                            #     optimizer_complex.zero_grad(set_to_none=True)

                    with record_function("other_losses"):
                        with torch.no_grad():
                            losses = consolidated_loss(
                                output.float(), target, loss_types=["l1-sum", "l2-sum", "linf", "sum-per-channel"]
                            )
                            assert isinstance(losses, dict)
                            # accumulate on-device; no host sync inside the loop
                            training_losses_gpu += losses["sum-per-channel"]
                            train_loss_gpu += losses["l1-sum"]
                            train_mse_gpu += losses["l2-sum"]
                            train_l8_gpu = torch.maximum(train_l8_gpu, losses["linf"])
                    if args.profile and (prof is not None):
                        prof.step()
                    # end of a step in the epoch
                t1 = time()

                # single device->host sync per epoch for the accumulated metrics
                train_loss = train_loss_gpu.item()
                train_mse = train_mse_gpu.item()
                train_l8 = train_l8_gpu.item()
                training_losses = training_losses_gpu.cpu()
                if use_sobolev:
                    train_sobolev = COMM.allreduce(train_sobolev_gpu.item(), op=MPI.SUM) / (size * n_train_loader)

                with record_function("communicate_losses_to_rank0"):
                    training_sums = np.array(
                        (
                            train_loss,
                            test_loss,
                            train_mse,
                            test_mse,
                            train_con,
                            test_con,
                            t1 - t0,
                        )
                    )
                    training_losses_np = np.array([training_losses[i].item() for i in range(n_models)])
                    training_sums = np.hstack((training_sums, training_losses_np))
                    training_maxs = np.array((train_l8, test_l8))

                    COMM.Allreduce(MPI.IN_PLACE, training_sums, op=MPI.SUM)
                    COMM.Allreduce(MPI.IN_PLACE, training_maxs, op=MPI.MAX)

                    if rank == 0:
                        history = {
                            "MAE/train": training_sums[0] / n_train_data,
                            "MAE/test": training_sums[1],
                            "MSE/train": training_sums[2] / n_train_data,
                            "MSE/test": training_sums[3],
                            "Continuity-Eqn/train": training_sums[4],
                            "L-infinity/train": training_maxs[0],
                            "L-infinity/test": training_maxs[1],
                            "Epoch-Time": t1 - t0,
                        }
                        for model_idx in range(n_models):
                            history[f"TrainMAE/{keys[model_idx]}"] = training_sums[7 + model_idx] / n_train_data  # noqa E:501
                        if use_sobolev:
                            history["Sobolev/train"] = train_sobolev
                        train_log = "{:04} {:04f} {:04f} {:04f} {:04f} {:04f} {:04f} {:04f} {:04f} {:04f}".format(  # noqa E:501
                            epoch,
                            history["MAE/train"],
                            0.0,
                            history["MSE/train"],
                            0.0,
                            history["L-infinity/train"],
                            0.0,
                            training_sums[4],
                            training_sums[5],
                            history["Epoch-Time"],
                        )
                        print(train_log)
                        if use_sobolev:
                            print(f"{epoch:04} sobolev {train_sobolev:04f}")

                        if writer is not None:
                            for key in history:
                                writer.add_scalar(key, history[key], epoch)

                if epoch % n_checkpoint == 0 or epoch == num_epochs - 1:
                    if hasattr(optimizer, "consolidate_state_dict"):
                        optimizer.consolidate_state_dict()
                    # if optimizer_complex is not None:
                    #     optimizer_complex.consolidate_state_dict()

                    if rank == 0:
                        filepath = f"{output_dir}/{epoch:04d}.pt"
                        state = {"optimizer": optimizer.state_dict()}
                        # if optimizer_complex is not None:
                        #     state["optimizer_complex"] = optimizer_complex.state_dict()

                        state["model"] = model.module.state_dict()
                        torch.save(state, filepath)

                        # Update the autogenerated config with the new checkpoint
                        if args.vis_config:
                            try:
                                update_config_checkpoint(
                                    config_fname=args.vis_config,
                                    checkpoint_path=filepath,
                                )
                            except Exception as e:
                                print(f"[Rank{rank}] Failed to update config: {repr(e)}")

                    # get val images and save them with tensorboard
                    val_images = []
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data = data.to(device)
                        with model.no_sync():
                            with torch.no_grad():
                                output = model(data).cpu()
                    # print(output.shape)
                    if not isinstance(output, torch.Tensor):
                        raise Exception(f"Model returned an unexpected value: {type(output)}")

                    if n_dims == 2:
                        val_images = output.view(1, n_channels, n_pixels_y, n_pixels_x)
                    elif n_dims == 3:
                        val_images = output.view(1, n_channels, n_pixels_y, n_pixels_x, n_pixels_z)
                    else:
                        raise Exception(f"Invalid number of dimensions: {n_dims}")

                    # need to convert this to 0 - 1 output
                    val_flat = output.view(n_channels, -1)
                    d_min = val_flat.min(dim=1)[0]
                    d_max = val_flat.max(dim=1)[0]
                    # print(d_min.shape)
                    for key_ind, key in enumerate(keys):
                        val_images[:, key_ind] -= d_min[key_ind]
                        val_images[:, key_ind] /= d_max[key_ind] - d_min[key_ind]  # noqa E501
                    if (rank == 0) and (writer is not None):
                        for key_ind, key in enumerate(keys):
                            img_view = []
                            if n_dims == 2:
                                img_view = val_images[:, key_ind].view(
                                    -1,
                                    1,
                                    n_pixels_y,
                                    n_pixels_x,
                                )
                            elif n_dims == 3:
                                # For 3D records, save the mid-point of the z dim
                                img_view = val_images[:, key_ind, :, :, n_pixels_z // 2].view(
                                    -1,
                                    1,
                                    n_pixels_y,
                                    n_pixels_x,
                                )

                            writer.add_images(
                                key,
                                img_view,
                                epoch,
                            )

    if rank == 0:
        if args.profile and (prof is not None):
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            prof.export_chrome_trace(f"{output_dir}/chrome_prof-trainer_size_{size}.json.gz")
            if args.profile_memory:
                prof.export_memory_timeline(f"{output_dir}/memory_prof-trainer_size_{size}.json.gz")
        if writer is not None:
            writer.close()

    print(f"[Rank{rank}]: finished")
    COMM.Barrier()
    torch.distributed.destroy_process_group()


# Note: keeping this entry point in case it is used in any workflows
if __name__ == "__main__":
    from professor.__main__ import prof_trainer

    prof_trainer()
