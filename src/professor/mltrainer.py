# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import glob
import argparse
import socket
import contextlib
import numpy as np
from numpy.typing import NDArray
import h5py
from mpi4py import MPI  # MUST BE IMPORTED BEFORE torch!
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import professor
from professor.utils import consolidated_loss
from professor.layers import AlphaLinear
from professor.torch_models import Generator
from professor.vela.config import build_template_config, update_config_checkpoint
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from time import time, sleep
from torchinfo import summary
from typing import cast, Tuple, Any


class CompleteDataset(Dataset[Any]):
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
        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.__getsizes__(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        myfile = f"{self._path}/{self.filelist[i]}"
        try:
            with h5py.File(myfile, "r") as f:
                # calling nan_to_num does seem to impact performance
                inputs = cast(h5py.Group, f["inputs"])[:]
                data = cast(h5py.Group, f["fields"])[:]
            inputs = torch.from_numpy(inputs)
            data = torch.from_numpy(data)
            return inputs.view(-1, 1, 1), data
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)

    def __getsizes__(self, i: int) -> None:
        x, y = self.__getitem__(i)
        self.n_input = x.shape[0]
        self.pixels_y = y.shape[1]
        self.pixels_x = y.shape[2]


class CompleteDatasetDivideScaling(CompleteDataset):
    def __init__(self, filelist: NDArray[np.str_], n_channels: int, path: str, scaling: torch.Tensor) -> None:
        self._length: int = len(filelist)
        self._path: str = path
        self.pixels_y: int = 0
        self.pixels_x: int = 0
        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.scaling: torch.Tensor = scaling
        self.__getsizes__(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        myfile = f"{self._path}/{self.filelist[i]}"
        try:
            with h5py.File(myfile, "r") as f:
                # calling nan_to_num does seem to impact performance
                inputs = cast(h5py.Group, f["inputs"])[:]
                data = cast(h5py.Group, f["fields"])[:]
            inputs = torch.from_numpy(inputs)
            data = torch.from_numpy(data)
            return inputs.view(-1, 1, 1) / self.scaling, data
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)


class CompleteDatasetOneFileSims(Dataset[Any]):
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
        self.n_channels: int = n_channels
        self.filelist: NDArray[np.str_] = filelist
        self.n_input: int = 0
        self.__getsizes__(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
            inputs = torch.from_numpy(inputs)
            data = torch.from_numpy(data)
            return inputs.view(-1, 1, 1), data
        except Exception as e:
            error = f"Exception {e} while processing {myfile}"
            raise RuntimeError(error)

    def __getsizes__(self, i: int) -> None:
        x, y = self.__getitem__(i)
        self.n_input = x.shape[0]
        self.pixels_y = y.shape[1]
        self.pixels_x = y.shape[2]


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
        x, y = self.__getitem__(i)
        self.n_input = x.shape[0]
        self.pixels_y = y.shape[1]
        self.pixels_x = y.shape[2]


def main(args: argparse.Namespace) -> None:
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
    os.environ["MASTER_PORT"] = "23456"

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

    batch_size = args.batch_size
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
    # accumulate_grad_batches * ngpu * bs * base_lr
    accumulate_grad_batches = 1
    lr = accumulate_grad_batches * size * batch_size * lr
    if rank == 0:
        print("New effective lr=", lr)

    # Writer will output to pwd/runs/ directory by default
    mycomment = f"GEN_NGPU_{size}_LR_{lr}_Batch_{batch_size}_Loss_{loss_target}_MaxFeat_{args.max_feature}_MinFeat_{args.min_feature}_keys_{args.keys}"  # noqa
    if args.run_directory:
        print(f"[rank{rank}]: Using run directory of {args.run_directory}")
        # look for latest checkpoint file in that directory
        checkpoints = glob.glob(f"{args.run_directory}/*.pt")
        checkpoints.sort()
        if checkpoints:
            checkpoint_file = (
                checkpoints[-1]
                .replace(
                    f"{args.run_directory}",
                    "",
                )
                .strip("/")
            )
            print(checkpoint_file)
            restart_model = f"{args.run_directory}/{checkpoint_file}"
            epoch_number = int(checkpoint_file.strip(".pt")) + 1
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

    if args.dataset_type == 0:
        TrainDataset = CompleteDataset(
            filelist_train,
            n_channels=n_channels,
            path=path,
        )
    elif args.dataset_type == 1:
        TrainDataset = CompleteDatasetOneFileSims(
            filelist_train,
            n_channels=n_channels,
            path=path,
        )
    elif args.dataset_type == 2:
        TrainDataset = CompleteDatasetDivideScaling(
            filelist_train,
            n_channels=n_channels,
            path=path,
            scaling=scaling,
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

    n_pixels = max(n_pixels_y, n_pixels_x)

    finalLayer = AlphaLinear(n_channels=n_channels)

    model = Generator(
        input_size=n_input,
        im_size=n_pixels,
        num_channels=n_models,
        max_features=args.max_feature,
        min_features=args.min_feature,
        first_layer=nn.Identity(),
        last_layer=finalLayer,
        y_kernel=args.y_kernel,
        x_kernel=args.x_kernel,
    )

    if rank == 0:
        model_stats = summary(
            model,
            input_size=(batch_size, n_input, 1, 1),
            depth=7,
        )
        with open(f"{output_dir}/model.info", "wb") as f:
            f.write(str(model_stats).encode("utf-8"))

    # broadcast parameters & optimizer state from root rank
    model.to(device)  # move to GPU

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

    train_sampler = DistributedSampler(TrainDataset, shuffle=True)
    train_loader = DataLoader(
        TrainDataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=args.dataloader_workers,
        prefetch_factor=1,
    )

    val_loader = torch.utils.data.DataLoader(
        ValDataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=lr,
    )
    if restart:
        # load optimizer state
        optimizer.load_state_dict(torch.load(restart_model)["optimizer"])
        print(f"[Rank{rank}] loaded previous optimizer state")

    if args.vis_config:
        # TODO: Estimate these values from the training data
        model_name = "Professor Model"
        model_params = [(f"parameter_{ii}", -1.0, 1.0) for ii in range(n_input)]

        build_template_config(
            config_fname=args.vis_config,
            model_name=model_name,
            checkpoint_path=f"{output_dir}/0000.pt",
            n_channels=n_channels,
            n_inputs=n_input,
            x_pixels=n_pixels_x,
            y_pixels=n_pixels_y,
            min_features=args.min_feature,
            max_features=args.max_feature,
            x_kernel=args.x_kernel,
            y_kernel=args.y_kernel,
            input_parameters=model_params,
        )

    # try to free up gpu memory
    torch.cuda.empty_cache()

    MAE = torch.nn.L1Loss()
    MSE = torch.nn.MSELoss()
    if loss_target == "l1":
        MyLoss = MAE
    else:
        MyLoss = MSE

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
                train_loss = 0.0
                test_loss = 0.0
                train_mse = 0.0
                test_mse = 0.0
                train_l8 = 0.0
                test_l8 = 0.0
                train_con = 0.0
                test_con = 0.0
                training_losses = torch.zeros(n_models)
            with record_function("epoch_iteration"):
                for batch_idx, (data, target) in enumerate(train_loader):
                    with record_function("optimizer_step"):
                        optimizer.zero_grad()
                        data = data.to(device)
                        target = target.to(device)
                        output = model(data)
                        loss = MyLoss(output, target)
                        loss.backward()
                        optimizer.step()
                    with record_function("other_losses"):
                        with torch.no_grad():
                            losses = consolidated_loss(
                                output, target, loss_types=["l1-sum", "l2-sum", "linf", "sum-per-channel"]
                            )
                            assert isinstance(losses, dict)
                            training_losses += losses["sum-per-channel"].cpu()
                            train_loss += losses["l1-sum"].item()
                            train_mse += losses["l2-sum"].item()
                            train_l8 = max(
                                losses["linf"].item(),
                                train_l8,
                            )
                    if args.profile and (prof is not None):
                        prof.step()
                    # end of a step in the epoch
                t1 = time()

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

                        if writer is not None:
                            for key in history:
                                writer.add_scalar(key, history[key], epoch)

                if epoch % n_checkpoint == 0 or epoch == num_epochs - 1:
                    optimizer.consolidate_state_dict()
                    if rank == 0:
                        filepath = f"{output_dir}/{epoch:04d}.pt"
                        state = {"optimizer": optimizer.state_dict()}
                        state["model"] = model.module.state_dict()
                        torch.save(state, filepath)

                        # Update the autogenerated config with the new checkpoint
                        if args.vis_config:
                            update_config_checkpoint(
                                config_fname=args.vis_config,
                                checkpoint_path=filepath,
                            )

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
                    val_images = output.view(1, n_channels, n_pixels_y, n_pixels_x)
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
                            writer.add_images(
                                key,
                                val_images[:, key_ind].view(
                                    -1,
                                    1,
                                    n_pixels_y,
                                    n_pixels_x,
                                ),
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
