import atexit
import json
import os
import time
from enum import StrEnum
from pathlib import Path
from socket import gethostname
from typing import Any

import torch
from mpi4py import MPI


class Event(StrEnum):
    RUN_CONFIG = "run_config"
    EPOCH_START = "epoch_start"
    FIRST_BATCH = "first_batch"
    STEP = "step"
    EPOCH_END = "epoch_end"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, StrEnum):
        return obj.value

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


class TrainerMetricsLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        prefix: str = "metrics",
        rank: int | None = None,
        world_size: int | None = None,
        sync_timing: bool = False,
        flush_every: int = 50,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.sync_timing = sync_timing
        self.flush_every = max(1, int(flush_every))
        self._writes_since_last_flush = 0
        self._file_handle = None

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank() if rank is None else int(rank)
        self.world_size = comm.Get_size() if world_size is None else int(world_size)
        self.hostname = gethostname()

        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"

        if self.enabled:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.filename = self.metrics_dir / f"{prefix}_rank{self.rank:04d}.jsonl"
            self._file_handle = self.filename.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            atexit.register(self.close)
        else:
            self.filename = None

    def close(self) -> None:
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.flush()
            self._file_handle.close()

    def __enter__(self) -> "TrainerMetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def synchronize(self) -> None:
        if not self.sync_timing:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def now(self) -> float:
        self.synchronize()
        return time.perf_counter()

    def elapsed(self, start: float) -> float:
        self.synchronize()
        return time.perf_counter() - start

    def log(self, event: str | Event, **kwargs: Any) -> None:
        if not self.enabled or self._file_handle is None:
            return

        payload = {
            "event": event,
            "timestamp_unix": time.time(),
            "perf_counter_s": time.perf_counter(),
            "world_size": self.world_size,
            "global_rank": self.rank,
            "hostname": self.hostname,
            **kwargs,
        }

        self._file_handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        self._file_handle.write("\n")

        self._writes_since_last_flush += 1
        if self._writes_since_last_flush >= self.flush_every:
            self._file_handle.flush()
            self._writes_since_last_flush = 0

    def log_run_config(self, **kwargs: Any) -> None:
        self.log(
            Event.RUN_CONFIG,
            pid=os.getpid(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            rocr_visible_devices=os.environ.get("ROCR_VISIBLE_DEVICES", ""),
            slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
            slurm_nnodes=os.environ.get("SLURM_JOB_NUM_NODES", ""),
            slurm_ntasks=os.environ.get("SLURM_NTASKS", ""),
            slurm_procid=os.environ.get("SLURM_PROCID", ""),
            slurm_localid=os.environ.get("SLURM_LOCALID", ""),
            flux_job_id=os.environ.get("FLUX_JOB_ID", ""),
            flux_task_rank=os.environ.get("FLUX_TASK_RANK", ""),
            flux_task_local_id=os.environ.get("FLUX_TASK_LOCAL_ID", ""),
            **kwargs,
        )

    def log_epoch_start(self, *, epoch: int, **kwargs: Any) -> None:
        self.log(Event.EPOCH_START, epoch=int(epoch), **kwargs)

    def log_first_batch(
        self,
        *,
        epoch: int,
        time_to_first_batch_s: float,
        **kwargs: Any,
    ) -> None:
        self.log(
            Event.FIRST_BATCH,
            epoch=int(epoch),
            time_to_first_batch_s=float(time_to_first_batch_s),
            **kwargs,
        )

    def log_step(
        self,
        *,
        epoch: int,
        step: int,
        dataloader_wait_s: float,
        step_time_s: float,
        batch_size: int,
        **kwargs: Any,
    ) -> None:
        self.log(
            Event.STEP,
            epoch=int(epoch),
            step=int(step),
            dataloader_wait_s=float(dataloader_wait_s),
            step_time_s=float(step_time_s),
            batch_size=int(batch_size),
            **kwargs,
        )

    def log_epoch_end(
        self,
        *,
        epoch: int,
        epoch_time_s: float,
        local_samples: int,
        **kwargs: Any,
    ) -> None:
        local_samples_per_sec = float(local_samples) / float(epoch_time_s) if epoch_time_s > 0 else None

        self.log(
            Event.EPOCH_END,
            epoch=int(epoch),
            epoch_time_s=float(epoch_time_s),
            local_samples=int(local_samples),
            local_samples_per_sec=local_samples_per_sec,
            **kwargs,
        )
