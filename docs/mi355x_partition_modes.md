# MI355X GPU Partition Modes and prof-trainer

The AMD Instinct MI355X supports splitting each physical GPU into multiple
compute partitions (SPX = whole GPU, DPX = 2, QPX = 4, CPX = 8) combined with
a memory-NUMA setting (NPS1 / NPS2). On the Vultr "lux" Slurm cluster the mode
is selected per job via `--comment`, e.g.:

```bash
sbatch --partition=lux --nodes=1 --exclusive \
  --gres=gpu:amd_instinct_mi355_oam:8 \
  --comment=amd_gpu_mode=cpx,nps1 ...
```

The prolog applies the mode before the job starts and the epilog resets the
node to SPX/NPS1 afterward.

This note records how these modes affect `prof-trainer` throughput, measured on
the analytical Fourier-modes example (512x512, fp32) with the optimized stack
(`MIOPEN_FIND_MODE=2` + `PROF_DROP_LAST=1` + `--compile`).

## TL;DR

- **CPX/NPS1 is a good optimization for a SINGLE GPU** (~+10% throughput vs the
  default SPX), because 8 concurrent smaller-batch partitions fill one physical
  GPU better than a single large-batch stream for this ConvTranspose model.
- **CPX (and partitioning in general) is NOT good for MULTI-GPU** training. Use
  **SPX (whole GPUs) for any DDP run across physical GPUs.** CPX's per-GPU edge
  reverses at 4 GPUs and fails outright at 8 GPUs (see below).
- **NPS1 vs NPS2 is within noise** (<=2%) for this workload; NPS1 slightly
  favored.

## Single physical GPU: throughput by mode

Whole physical GPU used in each mode (data-parallel across its partitions),
largest batch that fits each partition's memory. Aggregate images/s per GPU.

| Mode      | partitions | batch/part | s/epoch | img/s | vs SPX |
|-----------|-----------:|-----------:|--------:|------:|-------:|
| SPX/NPS1  |          1 |        255 |   27.33 |   532 |   base |
| DPX/NPS1  |          2 |        255 |   25.35 |   563 |  +5.9% |
| DPX/NPS2  |          2 |        255 |   25.50 |   560 |  +5.3% |
| QPX/NPS1  |          4 |        208 |   25.62 |   552 |  +3.8% |
| QPX/NPS2  |          4 |        208 |   25.33 |   558 |  +5.0% |
| **CPX/NPS1** |       8 |         96 |   24.98 | **584** | **+9.8%** |
| CPX/NPS2  |          8 |         96 |   25.38 |   575 |  +8.1% |

## Multi-GPU: CPX/NPS1 does not scale

CPX/NPS1 across physical GPUs (each = 8 partitions), vs SPX at the same GPU
count:

| Physical GPUs | SPX img/s | CPX/NPS1 img/s | CPX vs SPX |
|--------------:|----------:|---------------:|-----------:|
| 1             |       532 |            585 |      +10%  |
| 4             |      2043 |           1418 |      -31%  |
| 8             |      3797 | segfault (fails) |     --    |

- **4 GPUs (32 partitions):** CPX reaches only 1418 img/s (2.43x its 1-GPU rate,
  61% efficiency) vs SPX's 2043 (3.86x, 96%). With 32 ranks the DDP all-reduce
  fan-out and per-epoch overhead dominate (on this small dataset each rank only
  runs ~4 steps/epoch).
- **8 GPUs (64 partitions):** segfaults (SIGSEGV) at the first training step
  after all 64 ranks initialize. Reproduced with and without `--compile` and at
  batch 96 and 48, so it is a 64-way DDP/RCCL scaling limit, not a memory or
  compile issue.

By contrast, SPX (whole GPUs) scales near-linearly: 3.86x at 4 GPUs and 7.17x
at 8 GPUs.

## Recommendation

- **Single-GPU jobs:** prefer **CPX/NPS1** for ~10% more throughput.
- **Multi-GPU / DDP jobs:** use **SPX/NPS1** (the default). Do not use CPX.

Note: the 4-GPU CPX penalty is partly a strong-scaling artifact of the small
analytical dataset (few steps/rank at 32-way); it may narrow on larger
datasets. The 8-GPU (64-way) segfault, however, is dataset-independent.
