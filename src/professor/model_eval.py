from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def render_model_evaluation_cases(
    model_path: str,
    dataset_path: str,
    output_path: str,
    n_samples: int,
    dataset_file: str = "filelist.txt",
    dataset_type: int = 0,
    random_seed: int | None = None,
    divide_input_scale: str = "1,1,1,1",
    colormap: str = "turbo",
) -> None:
    """
    Render observed, predicted, and difference images for sampled dataset cases.

    Args:
        model_path: Path to a Professor/Vela model configuration YAML.
        dataset_path: Path containing the dataset and dataset file list.
        output_path: Output directory for PNG files.
        n_samples: Number of random cases to render.
        dataset_file: File list name under ``dataset_path``.
        dataset_type: Dataset type matching ``prof-trainer``.
        random_seed: Optional random seed for selecting case IDs.
        divide_input_scale: Comma-separated input scales for dataset type 2.
        colormap: Matplotlib colormap for observed and predicted images.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from professor.mltrainer import (
        CompleteDataset,
        CompleteDatasetDivideScaling,
        CompleteDatasetOneFileSims,
        CompleteDatasetRandom,
    )
    from professor.vela import config, model

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    cfg = config.Config(os.path.expanduser(model_path))
    n_channels = len(cfg.fields)

    filelist = np.genfromtxt(
        f"{dataset_path}/{dataset_file}",
        dtype=None,
        encoding="UTF-8",
    )
    if filelist.ndim == 0:
        filelist = filelist.reshape(1)
    if dataset_type == 1 and filelist.ndim == 1 and len(filelist) == 2:
        filelist = filelist.reshape(1, 2)
    if len(filelist) == 0:
        raise ValueError("Dataset filelist is empty.")

    scales = [float(x) for x in divide_input_scale.split(",")]
    scaling = torch.tensor(scales).float().view(-1, 1, 1)

    dataset = None
    if dataset_type == 0:
        dataset = CompleteDataset(filelist, n_channels=n_channels, path=dataset_path)
    elif dataset_type == 1:
        dataset = CompleteDatasetOneFileSims(filelist, n_channels=n_channels, path=dataset_path)
    elif dataset_type == 2:
        dataset = CompleteDatasetDivideScaling(
            filelist,
            n_channels=n_channels,
            path=dataset_path,
            scaling=scaling,
        )
    elif dataset_type == 3:
        dataset = CompleteDatasetRandom(filelist, n_channels=n_channels, path=dataset_path)
    else:
        raise ValueError(f"Unrecognized dataset type: {dataset_type}")

    N = min(len(dataset), n_samples)
    rng = np.random.default_rng(random_seed)
    selected_case_ids = rng.choice(len(dataset), size=N, replace=False)

    loader = DataLoader(
        Subset(dataset, selected_case_ids.tolist()),
        batch_size=1,
        shuffle=False,
    )

    prof_model = model.PyTorchModel(cfg)
    model_collection = prof_model.models[0]
    model_ref = model_collection.ref.eval()
    device = model_collection.device
    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), open(os.path.join(output_dir, "params.csv"), "w") as record_file:
        cases = zip(selected_case_ids, loader)
        for case_id, (inputs, observed) in tqdm(cases, total=N, desc="Rendering cases"):
            record_file.write(",".join([str(x) for x in inputs]) + "\n")
            inputs = inputs.to(device)
            if cfg.half_precision:
                inputs = inputs.half()
            predicted = model_ref(inputs).detach().cpu().float().numpy()[0]

            observed_array = observed.detach().cpu().float().numpy()[0]
            case_label = f"case_{case_id:06d}"
            _save_case_figures(
                observed=observed_array,
                predicted=predicted,
                fields=cfg.fields,
                case_label=case_label,
                colormap=colormap,
                output_dir=output_dir,
            )


def prof_eval(argv: Sequence[str] | None = None) -> None:
    """
    Console entry point for rendering model evaluation samples.
    """
    parser = argparse.ArgumentParser(
        prog="prof-eval",
        description="Render observed, predicted, and difference images for sampled Professor cases.",
    )
    parser.add_argument("model_path", type=str, help="Professor/Vela model configuration YAML.")
    parser.add_argument("-n", "--n-samples", type=int, default=4, help="Number of random cases to render.")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output directory for PNG files.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for case selection.")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, required=True)
    parser.add_argument("--dataset-file", "--dataset_file", dest="dataset_file", type=str, default="filelist.txt")
    parser.add_argument("--dataset-type", "--dataset_type", dest="dataset_type", type=int, default=0)
    parser.add_argument(
        "--divide-input-scale",
        "--divide_input_scale",
        dest="divide_input_scale",
        type=str,
        default="1,1,1,1",
    )
    parser.add_argument("--colormap", type=str, default="turbo")
    args = parser.parse_args(argv)

    render_model_evaluation_cases(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        n_samples=args.n_samples,
        dataset_file=args.dataset_file,
        dataset_type=args.dataset_type,
        random_seed=args.seed,
        divide_input_scale=args.divide_input_scale,
        colormap=args.colormap,
    )


def _save_case_figures(
    observed: np.ndarray,
    predicted: np.ndarray,
    fields: Sequence[str],
    case_label: str,
    colormap: str,
    output_dir: Path,
) -> None:
    if observed.ndim == 4:
        rows = _case_image_rows(observed, predicted, fields)
        for row_label, obs_slice, pred_slice in rows:
            fig = _build_image_triplet_figure(
                observed=obs_slice,
                predicted=pred_slice,
                row_label=row_label,
                title=f"Professor evaluation: {case_label} {row_label}",
                colormap=colormap,
            )
            fig.savefig(output_dir / f"{case_label}_{_filename_token(row_label)}.png")
            plt.close(fig)
        return

    fig = _build_case_figure(
        observed=observed,
        predicted=predicted,
        fields=fields,
        case_label=case_label,
        colormap=colormap,
    )
    fig.savefig(output_dir / f"{case_label}.png")
    plt.close(fig)


def _build_case_figure(
    observed: np.ndarray,
    predicted: np.ndarray,
    fields: Sequence[str],
    case_label: str,
    colormap: str,
):
    if observed.shape != predicted.shape:
        raise ValueError(f"Observed shape {observed.shape} does not match predicted shape {predicted.shape}.")

    rows = _case_image_rows(observed, predicted, fields)
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, max(3, 2.8 * len(rows))),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"Professor evaluation: {case_label}")

    for axes_row, (row_label, obs_slice, pred_slice) in zip(axes, rows):
        _plot_image_triplet(
            axes=axes_row,
            observed=obs_slice,
            predicted=pred_slice,
            row_label=row_label,
            colormap=colormap,
        )

    return fig


def _build_image_triplet_figure(
    observed: np.ndarray,
    predicted: np.ndarray,
    row_label: str,
    title: str,
    colormap: str,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title)
    _plot_image_triplet(
        axes=axes[0],
        observed=observed,
        predicted=predicted,
        row_label=row_label,
        colormap=colormap,
    )
    return fig


def _case_image_rows(
    observed: np.ndarray,
    predicted: np.ndarray,
    fields: Sequence[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if observed.shape != predicted.shape:
        raise ValueError(f"Observed shape {observed.shape} does not match predicted shape {predicted.shape}.")

    n_channels = observed.shape[0]

    if observed.ndim == 3:
        return [(fields[channel], observed[channel], predicted[channel]) for channel in range(n_channels)]

    if observed.ndim != 4:
        raise ValueError(f"Expected observed data with 3 or 4 dimensions, got shape {observed.shape}.")

    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for channel in range(n_channels):
        for plane_name, obs_slice, pred_slice in _midplane_slices(observed[channel], predicted[channel]):
            rows.append((f"{fields[channel]} {plane_name}", obs_slice, pred_slice))
    return rows


def _midplane_slices(observed: np.ndarray, predicted: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    shape = observed.shape
    ii = shape[0] // 2
    jj = shape[1] // 2
    kk = shape[2] // 2
    return [
        ("YZ", observed[ii, :, :], predicted[ii, :, :]),
        ("XZ", observed[:, jj, :], predicted[:, jj, :]),
        ("XY", observed[:, :, kk], predicted[:, :, kk]),
    ]


def _filename_token(label: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", label.lower())
    return token.strip("_")


def _plot_image_triplet(
    axes: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    row_label: str,
    colormap: str,
) -> None:
    diff = predicted - observed
    # image_min = min(float(np.nanmin(observed)), float(np.nanmin(predicted)))
    # image_max = max(float(np.nanmax(observed)), float(np.nanmax(predicted)))
    image_min = float(np.nanmin(observed))
    image_max = float(np.nanmax(observed))
    diff_limit = float(np.nanmax(np.abs(diff)))
    if diff_limit == 0.0:
        diff_limit = 1.0

    images = [
        ("Observed", observed, colormap, image_min, image_max),
        ("Predicted", predicted, colormap, image_min, image_max),
        ("Predicted - Observed", diff, "RdBu_r", -diff_limit, diff_limit),
    ]

    for axis, (title, image, cmap, vmin, vmax) in zip(axes, images):
        rendered = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        plt.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(row_label)
