import os
import argparse
import numpy as np
import h5py
import yaml
from tqdm import tqdm
from scipy.stats import qmc
import cutde.halfspace as solver


def get_fault_corners(x0: float, y0: float, z0: float, strike: float, dip: float, length: float, width: float) -> np.ndarray:
    """
    Build the four corners of a rectangular fault surface
    """
    # Top-Right (Along Strike)
    x1 = x0 + length * np.sin(np.radians(strike))
    y1 = y0 + length * np.cos(np.radians(strike))
    z1 = z0

    # Bottom-Right (Along Strike and Down Dip)
    dip_dx = np.sin(np.radians(strike + 90))
    dip_dy = np.cos(np.radians(strike + 90))
    x2 = x1 + width * np.cos(np.radians(dip)) * dip_dx
    y2 = y1 + width * np.cos(np.radians(dip)) * dip_dy
    z2 = z0 + width * np.sin(np.radians(dip))

    # Bottom-Left (Down Dip from Top-Left)
    x3 = x0 + width * np.cos(np.radians(dip)) * dip_dx
    y3 = y0 + width * np.cos(np.radians(dip)) * dip_dy
    z3 = z0 + width * np.sin(np.radians(dip))

    corners = [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
    return np.array(corners)


def get_fault_displacement(
    strike: float,
    dip: float,
    width: float,
    height: float,
    x: float,
    y: float,
    z: float,
    ds: float,
    dd: float,
    do: float,
    n_pixels: int = 128,
) -> np.ndarray:
    """
    Generate the displacement field around a slipping fault
    """
    slip = [ds, dd, do]  # displacement (strike, dip, opening)
    fault_pts = get_fault_corners(x, y, z, strike, dip, width, height)
    fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    xs = np.linspace(0, 1, n_pixels)
    ys = np.linspace(0, 1, n_pixels)
    zs = np.linspace(-1, 0, n_pixels)
    obsx, obsy, obsz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.array([obsx, obsy, obsz]).reshape((3, -1)).T.copy()

    disp_mat = solver.disp_matrix(obs_pts=pts, tris=fault_pts[fault_tris], nu=0.25)
    slip = np.array([slip, slip])
    disp = disp_mat.reshape((-1, 6)).dot(slip.flatten())
    disp_grid = disp.reshape((*obsx.shape, 3))

    return disp_grid


def record_case(output_root: str, sample_id: int, inputs: np.ndarray, data: np.ndarray) -> None:
    output_name = os.path.join(output_root, "image_{:06d}.h5".format(sample_id))

    # Note that compression greatly increases the time to load data
    with h5py.File(output_name, mode="w") as f:
        f.create_dataset(
            "inputs",
            data=inputs.astype(np.float32),
            dtype=np.float32,
        )
        f.create_dataset("fields", data=data.astype(np.float32), dtype=np.float32)


def record_scaling(output_root: str, parameter_min: list[float], parameter_max: list[float], field_min: list[float], field_max: list[float]) -> None:
    output_name = os.path.join(output_root, 'scaling.yaml')
    
    data = {
        'parameter_min': parameter_min,
        'parameter_max': parameter_max,
        'field_min': field_min,
        'field_max': field_max
    }

    with open(output_name, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", default=5, type=int, help="Number of samples to generate")
    parser.add_argument("-o", "--output-folder", default="./results", type=str, help="Output folder for data")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    sampler = qmc.LatinHypercube(d=10)
    sample = sampler.random(n=args.num_samples)
    lower_bounds = [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    upper_bounds = [360.0, 90.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.01, 0.01, 0.01]
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)

    # Check scale
    full_scale = 1e-20
    for ii in range(10):
        r = scaled_samples[ii]
        dx = get_fault_displacement(*r)
        full_scale = max(full_scale, np.amax(abs(dx)))

    for ii in tqdm(range(args.num_samples)):
        r = scaled_samples[ii]
        s = sample[ii]
        dx = get_fault_displacement(*r) / full_scale
        dx_reshaped = np.moveaxis(dx, -1, 0)
        record_case(args.output_folder, ii, np.array(s), dx_reshaped)

    # record_scaling(args.output_folder, lower_bounds, upper_bounds, dx_min, dx_max)

if __name__ == "__main__":
    main()
