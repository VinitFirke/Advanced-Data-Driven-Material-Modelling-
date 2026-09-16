"""Snapshot generation for POD: sample (G, K) pairs, solve the FOM, and stack the results."""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sample_parameters_lhs(
    n_samples: int,
    g_range: tuple[float, float] = (50e9, 140e9),
    k_range: tuple[float, float] = (50e9, 140e9),
    seed: int | None = 0,
) -> np.ndarray:
    """Latin Hypercube sample of (G, K) pairs covering the full 2D parameter box.

    Replaces the original notebook's `zip(linspace(G), linspace(K))`, which paired
    the i-th G with the i-th K and so only ever sampled the G=K diagonal of the
    parameter space instead of covering it.
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    bounds = np.array([g_range, k_range])
    return qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])


def generate_training_data(
    domain,
    facet_tags,
    parameters: np.ndarray,
    save_folder: str | Path = "FOM_displacement_data",
) -> list[Path]:
    """Solve the FOM at each (G, K) sample and write a displacement CSV per sample. Requires dolfinx."""
    from .fom import solve_fom

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    written = []
    for G, K in parameters:
        _, _, domain_sub, u_sub = solve_fom(domain, facet_tags, G, K)
        filename = save_folder / f"full_domain_data_G_{G:.4f}_K_{K:.4f}.csv"
        df = pd.DataFrame({"x": domain_sub[:, 0], "y": domain_sub[:, 1], "ux": u_sub[:, 0], "uy": u_sub[:, 1]})
        df.to_csv(filename, index=False)
        written.append(filename)
    return written


def construct_snapshot_matrix(
    data_folder: str | Path = "FOM_displacement_data",
    pattern: str = "full_domain_data_G_*.csv",
) -> np.ndarray:
    """Stack (ux, uy) from each per-sample CSV into a column-stacked snapshot matrix."""
    csv_files = sorted(glob.glob(str(Path(data_folder) / pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No snapshot CSVs found matching {data_folder}/{pattern}")

    snapshots = []
    for file in csv_files:
        df = pd.read_csv(file)
        column = np.concatenate([df["ux"].values, df["uy"].values]).reshape(-1, 1)
        snapshots.append(column)
    return np.column_stack(snapshots)


def load_material_params_and_coefficients(
    data_folder: str | Path,
    basis: np.ndarray,
    pattern: str = "full_domain_data_G_*.csv",
) -> tuple[np.ndarray, np.ndarray]:
    """Parse (G, K) from each snapshot filename and compute its POD coefficients.

    Used to build the (G, K) -> POD-coefficient training set for `RBFSurrogate`.
    """
    from .rom import compute_pod_coefficients_from_csv

    files = sorted(glob.glob(str(Path(data_folder) / pattern)))
    material_params, coefficients = [], []
    for file in files:
        # Anchored to the ".csv" suffix so the K capture doesn't greedily swallow
        # the extension's leading dot (a real bug in the original notebook, there
        # worked around with a `.rstrip(".")` rather than fixing the regex).
        match = re.search(r"G_([\d.]+)_K_([\d.]+)\.csv$", file)
        if not match:
            continue
        G, K = float(match.group(1)), float(match.group(2))
        coeffs, _x, _y = compute_pod_coefficients_from_csv(basis, file)
        material_params.append([G, K])
        coefficients.append(coeffs)
    return np.array(material_params), np.array(coefficients)
