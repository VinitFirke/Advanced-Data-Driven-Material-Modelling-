"""Proper Orthogonal Decomposition (POD): reduced-basis construction from snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ROM, ROMConfig


def compute_pod_basis(snapshot_matrix: np.ndarray, cfg: ROMConfig = ROM) -> np.ndarray:
    """Truncated left-singular-vector basis of the snapshot matrix (the POD modes)."""
    U, _singular_values, _Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
    return U[:, : cfg.num_modes]


def singular_value_spectrum(snapshot_matrix: np.ndarray) -> np.ndarray:
    """Singular values of the snapshot matrix, for scree / energy-capture plots."""
    return np.linalg.svd(snapshot_matrix, full_matrices=False, compute_uv=False)


def project_onto_basis(basis: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """POD coefficients a = basis^T @ displacement, for one stacked (ux; uy) column."""
    return basis.T @ np.asarray(displacement).reshape(-1, 1)


def reconstruct_from_basis(basis: np.ndarray, coefficients: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (ux, uy) at each node from POD coefficients: displacement = basis @ a."""
    displacement = (basis @ np.asarray(coefficients).reshape(-1, 1)).flatten()
    return displacement[:num_nodes], displacement[num_nodes:]


def compute_pod_coefficients_from_csv(basis: np.ndarray, displacement_csv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POD coefficients for one snapshot CSV (columns x, y, ux, uy), plus its (x, y) coords."""
    df = pd.read_csv(displacement_csv)
    displacement = np.concatenate([df["ux"].values, df["uy"].values])
    coefficients = project_onto_basis(basis, displacement)
    return coefficients.flatten(), df["x"].values, df["y"].values
