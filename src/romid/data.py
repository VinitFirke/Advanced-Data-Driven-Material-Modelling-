"""Experimental (DIC) displacement data: loading, outlier removal, hole masking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .config import SOLVER, SolverConfig


def mask_hole(x: np.ndarray, y: np.ndarray, cfg: SolverConfig = SOLVER) -> np.ndarray:
    """Boolean mask, True for points outside the specimen's central hole.

    The single canonical implementation of a function that used to be redefined
    three separate times (module-level and nested inside two other functions) in
    `final_fem_sol.py`.
    """
    cx, cy = cfg.hole_center
    distance = np.sqrt((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2)
    return distance >= cfg.hole_radius


def remove_outliers(df: pd.DataFrame, columns: list[str], threshold: float = 3.0) -> pd.DataFrame:
    """Drop rows where any of `columns` is more than `threshold` std devs from its mean."""
    cleaned = df.copy()
    for column in columns:
        mean = cleaned[column].mean()
        std = cleaned[column].std()
        z_score = (cleaned[column] - mean) / std
        cleaned = cleaned[np.abs(z_score) <= threshold]
    return cleaned


def load_experimental_displacements(
    csv_path: str,
    node_x: np.ndarray,
    node_y: np.ndarray,
    outlier_columns: tuple[str, str] = ("ux", "uy"),
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate DIC displacement measurements onto arbitrary (node_x, node_y) coordinates.

    The experimental CSV is expected to have columns x, y, ux, uy in millimeters
    (DIC convention); the interpolated output is returned in meters. Outliers
    caused by tensile-machine stiffness are removed before interpolating.
    """
    df = pd.read_csv(csv_path)
    df = remove_outliers(df, list(outlier_columns), threshold=3.0)

    points = df[["x", "y"]].values / 1000.0
    disp_x = df["ux"].values / 1000.0
    disp_y = df["uy"].values / 1000.0

    interp_x = np.nan_to_num(griddata(points, disp_x, (node_x, node_y), method="linear"), nan=0.0)
    interp_y = np.nan_to_num(griddata(points, disp_y, (node_x, node_y), method="linear"), nan=0.0)
    return interp_x, interp_y


def load_reference_nodes(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (x, y) node coordinates from a previously solved FOM/ROM displacement CSV.

    The subdomain grid geometry doesn't depend on (G, K), so any prior FOM sample's
    node coordinates can serve as the fixed reference grid for interpolation.
    """
    df = pd.read_csv(csv_path)
    return df["x"].values, df["y"].values
