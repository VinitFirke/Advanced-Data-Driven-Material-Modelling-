"""Inverse identification: recover (G, K) from experimental displacement data.

`identify_rom` (and `rom_loss`) depend only on numpy/scipy and the trained
`RBFSurrogate` + POD basis — no dolfinx required, so this is what the demo app and
CI tests exercise. `identify_fom` (and `fom_loss`) additionally need dolfinx,
imported lazily so importing this module doesn't require it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import IDENTIFICATION, IDENTIFICATION_FOM, SOLVER, IdentificationConfig, SolverConfig
from .data import mask_hole
from .rom import reconstruct_from_basis
from .surrogate import RBFSurrogate


@dataclass
class IdentificationResult:
    G: float
    K: float
    loss: float
    n_iterations: int
    n_function_evals: int
    elapsed_seconds: float


def fom_loss(domain, facet_tags, experimental: np.ndarray, G: float, K: float, cfg: SolverConfig = SOLVER) -> float:
    """Weighted mismatch between the FOM displacement field and experimental data.

    `experimental` is an (N, 4) array of columns (x, y, ux, uy) already interpolated
    onto the FOM's subdomain grid.
    """
    from .fom import solve_fom

    _, _, subdomain, sub_solution = solve_fom(domain, facet_tags, G, K)
    fom_mask = mask_hole(subdomain[:, 0], subdomain[:, 1], cfg)
    exp_mask = mask_hole(experimental[:, 0], experimental[:, 1], cfg)

    exp_ux, exp_uy = experimental[:, 2][exp_mask], experimental[:, 3][exp_mask]
    fom_ux, fom_uy = sub_solution[:, 0][fom_mask], sub_solution[:, 1][fom_mask]

    weight_x = 1.0 / np.mean(np.abs(exp_ux))
    weight_y = 1.0 / np.mean(np.abs(exp_uy))
    err_x = np.linalg.norm(weight_x * (exp_ux - fom_ux))
    err_y = np.linalg.norm(weight_y * (exp_uy - fom_uy))
    return 0.5 * (err_x**2 + err_y**2)


def rom_loss(
    surrogate: RBFSurrogate,
    basis: np.ndarray,
    experimental: np.ndarray,
    G: float,
    K: float,
    mean_x: float,
    std_x: float,
    mean_y: float,
    std_y: float,
) -> float:
    """Normalized mismatch between the ROM (surrogate) displacement field and experimental data."""
    num_nodes = experimental.shape[0]
    predicted_coeffs = surrogate.predict(G, K)
    rom_ux, rom_uy = reconstruct_from_basis(basis, predicted_coeffs, num_nodes)

    exp_ux_n = (experimental[:, 0] - mean_x) / std_x
    exp_uy_n = (experimental[:, 1] - mean_y) / std_y
    rom_ux_n = (rom_ux - mean_x) / std_x
    rom_uy_n = (rom_uy - mean_y) / std_y

    err_x = np.linalg.norm(exp_ux_n - rom_ux_n)
    err_y = np.linalg.norm(exp_uy_n - rom_uy_n)
    return 0.5 * (err_x**2 + err_y**2)


def identify_fom(
    domain, facet_tags, experimental: np.ndarray, cfg: IdentificationConfig = IDENTIFICATION_FOM
) -> IdentificationResult:
    """Identify (G, K) by minimizing `fom_loss` against the full-order model. Requires dolfinx."""
    start = time.time()
    result = minimize(
        fun=lambda beta: fom_loss(domain, facet_tags, experimental, *beta),
        x0=cfg.initial_guess,
        method=cfg.method,
        options={"maxiter": cfg.maxiter},
    )
    elapsed = time.time() - start
    G, K = result.x
    return IdentificationResult(G, K, float(result.fun), result.nit, result.nfev, elapsed)


def identify_rom(
    surrogate: RBFSurrogate,
    basis: np.ndarray,
    experimental_ux: np.ndarray,
    experimental_uy: np.ndarray,
    cfg: IdentificationConfig = IDENTIFICATION,
) -> IdentificationResult:
    """Identify (G, K) by minimizing `rom_loss` against the fast RBF surrogate.

    This is the ~1000x-faster path: no dolfinx/FEM solve involved, just a
    trained surrogate + reduced basis, so it runs in well under a second.
    """
    experimental = np.column_stack([experimental_ux, experimental_uy])
    mean_x, std_x = np.mean(experimental[:, 0]), np.std(experimental[:, 0])
    mean_y, std_y = np.mean(experimental[:, 1]), np.std(experimental[:, 1])

    start = time.time()
    result = minimize(
        fun=lambda beta: rom_loss(surrogate, basis, experimental, *beta, mean_x, std_x, mean_y, std_y),
        x0=cfg.initial_guess,
        method=cfg.method,
        options={"maxiter": cfg.maxiter},
    )
    elapsed = time.time() - start
    G, K = result.x
    return IdentificationResult(G, K, float(result.fun), result.nit, result.nfev, elapsed)
