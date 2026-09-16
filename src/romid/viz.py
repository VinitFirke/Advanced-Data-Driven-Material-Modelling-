"""Shared plotting helpers for displacement/error contour fields.

Dedupes what used to be ~5 copy-pasted 30-line matplotlib blocks across
`final_fem_sol.py` and the notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SOLVER, SolverConfig


def _draw_hole(ax, cfg: SolverConfig) -> None:
    cx, cy = cfg.hole_center
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill(cx + cfg.hole_radius * np.cos(theta), cy + cfg.hole_radius * np.sin(theta), "w", edgecolor="k", lw=2)


def _save(fig, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "pdf" if save_path.suffix == ".pdf" else None
    fig.savefig(save_path, format=fmt, bbox_inches="tight")


def plot_fields(
    x: np.ndarray,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    cfg: SolverConfig = SOLVER,
    save_path: str | Path | None = None,
    figsize_per_panel: tuple[float, float] = (20, 5),
):
    """Stacked filled-contour plot, one panel per named field.

    Used both for a single displacement field (`{"X Displacement": ux, "Y
    Displacement": uy}`) and for FOM/ROM/error comparisons (one entry per panel).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(fields), 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(fields)))
    if len(fields) == 1:
        axes = [axes]

    for ax, (title, field) in zip(axes, fields.items()):
        contour = ax.tricontourf(x, y, field, levels=100, cmap="turbo")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(label="Displacement (m)", size=16)
        ax.set_title(title, size=18)
        ax.set_xlabel("X Coordinate [m]", size=14)
        ax.set_ylabel("Y Coordinate [m]", size=14)
        ax.grid(True)
        _draw_hole(ax, cfg)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes
