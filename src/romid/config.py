"""Central configuration: geometry, boundary conditions, and solver settings.

Replaces the magic numbers that used to be scattered (and sometimes duplicated
inconsistently) across `final_fem_sol.py`, `geometry_gen (1).py`, and the notebook.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryConfig:
    """Tensile specimen geometry, all lengths in meters."""

    transition_length: float = 5e-3  # L_t
    transition_x: float = 15e-3  # L_roundx
    gauge_length: float = 80e-3  # L_specimen
    fixture_width: float = 30e-3  # W_fix
    gauge_width: float = 20e-3  # W_specimen
    hole_radius: float = 4e-3
    transition_radius: float = 25e-3
    mesh_size: float = 0.5e-3

    @property
    def hole_center(self) -> tuple[float, float]:
        return (self.transition_x + self.transition_length + self.gauge_length / 2, 0.0)


@dataclass(frozen=True)
class FacetTags:
    """Gmsh physical group tags written by `romid.geometry` and read by `romid.fom`."""

    surface: int = 1
    hole_boundary: int = 2
    clamped: int = 3  # Dirichlet (u=0) boundary
    traction: int = 4  # Neumann (loaded) boundary


@dataclass(frozen=True)
class SolverConfig:
    """FOM boundary conditions and the DIC-observed-subdomain crop/shift.

    The FOM mesh spans the full specimen; `subdomain_x_range` crops it to the
    region the DIC camera actually observed, and `subdomain_y_shift` re-origins
    that crop to match the experimental data's coordinate frame. `hole_center`/
    `hole_radius` are expressed *in that shifted frame*.
    """

    traction_x: float = 106.26e6  # Pa, applied on FacetTags.traction
    subdomain_x_range: tuple[float, float] = (20e-3, 100e-3)
    subdomain_y_shift: float = 10e-3
    hole_center: tuple[float, float] = (40e-3, 10e-3)
    hole_radius: float = 4e-3


@dataclass(frozen=True)
class ROMConfig:
    """POD truncation and RBF surrogate hyperparameters."""

    num_modes: int = 8
    rbf_function: str = "multiquadric"
    rbf_epsilon: float = 10.0


@dataclass(frozen=True)
class IdentificationConfig:
    """Nelder-Mead settings for the inverse identification optimization."""

    initial_guess: tuple[float, float] = (50e9, 50e9)  # (G, K) Pa
    method: str = "Nelder-Mead"
    maxiter: int = 300


GEOMETRY = GeometryConfig()
FACET_TAGS = FacetTags()
SOLVER = SolverConfig()
ROM = ROMConfig()
IDENTIFICATION = IdentificationConfig()
IDENTIFICATION_FOM = IdentificationConfig(maxiter=100)
