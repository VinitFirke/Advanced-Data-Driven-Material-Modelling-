"""romid: reduced-order modeling for FEM-based material parameter identification.

Submodules that touch dolfinx/gmsh (`fom`, `geometry`) are imported lazily by the
functions that need them, so `romid.rom`, `romid.surrogate`, `romid.data`,
`romid.viz`, and the ROM half of `romid.inverse` work in a plain numpy/scipy/
scikit-learn environment with no FEniCSx install required.
"""

__version__ = "0.1.0"
