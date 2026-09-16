# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research codebase for **data-driven material parameter identification** using reduced-order
modeling. It identifies the shear modulus (G) and bulk modulus (K) of a linear-elastic tensile
specimen by fitting a FEM/POD model against Digital Image Correlation (DIC) full-field displacement
data. See `Advanced-Data-Driven-Material-Modelling-/README.md` for the full mathematical
formulation (state equation, observation operator, POD-based reduced-order loss) and result plots.

This directory is **not a git repository**. Two subdirectories are, however, independent git repos
vendored/copied in here and should be treated as external, not as part of this project's own code:
- `viskex/` — a third-party FEM visualization library (dolfinx/firedrake), unrelated to the material
  identification code except as an optional plotting/visualization dependency.
- `Advanced-Data-Driven-Material-Modelling-/` — a separate published copy of this project (its own
  README + a duplicate of the source notebook under `src_code_notebook/`). Treat it as
  the "public writeup" copy; the actively-edited code lives at the repo root.

There is no dependency manifest (no `requirements.txt`/`environment.yml`/`pyproject.toml`) for the
main project and no test suite, linter, or CI config — infer required packages from imports when in
doubt (see Dependencies below).

## Directory guide

- `final_fem_sol.py` — standalone Python script version of the full-order model (FOM) solve +
  inverse identification pipeline (Nelder-Mead optimization of G, K against interpolated
  experimental data).
- `geometry_gen (1).py` — generates the tensile-specimen mesh via `gmsh` (plate with a central hole)
  and writes `tensile_test_specimen.msh` / `.xdmf`. Calls `gmsh.fltk.run()`, which opens an
  interactive GUI window — this will hang in a headless/non-interactive environment; comment that
  call out before running non-interactively.
- `ADDDMM_src_code.ipynb` — the main, actively-developed notebook. Pipeline order: FOM solve (FEniCSx)
  → snapshot generation for POD → SVD on the snapshot matrix → RBF surrogate trained to predict POD
  coefficients from (G, K) → inverse identification against experimental displacements → FOM/ROM/
  experimental error comparison plots.
- `old_versions/` — chronological scratch history of earlier POD script iterations
  (`POD_ver1.py` … `POD_ver19.ipynb`). Reference only; not maintained.
- `aaaaa/`, `root/` — scratch/output dump directories (duplicate notebook copy, raw output diagrams).
  Not canonical; avoid treating their contents as source of truth.
- `*.csv` at the repo root — intermediate data artifacts produced by the pipeline (snapshot matrices,
  reduced basis, sampled/interpolated displacement fields, LHS-sampled parameter sets). These are
  generated outputs, not hand-authored inputs, and get overwritten by re-running the notebook/scripts.
- `FOM_displacement_data/`, `synthetic_displacement_data_FOM/` — generated full-order-model
  displacement snapshots (one CSV per (G, K) sample), used to build the snapshot matrix for POD.
- `tensile_test_specimen.{msh,xdmf,h5}` — the FEM mesh (see `geometry_gen (1).py`) consumed by the
  FOM solver via `dolfinx.io.gmshio`.
- `Figures/`, `PDF/` — output figures.

## Dependencies

The FOM solver requires **FEniCSx** (`dolfinx`, `ufl`, `mpi4py`, PETSc bindings) plus `gmsh` and
`meshio` for mesh generation/conversion. The ROM/analysis side uses `numpy`, `pandas`, `scipy`
(`optimize`, `interpolate`, `spatial.distance`, `stats.qmc`), and `scikit-learn`
(`sklearn.linear_model`). Plotting uses `matplotlib`. FEniCSx is typically installed via conda
(`conda install -c conda-forge fenics-dolfinx mpich`) rather than pip — check the active environment
before assuming any package is missing.

## Running the pipeline

There is no build step. Typical workflow, in order:

1. Generate the mesh (opens a GUI window via `gmsh.fltk.run()` unless removed):
   ```
   python3 "geometry_gen (1).py"
   ```
2. Run the FOM inverse-identification script directly (uses `mpi4py.MPI.COMM_WORLD`; safe to run
   under plain `python3` for a serial solve, or `mpirun -n <N> python3 final_fem_sol.py` for a
   parallel one):
   ```
   python3 final_fem_sol.py
   ```
3. For the full FOM → POD/SVD → RBF surrogate → inverse-ROM pipeline, run `ADDDMM_src_code.ipynb`
   cell by cell (Jupyter/JupyterLab) — later cells depend on CSV artifacts produced by earlier ones
   (e.g. `snapshot_matrix_columnstacked.csv` must exist before the SVD cell runs).

There are no automated tests, lint configuration, or CI in this repository.
