# romid — Reduced-Order Material Parameter Identification

Identify a material's **shear modulus (G)** and **bulk modulus (K)** from full-field
Digital Image Correlation (DIC) displacement data, using a **POD + RBF reduced-order
model (ROM)** trained as a fast surrogate for an expensive FEniCSx finite-element
solve.

The headline result: the ROM surrogate identifies material parameters in
**~0.1 seconds**, versus ~80 seconds for the full-order FEM model, while staying
within a few percent of the full-order model's answer. See
[`docs/theory.md`](docs/theory.md) for the full derivation, boundary-value problem,
and FOM/ROM benchmark.

<p align="center"><img src="docs/assets/fom_rom_error_disp_x.png" width="70%" alt="FOM vs ROM vs error, X displacement"></p>

## Why a reduced-order model

Solving the inverse identification problem means running the forward FEM solve
many times inside an optimization loop. For this specimen that's cheap (~80s
total), but for larger meshes or nonlinear material models it stops being
tractable. The fix here: solve the FOM once at a set of sampled (G, K) points,
compress the resulting displacement snapshots with **Proper Orthogonal
Decomposition (POD)**, and fit a **radial-basis-function (RBF) surrogate** that maps
(G, K) directly to POD coefficients — skipping the FEM solve entirely during
optimization.

```
   sample (G, K)              POD/SVD                  RBF surrogate
  ──────────────►  FOM solve  ──────────►  snapshot   ──────────────►  (G, K) → POD
   (Latin Hypercube)  (FEniCSx)             matrix       (scikit/scipy)   coefficients
                                                                              │
                                                                              ▼
   experimental DIC data  ──────────────────────────────►  Nelder-Mead  ──► identified (G, K)
                              interpolated onto FEM grid      loss minimization
                                                             (against the ROM)
```

## Quickstart

The ROM identification path (the fast one) needs only numpy/scipy/scikit-learn —
no FEniCSx required:

```bash
pip install -e .
romid identify \
    --experimental data/sample/experimental_displacements.csv \
    --reference-nodes data/sample/reference_nodes.csv \
    --basis data/sample/reduced_basis.csv \
    --surrogate data/sample/rbf_surrogate.pkl
```

Or launch the interactive demo:

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Running the full-order model (mesh generation, FEM solve, snapshot generation)
additionally needs FEniCSx, which is conda-only:

```bash
conda env create -f environment.yml
conda activate romid
romid mesh --output data/sample/tensile_test_specimen.msh
romid solve-fom --G 71.4e9 --K 92.9e9 --output fom_displacement.csv
python scripts/regenerate_data.py --n-samples 300   # full snapshot/POD/surrogate rebuild
```

## Repository layout

```
src/romid/            installable package
  config.py             geometry / BC / solver / ROM settings (dataclasses, no magic numbers)
  geometry.py            mesh generation (gmsh)               } dolfinx/gmsh required
  fom.py                  full-order FEM solve (dolfinx)       }
  data.py                 DIC data loading, outlier removal, hole masking
  snapshots.py            LHS parameter sampling, snapshot matrix construction
  rom.py                  POD/SVD basis construction
  surrogate.py            RBF surrogate: (G, K) -> POD coefficients
  inverse.py              Nelder-Mead identification, FOM- or ROM-based loss
  viz.py                  shared displacement/error contour plotting
  cli.py                  `romid mesh|solve-fom|build-rom|identify`
app/streamlit_app.py    interactive demo (ROM-only, no dolfinx needed)
data/sample/            small pre-built artifacts (mesh, basis, surrogate, experimental data)
scripts/regenerate_data.py  rebuild the full snapshot/basis/surrogate set via the FOM
tests/                  pytest suite for the dolfinx-free ROM/surrogate/inverse math
docs/theory.md          full mathematical formulation and FOM/ROM results
```

## Results

| Model | K [GPa] | G [GPa] | Identification time |
|-------|---------|---------|----------------------|
| FOM   | 92.9    | 71.4    | 81.78 s (231 iterations) |
| ROM   | 94.8 (+2.04%) | 73.7 (+3.22%) | 0.078 s (242 iterations) |

Full derivation, boundary conditions, and additional result plots:
[`docs/theory.md`](docs/theory.md).

## Development

```bash
pip install -e ".[dev]"
pytest             # dolfinx-free: rom, surrogate, inverse, data, config
ruff check src tests
```

CI (`.github/workflows/ci.yml`) runs the same lint + test suite on every push —
scoped to the dolfinx-free parts, since dolfinx isn't pip-installable.

## License

[MIT](LICENSE)
