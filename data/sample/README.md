# Sample data

Small, checked-in artifacts so the ROM identification CLI and demo app work
out of the box with no FEniCSx/dolfinx install:

| File | What it is | Provenance |
|------|------------|------------|
| `tensile_test_specimen.{msh,xdmf,h5}` | FEM mesh | `romid mesh` (plate with a central hole) |
| `reduced_basis.csv` | POD basis, 8 modes, shape (15072, 8) | `romid build-rom` on 100 Latin-Hypercube-sampled (G, K) FOM solves |
| `rbf_surrogate.pkl` | Trained `RBFSurrogate` mapping (G, K) -> POD coefficients | same `build-rom` run |
| `reference_nodes.csv` | (x, y) node coordinates for the FOM subdomain grid | node coordinates are independent of (G, K), so any FOM sample's grid works as the reference |
| `experimental_displacements.csv` | Real DIC displacement measurements (courtesy Tröger et al.) | original experimental dataset used throughout this project |

To rebuild these (or a larger version) from scratch, see
`scripts/regenerate_data.py` — it requires the `fenicsx` conda environment
(`environment.yml`) since it drives the full-order FEM solver.
