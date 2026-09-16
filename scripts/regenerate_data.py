"""Regenerate the full snapshot / POD basis / RBF surrogate artifacts from scratch.

`data/sample/` ships a small, pre-built set of these artifacts (100 LHS samples)
so the CLI and demo app work with no FEniCSx install. This script rebuilds them
properly -- requires the `fenicsx` conda environment (see environment.yml) since
it drives the full-order FEM solver.

Usage:
    conda env create -f environment.yml   # once
    conda activate romid
    python scripts/regenerate_data.py --n-samples 300 --out-dir data/generated
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="data/sample/tensile_test_specimen.msh")
    parser.add_argument("--n-samples", type=int, default=300, help="Number of (G, K) LHS samples.")
    parser.add_argument("--g-range", type=float, nargs=2, default=(50e9, 140e9))
    parser.add_argument("--k-range", type=float, nargs=2, default=(50e9, 140e9))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/generated")
    args = parser.parse_args()

    from romid.fom import read_mesh
    from romid.rom import compute_pod_basis
    from romid.snapshots import (
        construct_snapshot_matrix,
        generate_training_data,
        load_material_params_and_coefficients,
        sample_parameters_lhs,
    )
    from romid.surrogate import RBFSurrogate

    out_dir = Path(args.out_dir)
    snapshots_dir = out_dir / "FOM_displacement_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh from {args.mesh} ...")
    domain, _cell_tags, facet_tags = read_mesh(args.mesh)

    print(f"Sampling {args.n_samples} (G, K) pairs via Latin Hypercube ...")
    params = sample_parameters_lhs(args.n_samples, tuple(args.g_range), tuple(args.k_range), seed=args.seed)

    print(f"Solving the FOM at each sample point (this is the slow step, ~1-2s each) ...")
    start = time.time()
    generate_training_data(domain, facet_tags, params, save_folder=snapshots_dir)
    print(f"  done in {time.time() - start:.1f}s")

    print("Building the POD basis ...")
    snapshot_matrix = construct_snapshot_matrix(snapshots_dir)
    basis = compute_pod_basis(snapshot_matrix)
    basis_path = out_dir / "reduced_basis.csv"
    import numpy as np

    np.savetxt(basis_path, basis, delimiter=",")
    print(f"  basis shape {basis.shape} -> {basis_path}")

    print("Training the RBF surrogate ...")
    material_params, coefficients = load_material_params_and_coefficients(snapshots_dir, basis)
    surrogate = RBFSurrogate().fit(material_params, coefficients)
    surrogate_path = out_dir / "rbf_surrogate.pkl"
    surrogate.save(surrogate_path)
    print(f"  trained on {len(material_params)} samples -> {surrogate_path}")

    print("Done. To identify parameters against these artifacts:")
    print(
        f"  romid identify --experimental <exp.csv> --reference-nodes <ref.csv> "
        f"--basis {basis_path} --surrogate {surrogate_path}"
    )


if __name__ == "__main__":
    main()
