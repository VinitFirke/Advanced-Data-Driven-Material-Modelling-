"""Command-line entry points for the romid pipeline: mesh -> solve-fom -> build-rom -> identify.

Only `mesh`, `solve-fom`, and `build-rom` (the snapshot-generation half) require
dolfinx/gmsh; `identify` runs against a pre-trained surrogate and needs neither.
"""
from __future__ import annotations

import argparse
import sys


def _cmd_mesh(args: argparse.Namespace) -> None:
    from .geometry import build_mesh

    path = build_mesh(msh_path=args.output, show=args.show)
    print(f"Mesh written to {path}")


def _cmd_solve_fom(args: argparse.Namespace) -> None:
    import pandas as pd

    from .fom import read_mesh, solve_fom

    domain, _cell_tags, facet_tags = read_mesh(args.mesh)
    _, _, domain_sub, u_sub = solve_fom(domain, facet_tags, args.G, args.K)

    df = pd.DataFrame({"x": domain_sub[:, 0], "y": domain_sub[:, 1], "ux": u_sub[:, 0], "uy": u_sub[:, 1]})
    df.to_csv(args.output, index=False)
    print(f"Solved FOM at G={args.G:.3e} Pa, K={args.K:.3e} Pa -> {args.output} ({len(df)} nodes)")


def _cmd_build_rom(args: argparse.Namespace) -> None:
    import numpy as np

    from .rom import compute_pod_basis
    from .snapshots import construct_snapshot_matrix, load_material_params_and_coefficients
    from .surrogate import RBFSurrogate

    snapshot_matrix = construct_snapshot_matrix(args.data_folder)
    basis = compute_pod_basis(snapshot_matrix)
    np.savetxt(args.basis_output, basis, delimiter=",")
    print(f"Saved reduced basis {basis.shape} to {args.basis_output}")

    material_params, coefficients = load_material_params_and_coefficients(args.data_folder, basis)
    surrogate = RBFSurrogate().fit(material_params, coefficients)
    surrogate.save(args.surrogate_output)
    print(f"Trained RBF surrogate on {len(material_params)} samples -> {args.surrogate_output}")


def _cmd_identify(args: argparse.Namespace) -> None:
    import numpy as np

    from .data import load_experimental_displacements, load_reference_nodes
    from .inverse import identify_rom
    from .surrogate import RBFSurrogate

    node_x, node_y = load_reference_nodes(args.reference_nodes)
    ux, uy = load_experimental_displacements(args.experimental, node_x, node_y)

    basis = np.loadtxt(args.basis, delimiter=",")
    surrogate = RBFSurrogate.load(args.surrogate)

    result = identify_rom(surrogate, basis, ux, uy)
    print(
        f"Identified G={result.G:.4e} Pa, K={result.K:.4e} Pa "
        f"(loss={result.loss:.6f}, {result.n_iterations} iterations, {result.elapsed_seconds:.4f}s)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="romid", description="Reduced-order material identification pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mesh = sub.add_parser("mesh", help="Generate the tensile-specimen mesh (requires gmsh).")
    p_mesh.add_argument("--output", default="tensile_test_specimen.msh")
    p_mesh.add_argument("--show", action="store_true", help="Open the interactive Gmsh GUI after meshing.")
    p_mesh.set_defaults(func=_cmd_mesh)

    p_solve = sub.add_parser("solve-fom", help="Solve the full-order FEM model at a given (G, K). Requires dolfinx.")
    p_solve.add_argument("--mesh", default="tensile_test_specimen.msh")
    p_solve.add_argument("--G", type=float, required=True)
    p_solve.add_argument("--K", type=float, required=True)
    p_solve.add_argument("--output", default="fom_displacement.csv")
    p_solve.set_defaults(func=_cmd_solve_fom)

    p_rom = sub.add_parser("build-rom", help="Build the POD basis + RBF surrogate from generated FOM snapshots.")
    p_rom.add_argument("--data-folder", default="FOM_displacement_data")
    p_rom.add_argument("--basis-output", default="reduced_basis.csv")
    p_rom.add_argument("--surrogate-output", default="rbf_surrogate.pkl")
    p_rom.set_defaults(func=_cmd_build_rom)

    p_id = sub.add_parser("identify", help="Identify (G, K) from experimental data via the ROM surrogate (no dolfinx needed).")
    p_id.add_argument("--experimental", required=True, help="CSV of DIC displacement measurements (x, y, ux, uy in mm).")
    p_id.add_argument("--reference-nodes", required=True, help="CSV with x, y columns giving the FOM subdomain grid.")
    p_id.add_argument("--basis", required=True, help="Reduced POD basis CSV (from `build-rom`).")
    p_id.add_argument("--surrogate", required=True, help="Trained RBFSurrogate pickle (from `build-rom`).")
    p_id.set_defaults(func=_cmd_identify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
