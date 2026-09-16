"""Full-order model (FOM): linear-elastic FEM solve for the tensile specimen.

Requires `dolfinx`, `ufl`, and `mpi4py` (see the `fem` extra in pyproject.toml).
This is the single canonical implementation of the solve that used to be
duplicated near-verbatim between `final_fem_sol.py` and the notebook.
"""
from __future__ import annotations

from .config import FACET_TAGS, SOLVER, FacetTags, SolverConfig


def read_mesh(msh_path: str, comm=None, rank: int = 0):
    """Load the mesh + cell/facet tags written by `romid.geometry.build_mesh`."""
    from dolfinx.io import gmshio
    from mpi4py import MPI

    comm = comm or MPI.COMM_WORLD
    return gmshio.read_from_msh(msh_path, comm, rank)


def solve_fom(
    domain,
    facet_tags,
    G: float,
    K: float,
    tags: FacetTags = FACET_TAGS,
    solver_cfg: SolverConfig = SOLVER,
):
    """Solve the linear-elastic FOM for shear modulus `G` and bulk modulus `K`.

    Returns
    -------
    domain : the dolfinx mesh (passed through for convenience)
    uh : the full displacement solution function
    domain_sub : (N, 3) node coordinates cropped to the DIC-observed subdomain and
        shifted into the experimental data's coordinate frame (see `SolverConfig`)
    u_sub : (N, 3) displacement values at those nodes
    """
    import numpy as np
    import ufl
    from dolfinx import default_scalar_type, fem
    from dolfinx.fem import locate_dofs_topological
    from dolfinx.fem.petsc import LinearProblem

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    # Clamped boundary: zero displacement.
    b_D = locate_dofs_topological(V, fdim, facet_tags.find(tags.clamped))
    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc_D = fem.dirichletbc(u_D, b_D, V)

    # Loaded boundary: constant traction in x.
    T_neumann = fem.Constant(domain, default_scalar_type((solver_cfg.traction_x, 0, 0)))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return (-2 / 3 * G + K) * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * G * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T_neumann, v) * ds(tags.traction)

    problem = LinearProblem(a, L, bcs=[bc_D], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    x_min, x_max = solver_cfg.subdomain_x_range
    domain_arr = domain.geometry.x
    subdomain_mask = (domain_arr[:, 0] >= x_min) & (domain_arr[:, 0] <= x_max)

    num_dofs = V.dofmap.index_map.size_local
    uh_arr = uh.x.array[:].reshape((num_dofs, -1))

    domain_sub = domain_arr[subdomain_mask].copy()
    domain_sub[:, 0] -= x_min
    domain_sub[:, 1] += solver_cfg.subdomain_y_shift
    u_sub = uh_arr[subdomain_mask]

    return domain, uh, domain_sub, u_sub
