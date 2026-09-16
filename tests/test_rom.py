import numpy as np

from romid.config import ROMConfig
from romid.rom import compute_pod_basis, project_onto_basis, reconstruct_from_basis, singular_value_spectrum


def _make_low_rank_snapshots(num_dofs=200, num_samples=30, true_rank=4, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(num_dofs, true_rank))
    coeffs = rng.normal(size=(true_rank, num_samples))
    return basis @ coeffs


def test_pod_basis_is_orthonormal():
    snapshots = _make_low_rank_snapshots()
    basis = compute_pod_basis(snapshots, cfg=ROMConfig(num_modes=4))
    gram = basis.T @ basis
    assert np.allclose(gram, np.eye(4), atol=1e-8)


def test_pod_basis_reconstructs_low_rank_data_exactly():
    snapshots = _make_low_rank_snapshots(true_rank=4)
    basis = compute_pod_basis(snapshots, cfg=ROMConfig(num_modes=4))

    for column in snapshots.T:
        coeffs = project_onto_basis(basis, column)
        reconstructed = basis @ coeffs
        assert np.allclose(reconstructed.flatten(), column, atol=1e-8)


def test_truncated_basis_shape():
    snapshots = _make_low_rank_snapshots(num_dofs=200, num_samples=30, true_rank=10)
    basis = compute_pod_basis(snapshots, cfg=ROMConfig(num_modes=5))
    assert basis.shape == (200, 5)


def test_singular_value_spectrum_is_sorted_descending():
    snapshots = _make_low_rank_snapshots()
    values = singular_value_spectrum(snapshots)
    assert np.all(np.diff(values) <= 1e-10)


def test_reconstruct_from_basis_splits_ux_uy():
    num_nodes = 50
    basis = np.eye(2 * num_nodes)[:, :3]
    coeffs = np.array([1.0, 2.0, 3.0])
    ux, uy = reconstruct_from_basis(basis, coeffs, num_nodes)
    assert ux.shape == (num_nodes,)
    assert uy.shape == (num_nodes,)
