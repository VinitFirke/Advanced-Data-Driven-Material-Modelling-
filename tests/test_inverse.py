import numpy as np

from romid.config import IdentificationConfig
from romid.inverse import identify_rom
from romid.rom import reconstruct_from_basis
from romid.surrogate import RBFSurrogate


def _synthetic_rom_problem(num_nodes=40, num_modes=4, n_train=100, seed=1):
    rng = np.random.default_rng(seed)

    # Orthonormal basis so reconstruction is well-conditioned.
    raw = rng.normal(size=(2 * num_nodes, num_modes))
    basis, _ = np.linalg.qr(raw)
    basis = basis[:, :num_modes]

    # Mostly-linear, injective map (G, K) -> coefficients (small smooth nonlinear
    # terms added for realism) so the inverse problem has a single, well-conditioned
    # minimum near the true parameters -- mirrors the well-posedness of the real
    # physical FOM/ROM mapping.
    def true_coeffs(G, K):
        g_n, k_n = G / 1e9, K / 1e9
        return np.array(
            [
                0.02 * g_n + 0.01 * k_n,
                -0.015 * g_n + 0.02 * k_n,
                0.01 * g_n + 0.005 * k_n + 0.05 * np.sin(g_n / 40),
                0.005 * g_n - 0.01 * k_n + 0.05 * np.cos(k_n / 40),
            ]
        )

    G_train = rng.uniform(50e9, 140e9, size=n_train)
    K_train = rng.uniform(50e9, 140e9, size=n_train)
    params = np.column_stack([G_train, K_train])
    coeffs = np.array([true_coeffs(g, k) for g, k in params])

    surrogate = RBFSurrogate().fit(params, coeffs)
    return basis, surrogate, true_coeffs


def test_identify_rom_recovers_known_parameters():
    num_nodes = 40
    basis, surrogate, true_coeffs = _synthetic_rom_problem(num_nodes=num_nodes)

    G_true, K_true = 90e9, 100e9
    ux, uy = reconstruct_from_basis(basis, true_coeffs(G_true, K_true), num_nodes)

    cfg = IdentificationConfig(initial_guess=(70e9, 70e9), maxiter=500)
    result = identify_rom(surrogate, basis, ux, uy, cfg=cfg)

    assert abs(result.G - G_true) / G_true < 0.05
    assert abs(result.K - K_true) / K_true < 0.05
    assert result.loss < 1e-3


def test_identify_rom_result_has_positive_elapsed_time():
    num_nodes = 40
    basis, surrogate, true_coeffs = _synthetic_rom_problem(num_nodes=num_nodes)
    ux, uy = reconstruct_from_basis(basis, true_coeffs(95e9, 95e9), num_nodes)

    result = identify_rom(surrogate, basis, ux, uy)
    assert result.elapsed_seconds >= 0
    assert result.n_function_evals > 0
