import numpy as np
import pytest

from romid.surrogate import RBFSurrogate


def _training_set(seed=0, n=40):
    rng = np.random.default_rng(seed)
    G = rng.uniform(50e9, 140e9, size=n)
    K = rng.uniform(50e9, 140e9, size=n)
    params = np.column_stack([G, K])
    # Smooth synthetic function standing in for real POD coefficients.
    g_n, k_n = G / 1e9, K / 1e9
    coeffs = np.column_stack(
        [
            np.sin(g_n / 20) + 0.5 * k_n / 100,
            np.cos(k_n / 30) - g_n / 200,
        ]
    )
    return params, coeffs


def test_rbf_surrogate_interpolates_training_points_exactly():
    params, coeffs = _training_set()
    surrogate = RBFSurrogate().fit(params, coeffs)

    for i in range(len(params)):
        predicted = surrogate.predict(params[i, 0], params[i, 1])
        assert np.allclose(predicted, coeffs[i], atol=1e-6)


def test_rbf_surrogate_predict_before_fit_raises():
    surrogate = RBFSurrogate()
    with pytest.raises(RuntimeError):
        surrogate.predict(70e9, 100e9)


def test_rbf_surrogate_save_load_roundtrip(tmp_path):
    params, coeffs = _training_set()
    surrogate = RBFSurrogate().fit(params, coeffs)

    path = tmp_path / "surrogate.pkl"
    surrogate.save(path)
    loaded = RBFSurrogate.load(path)

    test_G, test_K = 82e9, 105e9
    assert np.allclose(loaded.predict(test_G, test_K), surrogate.predict(test_G, test_K))
