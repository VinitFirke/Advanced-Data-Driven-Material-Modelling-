"""Surrogate model: maps material parameters (G, K) to POD coefficients.

This is the ~1000x-faster stand-in for the FOM used during ROM-based inverse
identification (see `romid.inverse.identify_rom`) and by the interactive demo app.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import ROM, ROMConfig


class RBFSurrogate:
    """Per-POD-coefficient radial basis function surrogate for (G, K) -> POD coefficients."""

    def __init__(self, cfg: ROMConfig = ROM):
        self.cfg = cfg
        self._models: list | None = None

    def fit(self, material_params: np.ndarray, pod_coefficients: np.ndarray) -> RBFSurrogate:
        """`material_params`: (N, 2) array of (G, K). `pod_coefficients`: (N, num_modes)."""
        from scipy.interpolate import Rbf

        G, K = material_params[:, 0], material_params[:, 1]
        self._models = [
            Rbf(G, K, pod_coefficients[:, i], function=self.cfg.rbf_function, epsilon=self.cfg.rbf_epsilon)
            for i in range(pod_coefficients.shape[1])
        ]
        return self

    def predict(self, G: float, K: float) -> np.ndarray:
        if self._models is None:
            raise RuntimeError("RBFSurrogate.fit() (or .load()) must be called before predict().")
        return np.array([model(G, K) for model in self._models])

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> RBFSurrogate:
        with open(path, "rb") as f:
            surrogate = pickle.load(f)
        if not isinstance(surrogate, RBFSurrogate):
            raise TypeError(f"{path} does not contain an RBFSurrogate")
        return surrogate
