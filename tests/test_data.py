import numpy as np
import pandas as pd

from romid.config import SolverConfig
from romid.data import mask_hole, remove_outliers


def test_mask_hole_excludes_center_includes_far_points():
    cfg = SolverConfig()
    cx, cy = cfg.hole_center
    x = np.array([cx, cx + cfg.hole_radius * 2, cx - 0.05])
    y = np.array([cy, cy, cy])
    mask = mask_hole(x, y, cfg)
    assert not mask[0]  # at the hole center -> masked out
    assert mask[1]  # clearly outside the hole
    assert mask[2]  # far away -> outside the hole


def test_remove_outliers_drops_extreme_values():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(loc=1.0, scale=0.05, size=200), [1000.0]])
    df = pd.DataFrame({"v": values})
    cleaned = remove_outliers(df, ["v"], threshold=3.0)
    assert 1000.0 not in cleaned["v"].values
    assert len(cleaned) == 200
