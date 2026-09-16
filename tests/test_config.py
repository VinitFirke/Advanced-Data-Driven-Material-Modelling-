from romid.config import GEOMETRY, SOLVER


def test_hole_center_matches_subdomain_frame():
    # The hole center in the original (unshifted) geometry frame, cropped and
    # shifted per SolverConfig, must land on SOLVER.hole_center — this is the
    # invariant that made `mask_hole` usable both pre- and post-crop in the
    # original scripts.
    raw_x, raw_y = GEOMETRY.hole_center
    x_min, _x_max = SOLVER.subdomain_x_range

    shifted_x = raw_x - x_min
    shifted_y = raw_y + SOLVER.subdomain_y_shift

    assert abs(shifted_x - SOLVER.hole_center[0]) < 1e-9
    assert abs(shifted_y - SOLVER.hole_center[1]) < 1e-9


def test_hole_radius_consistent():
    assert GEOMETRY.hole_radius == SOLVER.hole_radius
