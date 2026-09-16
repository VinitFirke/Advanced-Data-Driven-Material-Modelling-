"""Tensile-specimen mesh generation via Gmsh.

Requires `gmsh` and `meshio` (see the `fem` extra in pyproject.toml).
"""
from __future__ import annotations

from pathlib import Path

from .config import FACET_TAGS, GEOMETRY, GeometryConfig


def build_mesh(
    config: GeometryConfig = GEOMETRY,
    msh_path: str | Path = "tensile_test_specimen.msh",
    xdmf_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    """Generate the tensile-specimen mesh (a plate with a central hole) and write it to disk.

    Physical groups follow `romid.config.FacetTags`: 1=surface, 2=hole boundary,
    3=clamped edge, 4=loaded edge.

    `show=True` opens the interactive Gmsh GUI after meshing (blocks until closed) —
    off by default so this runs headlessly in scripts/CI.
    """
    import gmsh
    import meshio

    msh_path = Path(msh_path)
    xdmf_path = Path(xdmf_path) if xdmf_path else msh_path.with_suffix(".xdmf")

    gmsh.initialize()
    try:
        gmsh.model.add("tensile_test_specimen")

        L_t = config.transition_length
        L_roundx = config.transition_x
        L_specimen = config.gauge_length
        W_fix = config.fixture_width
        W_specimen = config.gauge_width
        hole_center = config.hole_center
        hole_radius = config.hole_radius
        transition_radius = config.transition_radius

        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(0, -W_fix / 2, 0)
        p3 = gmsh.model.geo.addPoint(L_roundx, -W_specimen / 2, 0)
        p4 = gmsh.model.geo.addPoint(L_roundx + L_t, -W_specimen / 2, 0)
        p5 = gmsh.model.geo.addPoint(L_roundx + L_t + L_specimen, -W_specimen / 2, 0)
        p6 = gmsh.model.geo.addPoint(L_roundx + L_t + L_specimen, W_specimen / 2, 0)
        p7 = gmsh.model.geo.addPoint(L_roundx + L_t, W_specimen / 2, 0)
        p8 = gmsh.model.geo.addPoint(L_roundx, W_specimen / 2, 0)
        p9 = gmsh.model.geo.addPoint(0, W_fix / 2, 0)
        p10 = gmsh.model.geo.addPoint(hole_center[0], hole_center[1], 0)

        cp2 = gmsh.model.geo.addPoint(L_roundx, transition_radius * 4 / 5 + W_fix / 2, 0)
        cp1 = gmsh.model.geo.addPoint(L_roundx, -(transition_radius * 4 / 5 + W_fix / 2), 0)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2arc = gmsh.model.geo.addCircleArc(p2, cp1, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p5)
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p7)
        l7 = gmsh.model.geo.addLine(p7, p8)
        l8arc = gmsh.model.geo.addCircleArc(p8, cp2, p9)
        l9 = gmsh.model.geo.addLine(p9, p1)

        p_hole1 = gmsh.model.geo.addPoint(hole_center[0] + hole_radius, hole_center[1], 0)
        p_hole2 = gmsh.model.geo.addPoint(hole_center[0] - hole_radius, hole_center[1], 0)
        arc1 = gmsh.model.geo.addCircleArc(p_hole1, p10, p_hole2)
        arc2 = gmsh.model.geo.addCircleArc(p_hole2, p10, p_hole1)

        outer_loop = gmsh.model.geo.addCurveLoop([l1, l2arc, l3, l4, l5, l6, l7, l8arc, l9])
        hole_loop = gmsh.model.geo.addCurveLoop([arc1, arc2])
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

        gmsh.model.addPhysicalGroup(2, [surface], tag=FACET_TAGS.surface)
        gmsh.model.addPhysicalGroup(1, [arc1, arc2], tag=FACET_TAGS.hole_boundary)
        gmsh.model.addPhysicalGroup(1, [l1, l9], tag=FACET_TAGS.clamped)
        gmsh.model.addPhysicalGroup(1, [l5], tag=FACET_TAGS.traction)

        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", config.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", config.mesh_size)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.write(str(msh_path))
        if show:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()

    mesh = meshio.read(str(msh_path))
    meshio.write(str(xdmf_path), mesh)
    return msh_path
