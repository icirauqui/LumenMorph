import numpy as np
import pytest

from backend.src.benchmark_validation import (
    _directions, _ray_hits, audit_bridge_signal, audit_mesh_fields,
    audit_raster_samples,
)


INTR = {"width": 8, "height": 6, "fx": 8., "fy": 8., "cx": 4., "cy": 3.}


def square_mesh():
    return (np.array([[-3., -3., 2.], [3., -3., 2.], [3., 3., 2.], [-3., 3., 2.]]),
            np.array([[0, 1, 2], [0, 2, 3]]))


def analytic_plane_buffers(vertices, triangles, z_intercept=2., x_slope=0.):
    """Analytic z=z_intercept+x_slope*x plane; barycentrics from linear solve."""
    ids = np.empty((6, 8), dtype=int)
    bary = np.empty((6, 8, 3))
    depth = np.empty((6, 8))
    for row in range(6):
        for col in range(8):
            direction = np.array([(col+.5-4)/8, (3-row-.5)/8, 1.])
            z = z_intercept / (1 - x_slope * direction[0])
            point = z * direction
            for index, tri in enumerate(triangles):
                weights = np.linalg.lstsq(np.vstack((vertices[tri].T, np.ones(3))), np.r_[point, 1.], rcond=None)[0]
                if weights.min() >= -1e-10:
                    ids[row, col] = index
                    bary[row, col] = weights
                    depth[row, col] = z
                    break
            else:
                raise AssertionError("analytic point outside manufactured plane")
    return ids, bary, depth


def test_ray_audit_perspective_depth_and_negative_image_y():
    vertices, triangles = square_mesh()
    vertices[:, 2] = 2 + .2 * vertices[:, 0]
    ids, bary, z = analytic_plane_buffers(vertices, triangles, 2, .2)
    report = audit_raster_samples(vertices, triangles, np.eye(3), np.zeros(3), INTR, ids, bary, z, sample_count=48)
    assert report["passed"]
    assert report["max_reprojection_error_px"] < 1e-12
    bad = bary[::-1].copy()
    assert not audit_raster_samples(vertices, triangles, np.eye(3), np.zeros(3), INTR, ids, bad, z, sample_count=48)["passed"]


def test_physical_near_surface_overrides_plausible_far_labels():
    near, tri = square_mesh()
    far = near * np.array([1., 1., 2.])
    vertices = np.vstack((near, far))
    triangles = np.vstack((tri, tri+4))
    ids, bary, depth = analytic_plane_buffers(far, tri, 4)
    report = audit_raster_samples(vertices, triangles, np.eye(3), np.zeros(3), INTR, ids+2, bary, depth, sample_count=48)
    assert not report["passed"]
    assert report["max_relative_depth_error"] == pytest.approx(1.)
    assert any("frontmost_physical_depth" in f["reasons"] for f in report["failures"])


def test_ray_inverse_uses_actual_float_camera_matrix():
    # Exact inverse matters if stored matrix is not perfectly orthogonal.
    R = np.array([[1.001, .001, 0], [0, .999, 0], [0, 0, 1.0001]])
    pixels = np.array([[1.5, 2.5], [5.5, 4.5]])
    directions = _directions(pixels, INTR, np.linalg.inv(R.T))
    camera = directions @ R
    actual = np.column_stack((8*camera[:, 0]/camera[:, 2]+4, 3-8*camera[:, 1]/camera[:, 2]))
    assert np.allclose(actual, pixels, atol=1e-14)


def test_audit_identifies_renderer_hole_from_crossing_triangle():
    vertices = np.array([[-3., -3., 2.], [3., -3., 2.], [0., 3., -1.]])
    tri = np.array([[0, 1, 2]])
    empty_ids = np.full((6, 8), -1)
    empty_bary = np.full((6, 8, 3), np.nan)
    empty_depth = np.full((6, 8), np.inf)
    report = audit_raster_samples(vertices, tri, np.eye(3), np.zeros(3), INTR, empty_ids, empty_bary, empty_depth, sample_count=48)
    assert report["ray_foreground_count"] > 0
    assert not report["passed"]


def test_whole_field_zero_rigid_translation_and_edge_distortion():
    vertices, triangles = square_mesh()
    states = np.stack((vertices, vertices + [0, 0, .02], vertices * [1.1, 1., 1.]))
    report = audit_mesh_fields(vertices, triangles, states, 1.)
    assert report["passed"]
    assert report["frames"][0]["peak_displacement_over_radius"] == 0
    assert report["frames"][1]["max_absolute_edge_length_change"] == 0
    assert report["frames"][2]["max_absolute_edge_length_change"] == pytest.approx(.1)
    collapsed = np.repeat(vertices[:1], len(vertices), axis=0)
    assert not audit_mesh_fields(vertices, triangles, collapsed[None], 1.)["passed"]


def test_signal_fixed_cohort_and_exact_material_contrast():
    vertices, triangles = square_mesh()
    states = np.stack((vertices, vertices + [0, 0, .02]))
    report = audit_bridge_signal(vertices, triangles, states, np.eye(3), np.zeros(3), INTR, [0], [1], radius=1., sample_count=48)
    assert report["nominal_signal_qualified"]
    assert report["sampled_image_ray_count"] == report["reference_visible_count"] == 48
    assert report["mean_target_over_radius"] == pytest.approx(.02)
    assert report["p95_target_over_radius"] == pytest.approx(.02)
    assert report["common_visible_fraction"] == 1
    static = audit_bridge_signal(vertices, triangles, np.stack((vertices, vertices)), np.eye(3), np.zeros(3), INTR, [0], [1], radius=1., sample_count=48)
    assert not static["nominal_signal_qualified"]
    assert static["mean_target_over_radius"] == 0
    assert report["cohort"]["pixels_xy"] == static["cohort"]["pixels_xy"]


def test_signal_preserves_occluded_material_in_target_denominator():
    near, tri = square_mesh()
    far = near + [0, 0, 2]
    vertices = np.vstack((near, far))
    triangles = np.vstack((tri, tri+4))
    moved = vertices.copy()
    moved[:4, 2] += 3  # The original cohort becomes occluded by the unchanged back plane.
    report = audit_bridge_signal(vertices, triangles, np.stack((vertices, moved)), np.eye(3), np.zeros(3), INTR, [0], [1], radius=1., sample_count=48)
    assert report["mean_target_over_radius"] == pytest.approx(3.)
    assert report["common_visible_count"] == 0
    assert report["reference_visible_count"] == 48
    assert not report["nominal_signal_qualified"]


def test_bad_contrast_rejected():
    vertices, tri = square_mesh()
    with pytest.raises(ValueError, match="disjoint"):
        audit_bridge_signal(vertices, tri, np.stack((vertices, vertices)), np.eye(3), np.zeros(3), INTR, [0], [0], radius=1.)


def test_clean_rgb_material_texture_mismatch_is_detected():
    vertices, tri = square_mesh()
    ids, bary, depth = analytic_plane_buffers(vertices, tri)
    texture = np.zeros((2, 2, 3), dtype=np.uint8)
    texture[:] = [41, 83, 127]
    uv = np.full((4, 2), .3)
    rgb = np.broadcast_to(texture[0, 0], (6, 8, 3)).copy()
    assert audit_raster_samples(vertices, tri, np.eye(3), np.zeros(3), INTR, ids, bary, depth, sample_count=48, clean_rgb=rgb, texture=texture, uv_vertices=uv)["passed"]
    rgb[:] = 0
    assert not audit_raster_samples(vertices, tri, np.eye(3), np.zeros(3), INTR, ids, bary, depth, sample_count=48, clean_rgb=rgb, texture=texture, uv_vertices=uv)["passed"]


def test_signal_caches_identical_holds_without_changing_frame_denominators():
    vertices, triangles = square_mesh()
    states = np.stack([vertices]*4 + [vertices+[0, 0, .02]]*4)
    report = audit_bridge_signal(vertices, triangles, states, np.eye(3), np.zeros(3), INTR, [0, 1, 2, 3], [4, 5, 6, 7], radius=1., sample_count=48)
    assert report["nominal_signal_qualified"]
    assert len(report["visibility_by_frame"]) == 8
    assert report["distinct_geometry_visibility_evaluations"] == 2
    assert report["mean_target_over_radius"] == pytest.approx(.02)


@pytest.fixture
def solved_straight_fea():
    import json
    from pathlib import Path
    pytest.importorskip("scipy")
    from backend.src.config import load_config_dict
    from backend.src.tube import build_tube_static_geometry
    from backend.src.benchmark_mechanics import solve_benchmark_fields
    config_path = Path(__file__).resolve().parents[1] / "configs/tube_fea_c3d6_one_area.json"
    raw = json.loads(config_path.read_text())
    raw["tube"]["deformation"]["fea"].update(axial_elements=4, theta_elements=8)
    cfg = load_config_dict(raw)
    geom = build_tube_static_geometry(cfg.tube, next(iter(cfg.difficulty_presets.values())), np.random.default_rng(512))
    fields, arrays, diagnostics = solve_benchmark_fields(geom, cfg.tube, [dict(s=.5, theta=.25, sigma_s=.2, sigma_theta=.15, direction=[-1, 0, 0])], [.01])
    return cfg, geom, fields, arrays, diagnostics


def test_independent_fea_energy_balance_strain_and_corruption(solved_straight_fea):
    from backend.src.benchmark_validation import audit_c3d6_mechanics
    _, _, _, arrays, _ = solved_straight_fea
    report = audit_c3d6_mechanics(arrays, coefficients=[[0], [1], [2]])
    assert report["passed"]
    assert report["minimum_reference_jacobian_at_gauss_points"] > 0
    assert report["states"][0]["minimum_deformation_jacobian_at_gauss_points"] == 1
    assert report["states"][2]["maximum_green_minus_infinitesimal_strain_norm"] == pytest.approx(4 * report["states"][1]["maximum_green_minus_infinitesimal_strain_norm"])
    corrupted = {**arrays, "basis_gauss_strain": arrays["basis_gauss_strain"] * 1.01}
    assert not audit_c3d6_mechanics(corrupted)["passed"]
    corrupted = {**arrays, "basis_loads": arrays["basis_loads"] * 1.01}
    assert not audit_c3d6_mechanics(corrupted)["passed"]


def test_straight_mapping_and_stiffness_match_rotated_builder(solved_straight_fea):
    from scipy.sparse import csr_matrix, kron, eye
    from backend.src.tube_fea import _load_patch_solver
    cfg, geom, _, arrays, _ = solved_straight_fea
    Builder, Solver = _load_patch_solver()
    tube = cfg.tube
    original = Builder.build_hollow_cylinder_c3d6(length=tube.length, inner_radius=tube.base_radius-tube.shell_thickness, outer_radius=tube.base_radius, n_axial=4, n_theta=8, E=tube.deformation.fea.young_modulus_pa, nu=tube.deformation.fea.poisson_ratio, fix_ends=True)
    Q = np.column_stack((geom.tangent[0], geom.frame_right[0], geom.frame_up[0]))
    assert np.linalg.det(Q) == pytest.approx(1.)
    assert np.allclose(arrays["nodes"], original.nodes @ Q.T + geom.centerline[0], atol=1e-6)
    rotation = kron(eye(len(original.nodes)), Q)
    expected = (rotation @ Solver.assemble_global_K(original) @ rotation.T).tocsr()
    actual = csr_matrix((arrays["K_data"], arrays["K_indices"], arrays["K_indptr"]), shape=tuple(arrays["K_shape"]))
    assert np.linalg.norm((actual-expected).data) / np.linalg.norm(expected.data) < 1e-6
