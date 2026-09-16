"""Independent geometry QA for the tubular benchmark (never recovery scoring).

Ray intersections and projection here deliberately do not call renderer/camera
helpers. World-to-camera is ``(X - t) @ R``; image y points down. Saved float32
rotations are inverted exactly instead of assuming exact orthogonality.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _intr(intr, name):
    return intr[name] if isinstance(intr, dict) else getattr(intr, name)


def _geometry(vertices, triangles):
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite [N,3]")
    if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3:
        raise ValueError("triangles must have shape [M,3]")
    triangles = raw_triangles.astype(np.int64)
    if not np.array_equal(triangles, raw_triangles) or not len(triangles):
        raise ValueError("triangles must contain integer indices and be nonempty")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangle vertex outside mesh")
    return vertices, triangles


def _camera(R_wc, t_wc):
    R = np.asarray(R_wc, dtype=np.float64)
    t = np.asarray(t_wc, dtype=np.float64)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("invalid camera arrays")
    return R, t, np.linalg.inv(R.T)


def _project(points, R, t, intr):
    camera = (points - t) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        u = _intr(intr, "fx") * camera[:, 0] / camera[:, 2] + _intr(intr, "cx")
        v = _intr(intr, "cy") - _intr(intr, "fy") * camera[:, 1] / camera[:, 2]
    return np.column_stack((u, v)), camera[:, 2]


def _ray_hits(vertices, triangles, origin, directions):
    """Closest positive two-sided Moller-Trumbore intersections, one ray at a time.

    Ray direction has camera z=1, so ray parameter equals camera depth.
    No screen bounding box, interpolated z or rasterizer is involved.
    """
    face = vertices[triangles]
    edge1, edge2 = face[:, 1] - face[:, 0], face[:, 2] - face[:, 0]
    offset = origin - face[:, 0]
    q = np.cross(offset, edge1)
    edge_scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    z_numerator = np.einsum("ij,ij->i", edge2, q)
    result_id = np.full(len(directions), -1, dtype=np.int64)
    result_z = np.full(len(directions), np.inf)
    result_bary = np.full((len(directions), 3), np.nan)
    for i, direction in enumerate(directions):
        p = np.cross(direction, edge2)
        det = np.einsum("ij,ij->i", edge1, p)
        good_det = np.abs(det) > 1e-13 * np.maximum(
            edge_scale * np.linalg.norm(direction), 1e-30
        )
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=good_det)
        u = np.einsum("ij,ij->i", offset, p) * inv
        v = np.einsum("ij,j->i", q, direction) * inv
        z = z_numerator * inv
        valid = good_det & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (z > 1e-5)
        z[~valid] = np.inf
        chosen = int(np.argmin(z))
        if np.isfinite(z[chosen]):
            result_id[i] = chosen
            result_z[i] = z[chosen]
            result_bary[i] = [1 - u[chosen] - v[chosen], u[chosen], v[chosen]]
    return result_id, result_z, result_bary


def _directions(pixels, intr, inverse_Rt):
    camera = np.column_stack((
        (pixels[:, 0] - _intr(intr, "cx")) / _intr(intr, "fx"),
        (_intr(intr, "cy") - pixels[:, 1]) / _intr(intr, "fy"),
        np.ones(len(pixels)),
    ))
    return camera @ inverse_Rt.T


def audit_raster_samples(vertices, triangles, R_wc, t_wc, intr,
                         triangle_index, physical_barycentric, depth, *,
                         sample_count=128, seed=0, pixel_tolerance=1e-3,
                         relative_depth_tolerance=1e-5, clean_rgb=None,
                         texture=None, uv_vertices=None):
    """Check fixed random image samples against independent physical rays.

    Optional clean RGB comparison assumes unshaded, untinted nearest-neighbour
    texturing. It must receive pre-photometric RGB, not noisy inference RGB.
    Background samples are included to detect missing/false raster support.
    """
    vertices, triangles = _geometry(vertices, triangles)
    R, t, inverse_Rt = _camera(R_wc, t_wc)
    height, width = int(_intr(intr, "height")), int(_intr(intr, "width"))
    ids, bary, depth = np.asarray(triangle_index), np.asarray(physical_barycentric), np.asarray(depth)
    if ids.shape != (height, width) or depth.shape != ids.shape or bary.shape != (height, width, 3):
        raise ValueError("raster shape differs from declared intrinsics")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    flat = rng.choice(height * width, min(sample_count, height * width), replace=False)
    rows, cols = np.unravel_index(flat, (height, width))
    pixels = np.column_stack((cols + 0.5, rows + 0.5))
    ray_ids, ray_depth, ray_bary = _ray_hits(vertices, triangles, t, _directions(pixels, intr, inverse_Rt))
    observed_ids = ids[rows, cols].astype(np.int64)
    observed_bary, observed_depth = bary[rows, cols], depth[rows, cols]
    failures = []
    max_pixel = max_depth = max_bary = max_rgb = 0.0
    rgb_checked = rgb_boundary = 0
    for i, (row, col) in enumerate(zip(rows, cols)):
        observed_id, ray_id = observed_ids[i], ray_ids[i]
        reason = []
        if observed_id < 0 and ray_id < 0:
            if np.isfinite(observed_depth[i]):
                reason.append("background_has_finite_depth")
        elif observed_id < 0 or ray_id < 0:
            reason.append("physical_visibility_mismatch")
        elif observed_id >= len(triangles) or not np.isfinite(observed_bary[i]).all() or not np.isfinite(observed_depth[i]):
            reason.append("invalid_foreground_label")
        else:
            point = observed_bary[i] @ vertices[triangles[observed_id]]
            projected, point_z = _project(point[None], R, t, intr)
            pixel_error = float(np.linalg.norm(projected[0] - pixels[i]))
            depth_error = float(abs(observed_depth[i] - ray_depth[i]) / max(abs(ray_depth[i]), 1e-12))
            point_depth_error = float(abs(point_z[0] - observed_depth[i]) / max(abs(ray_depth[i]), 1e-12))
            max_pixel, max_depth = max(max_pixel, pixel_error), max(max_depth, depth_error, point_depth_error)
            if pixel_error >= pixel_tolerance:
                reason.append("material_point_reprojection")
            if max(depth_error, point_depth_error) >= relative_depth_tolerance:
                reason.append("frontmost_physical_depth")
            if abs(float(observed_bary[i].sum()) - 1) > 1e-5 or observed_bary[i].min() < -1e-5:
                reason.append("invalid_material_barycentric")
            if observed_id != ray_id:
                # Adjacent triangles can tie exactly at their shared edge.
                if depth_error >= relative_depth_tolerance or min(ray_bary[i]) > 1e-5:
                    reason.append("frontmost_triangle_identity")
            else:
                max_bary = max(max_bary, float(np.max(np.abs(observed_bary[i] - ray_bary[i]))))
            if clean_rgb is not None:
                if texture is None or uv_vertices is None:
                    raise ValueError("clean_rgb audit requires texture and uv_vertices")
                uv = ray_bary[i] @ np.asarray(uv_vertices)[triangles[ray_id]]
                tex_h, tex_w = texture.shape[:2]
                xy = np.asarray([uv[0] * (tex_w - 1), (1 - uv[1]) * (tex_h - 1)])
                # Float32 interpolation may cross a texel boundary within roundoff.
                if np.min(np.abs(xy - np.round(xy))) < 2e-3:
                    rgb_boundary += 1
                else:
                    tx, ty = np.clip(xy.astype(int), [0, 0], [tex_w - 1, tex_h - 1])
                    rgb_error = float(np.max(np.abs(np.asarray(clean_rgb[row, col], float) - texture[ty, tx])))
                    max_rgb = max(max_rgb, rgb_error)
                    rgb_checked += 1
                    if rgb_error > 1:
                        reason.append("clean_rgb_material_texture_mismatch")
        if reason:
            failures.append({"row": int(row), "column": int(col), "reasons": reason})
    return {"passed": not failures, "sample_count": len(flat), "seed": int(seed),
            "ray_foreground_count": int((ray_ids >= 0).sum()), "failures": failures,
            "max_reprojection_error_px": max_pixel, "max_relative_depth_error": max_depth,
            "max_material_barycentric_error": max_bary,
            "pixel_tolerance": pixel_tolerance, "relative_depth_tolerance": relative_depth_tolerance,
            "clean_rgb_checked": rgb_checked, "clean_rgb_texel_boundary_skipped": rgb_boundary,
            "max_clean_rgb_channel_error": max_rgb,
            "scope": "sampled physical ray/render-label QA; not full-frame or recovery validation"}


def audit_mesh_fields(reference_vertices, triangles, vertices_sequence, radius):
    """Whole-field numerical and surface distortion diagnostics, not mechanics certification."""
    reference, triangles = _geometry(reference_vertices, triangles)
    states = np.asarray(vertices_sequence, dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != reference.shape or not len(states):
        raise ValueError("states must be [T,N,3] matching reference")
    if not np.isfinite(states).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("states must be finite and radius positive")
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
    lengths0 = np.linalg.norm(reference[edges[:, 1]] - reference[edges[:, 0]], axis=1)
    normals0 = np.cross(reference[triangles[:, 1]] - reference[triangles[:, 0]], reference[triangles[:, 2]] - reference[triangles[:, 0]])
    areas0 = np.linalg.norm(normals0, axis=1)
    if lengths0.min() <= 1e-12 * radius or areas0.min() <= 1e-12 * radius ** 2:
        raise ValueError("reference contains degenerate edges or triangles")
    frames = []
    for index, state in enumerate(states):
        lengths = np.linalg.norm(state[edges[:, 1]] - state[edges[:, 0]], axis=1)
        normals = np.cross(state[triangles[:, 1]] - state[triangles[:, 0]], state[triangles[:, 2]] - state[triangles[:, 0]])
        area_ratio = np.linalg.norm(normals, axis=1) / areas0
        ratio = lengths / lengths0
        displacement = np.linalg.norm(state - reference, axis=1) / radius
        frames.append({"frame": index, "peak_displacement_over_radius": float(displacement.max()),
                       "mean_displacement_over_radius": float(displacement.mean()),
                       "max_absolute_edge_length_change": float(np.max(np.abs(ratio - 1))),
                       "min_edge_ratio": float(ratio.min()), "max_edge_ratio": float(ratio.max()),
                       "min_triangle_area_ratio": float(area_ratio.min()),
                       "normal_reversal_count": int((np.einsum("ij,ij->i", normals, normals0) < 0).sum()),
                       "degenerate_triangle_count": int((area_ratio <= 1e-10).sum())})
    return {"passed": all(f["degenerate_triangle_count"] == 0 for f in frames), "frames": frames,
            "vertex_count": len(reference), "triangle_count": len(triangles), "unique_edge_count": len(edges),
            "radius": float(radius), "finite": True,
            "scope": "surface geometry only; normal reversal is descriptive; no volume Jacobian, self-intersection, equilibrium or constitutive validity certified"}


def audit_bridge_signal(reference_vertices, triangles, vertices_sequence, R_wc, t_wc, intr,
                        contrast_a, contrast_b, *, radius, sample_count=2048, seed=0,
                        nominal_mean_floor=0.006, nominal_p95_floor=0.015,
                        visibility_floor=0.70):
    """Evaluate fixed bridge image-ray cohort and contrast, with no estimator or tracker.

    Sample all image pixels uniformly before inspecting deformation. Reference
    misses are reported; no foreground or high-motion refill is performed.
    Visibility is audited at every specified contrast frame at the bridge camera.
    All reference hits, including temporarily invisible points, enter target
    magnitude statistics. Thus visibility cannot inflate the declared signal.
    """
    reference, triangles = _geometry(reference_vertices, triangles)
    states = np.asarray(vertices_sequence, dtype=np.float64)
    if states.ndim != 3 or states.shape[1:] != reference.shape or not np.isfinite(states).all():
        raise ValueError("invalid states")
    groups = [np.asarray(contrast_a), np.asarray(contrast_b)]
    if any(g.ndim != 1 or not len(g) or not np.array_equal(g, g.astype(int)) or g.min() < 0 or g.max() >= len(states) for g in groups):
        raise ValueError("contrast indices must be nonempty in-range integer lists")
    if len(np.intersect1d(*groups)) or any(len(np.unique(g)) != len(g) for g in groups):
        raise ValueError("contrasts must be disjoint and have no repeated indices")
    if radius <= 0 or sample_count <= 0:
        raise ValueError("radius and sample_count must be positive")
    R, t, inverse_Rt = _camera(R_wc, t_wc)
    h, w = int(_intr(intr, "height")), int(_intr(intr, "width"))
    rng = np.random.default_rng(seed)
    flat = rng.choice(h * w, min(sample_count, h * w), replace=False)
    rows, cols = np.unravel_index(flat, (h, w))
    pixels = np.column_stack((cols + 0.5, rows + 0.5))
    ids, _, bary = _ray_hits(reference, triangles, t, _directions(pixels, intr, inverse_Rt))
    valid = ids >= 0
    cohort = np.flatnonzero(valid)
    face, weights = triangles[ids[valid]], bary[valid]
    positions = np.einsum("qi,tqij->tqj", weights, states[:, face])
    magnitudes = np.linalg.norm(positions[groups[1].astype(int)].mean(axis=0) - positions[groups[0].astype(int)].mean(axis=0), axis=1) / radius
    common_visible = np.ones(len(cohort), dtype=bool)
    visible_counts = []
    visibility_cache = {}
    for index in np.concatenate(groups).astype(int):
        # Hold phases repeat identical arrays. Cache exact geometry, never an
        # approximate state or a reduced temporal cohort, preserving every row.
        state_key = hashlib.sha256(states[index].tobytes()).digest()
        if state_key not in visibility_cache:
            projected, z = _project(positions[index], R, t, intr)
            in_frame = (z > 1e-5) & (projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)
            _, first_z, _ = _ray_hits(states[index], triangles, t, _directions(projected, intr, inverse_Rt))
            visibility_cache[state_key] = in_frame & (np.abs(first_z - z) <= 1e-5 * np.maximum(np.abs(z), 1e-12))
        visible = visibility_cache[state_key]
        common_visible &= visible
        visible_counts.append({"frame": int(index), "visible_count": int(visible.sum())})
    mean = float(magnitudes.mean()) if len(magnitudes) else None
    p95 = float(np.quantile(magnitudes, 0.95)) if len(magnitudes) else None
    coverage = float(common_visible.mean()) if len(cohort) else 0.0
    qualified = bool(len(cohort) and mean >= nominal_mean_floor and p95 >= nominal_p95_floor and coverage >= visibility_floor)
    return {"nominal_signal_qualified": qualified, "seed": int(seed), "sampled_image_ray_count": len(flat),
            "reference_visible_count": len(cohort), "reference_miss_count": int((~valid).sum()),
            "contrast_a": groups[0].astype(int).tolist(), "contrast_b": groups[1].astype(int).tolist(),
            "mean_target_over_radius": mean, "p95_target_over_radius": p95,
            "max_target_over_radius": float(magnitudes.max()) if len(magnitudes) else None,
            "common_visible_count": int(common_visible.sum()), "common_visible_fraction": coverage,
            "distinct_geometry_visibility_evaluations": len(visibility_cache),
            "visibility_by_frame": visible_counts,
            "criteria": {"nominal_mean_floor": nominal_mean_floor, "nominal_p95_floor": nominal_p95_floor, "visibility_floor": visibility_floor},
            "cohort": {"pixels_xy": pixels[cohort].tolist(), "triangle_ids": ids[valid].tolist(), "material_barycentric": weights.tolist(),
                       "target_magnitude_over_radius": magnitudes.tolist(), "common_visible": common_visible.tolist()},
            "scope": "generation/evaluation-only fixed ray cohort; engineered signal stratum, not identifiability or recovery performance"}


def _wedge_derivatives(r, s, z):
    """Product of triangle barycentric and line shape functions, independent kernel."""
    triangle = np.array([1-r-s, r, s])
    triangle_derivatives = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    result = np.empty((6, 3))
    for layer, sign in enumerate((-1., 1.)):
        sl = slice(3*layer, 3*layer+3)
        result[sl, :2] = .5 * (1+sign*z) * triangle_derivatives
        result[sl, 2] = .5 * sign * triangle
    return result


def audit_c3d6_mechanics(arrays, *, coefficients=None, tolerance=1e-7):
    """Independent reference quadrature, displacement, work and balance checks.

    This audits the saved C3D6 linear solve. Coefficients optionally enumerate
    every rendered state's linear basis mixture for deformation-gradient checks.
    Finite-strain discrepancy and deformed Jacobians are diagnostics, not claims
    that a small-strain solve is faithful large-deformation mechanics.
    """
    from scipy.sparse import csr_matrix
    nodes = np.asarray(arrays["nodes"], float)
    elements = np.asarray(arrays["elements_c3d6"], int)
    fixed = np.asarray(arrays["fixed"], bool)
    displacement = np.asarray(arrays["basis_displacement"], float)
    loads, reactions = np.asarray(arrays["basis_loads"], float), np.asarray(arrays["basis_reactions"], float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or elements.ndim != 2 or elements.shape[1] != 6:
        raise ValueError("expected C3D6 node/element arrays")
    if displacement.ndim != 3 or displacement.shape[1:] != nodes.shape or loads.shape != displacement.shape or reactions.shape != displacement.shape or fixed.shape != nodes.shape:
        raise ValueError("inconsistent mechanical state arrays")
    if not all(np.isfinite(a).all() for a in (nodes, displacement, loads, reactions)):
        raise ValueError("nonfinite mechanics")
    stiffness = csr_matrix((arrays["K_data"], arrays["K_indices"], arrays["K_indptr"]), shape=tuple(arrays["K_shape"]))
    symmetry = stiffness - stiffness.T
    symmetry_error = float(np.linalg.norm(symmetry.data) / max(np.linalg.norm(stiffness.data), 1e-30))
    gauss = [(r, s, z, 1/6) for r, s in ((1/6, 1/6), (2/3, 1/6), (1/6, 2/3)) for z in (-1/np.sqrt(3), 1/np.sqrt(3))]
    min_reference_det = np.inf
    gradients, determinants = [], []
    for element in elements:
        element_grad, element_det = [], []
        for r, s, z, _ in gauss:
            natural = _wedge_derivatives(r, s, z)
            jacobian = nodes[element].T @ natural
            det = float(np.linalg.det(jacobian))
            min_reference_det = min(min_reference_det, det)
            if det <= 0:
                raise ValueError("nonpositive reference C3D6 Jacobian")
            element_grad.append(natural @ np.linalg.inv(jacobian))
            element_det.append(det)
        gradients.append(element_grad)
        determinants.append(element_det)
    gradients, determinants = np.asarray(gradients), np.asarray(determinants)
    E, nu = np.asarray(arrays["E"], float), np.asarray(arrays["nu"], float)
    if E.shape != (len(elements),) or nu.shape != E.shape or np.any(E <= 0) or np.any(nu <= -1) or np.any(nu >= .5):
        raise ValueError("invalid isotropic material arrays")
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    basis_reports = []
    all_grad_u = []
    free = ~fixed.reshape(-1)
    for index, (u, f, reaction) in enumerate(zip(displacement, loads, reactions)):
        grad_u = np.einsum("eia,eqib->eqab", u[elements], gradients)
        all_grad_u.append(grad_u)
        strain = .5 * (grad_u + np.swapaxes(grad_u, -1, -2))
        stress = 2 * mu[:, None, None, None] * strain + lam[:, None, None, None] * np.trace(strain, axis1=-2, axis2=-1)[:, :, None, None] * np.eye(3)
        engineering = np.stack((strain[..., 0, 0], strain[..., 1, 1], strain[..., 2, 2], 2*strain[..., 0, 1], 2*strain[..., 1, 2], 2*strain[..., 0, 2]), axis=-1).reshape(-1, 6)
        stress_components = np.stack((stress[..., 0, 0], stress[..., 1, 1], stress[..., 2, 2], stress[..., 0, 1], stress[..., 1, 2], stress[..., 0, 2]), axis=-1).reshape(-1, 6)
        saved_strain, saved_stress = np.asarray(arrays["basis_gauss_strain"])[index], np.asarray(arrays["basis_gauss_stress"])[index]
        if saved_strain.shape != engineering.shape or saved_stress.shape != engineering.shape:
            raise ValueError("saved Gauss ordering/shape not six points per C3D6")
        strain_error = float(np.linalg.norm(engineering-saved_strain) / max(np.linalg.norm(engineering), 1e-30))
        stress_error = float(np.linalg.norm(stress_components-saved_stress) / max(np.linalg.norm(stress_components), 1e-30))
        energy_per_element = .5 * np.sum(np.einsum("eqij,eqij->eq", strain, stress) * determinants / 6, axis=1)
        energy = float(energy_per_element.sum())
        energy_K = float(.5 * u.reshape(-1) @ (stiffness @ u.reshape(-1)))
        work = float(.5 * np.sum(u*f))
        energy_error = max(abs(energy-energy_K), abs(energy-work), float(np.max(np.abs(energy_per_element-np.asarray(arrays["basis_element_energy"])[index])))) / max(abs(energy), 1e-30)
        residual = (stiffness @ u.reshape(-1) - f.reshape(-1)).reshape(-1, 3)
        force_scale = max(float(np.linalg.norm(f)), 1e-30)
        residual_error = float(np.linalg.norm(residual.reshape(-1)[free]) / force_scale)
        reaction_error = float(np.linalg.norm(residual-reaction) / force_scale)
        # Fixed reactions balance externally applied forces globally.
        supported_reaction = np.where(fixed, reaction, 0)
        net = f + supported_reaction
        force_balance = float(np.linalg.norm(net.sum(axis=0)) / force_scale)
        centre = nodes.mean(axis=0)
        moment_balance = float(np.linalg.norm(np.cross(nodes-centre, net).sum(axis=0)) / max(force_scale*np.max(np.linalg.norm(nodes-centre, axis=1)), 1e-30))
        fixed_error = float(np.max(np.abs(u[fixed]))) if fixed.any() else 0.
        basis_reports.append({"basis": index, "relative_free_equilibrium_error": residual_error,
                              "relative_reaction_error": reaction_error, "relative_force_balance_error": force_balance,
                              "relative_moment_balance_error": moment_balance, "fixed_displacement_max": fixed_error,
                              "relative_gauss_strain_error": strain_error, "relative_gauss_stress_error": stress_error,
                              "relative_energy_error": energy_error, "integrated_strain_energy": energy,
                              "passed": max(residual_error, reaction_error, force_balance, moment_balance, strain_error, stress_error, energy_error) <= tolerance and fixed_error <= 1e-12})
    mixtures = np.eye(len(displacement)) if coefficients is None else np.asarray(coefficients, float)
    if mixtures.ndim != 2 or mixtures.shape[1] != len(displacement) or not np.isfinite(mixtures).all():
        raise ValueError("coefficients must be finite [T,number_of_bases]")
    mixture_reports = []
    basis_grad = np.asarray(all_grad_u)
    for index, weights in enumerate(mixtures):
        grad = np.einsum("b,beqij->eqij", weights, basis_grad)
        infinitesimal = .5 * (grad + np.swapaxes(grad, -1, -2))
        nonlinear_term = .5 * np.swapaxes(grad, -1, -2) @ grad
        det_F = np.linalg.det(np.eye(3)+grad)
        mixture_reports.append({"state": index, "minimum_deformation_jacobian_at_gauss_points": float(det_F.min()),
                                "maximum_absolute_infinitesimal_tensor_strain": float(np.abs(infinitesimal).max()),
                                "maximum_absolute_engineering_shear_strain": float(2*np.max(np.abs(infinitesimal[..., [0, 1, 0], [1, 2, 2]]))),
                                "maximum_displacement_gradient_norm": float(np.linalg.norm(grad, axis=(-1,-2)).max()),
                                "maximum_green_minus_infinitesimal_strain_norm": float(np.linalg.norm(nonlinear_term, axis=(-1,-2)).max()),
                                "relative_green_minus_infinitesimal_strain_norm": float(np.linalg.norm(nonlinear_term)/max(np.linalg.norm(infinitesimal), 1e-30))})
    return {"passed": symmetry_error <= tolerance and all(b["passed"] for b in basis_reports) and all(s["minimum_deformation_jacobian_at_gauss_points"] > 0 for s in mixture_reports),
            "relative_K_symmetry_error": symmetry_error, "minimum_reference_jacobian_at_gauss_points": float(min_reference_det),
            "bases": basis_reports, "states": mixture_reports, "tolerance": tolerance,
            "scope": "independent small-strain discrete-equation arithmetic; positive Jacobians only at quadrature points, not global injectivity or constitutive/anatomical validation"}
