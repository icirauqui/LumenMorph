"""Canonical RGB and dense evaluator truth for persistent triangular surfaces.

Coordinates are the generator convention: camera x right, y up, z forward;
R_wc maps camera to world; t_wc is the camera center. Pixel samples are
(column + .5, row + .5), and v = cy - fy*y/z. Units follow mesh world units.
Normals are unit world-space face normals following mesh winding, not a
view-dependent shading normal. No ground truth is an inference observation.

The rasterizer deliberately inherits the renderer's near-plane rule: a triangle
with any vertex at z <= 1e-5 is discarded rather than clipped. Missing samples
are invalid and must never be interpreted as zero depth or zero deformation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .renderer import _camera_projection, perspective_correct_barycentric, rasterize_mesh_buffers
from .types import CameraIntrinsics


def render_rgb_and_truth(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    uv_vertices: np.ndarray,
    texture_bgr: np.ndarray,
    intr: CameraIntrinsics,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    bg_color: tuple[int, int, int] = (12, 12, 12),
    *,
    periodic_u: bool = False,
    high_precision: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Render unshaded RGB and truth from one perspective-correct z-buffer.

    ``triangle_id`` and ``material_barycentric`` jointly identify the same
    material point at every time when mesh vertex identity/topology persist.
    Invalid depth/barycentric/normals are NaN; invalid triangle ids are -1.
    Texture interpolation is perspective-correct, with nearest texel sampling,
    the generator's existing UV v reversal, and clamping at texture boundaries.
    ``periodic_u=True`` unwraps seam-crossing triangle U coordinates before
    interpolation and wraps the resulting U modulo one, for welded tubes.
    The default preserves the historical nonperiodic texture convention.
    ``high_precision=True`` promotes geometry/camera arithmetic before any
    transform or projection; dense stored depths/barycentrics remain float32.
    Apply photometric noise to the returned image separately if desired.
    """
    vertices_world = np.asarray(vertices_world)
    triangles = np.asarray(triangles)
    buffers = rasterize_mesh_buffers(
        vertices_world, triangles, intr, R_wc, t_wc, perspective_correct=True,
        high_precision=high_precision,
    )
    valid = buffers.triangle_index >= 0
    face_vertices = triangles[buffers.triangle_index[valid]]
    vertices_cam, _, _ = _camera_projection(vertices_world, intr, R_wc, t_wc,
                                           high_precision=high_precision)
    bary = np.full((*valid.shape, 3), np.nan, dtype=np.float32)
    bary[valid] = perspective_correct_barycentric(
        buffers.screen_barycentric[valid], vertices_cam[face_vertices, 2]
    ).astype(np.float32)
    # Both RGB and all evaluator fields use these same stored material weights.
    triangle_uv = np.asarray(uv_vertices)[face_vertices].copy()
    if periodic_u:
        u = triangle_uv[:, :, 0]
        u %= 1.
        crossing = np.ptp(u, axis=1) > .5
        u += (crossing[:, None] & (u < .5)).astype(u.dtype)
    uv = np.sum(bary[valid, :, None] * triangle_uv, axis=1)
    if periodic_u:
        uv[:, 0] %= 1.
    tex_h, tex_w = texture_bgr.shape[:2]
    # Retain the historical width-minus-one texel convention in periodic mode;
    # changing to floor(U*width) would resample every nonseam triangle as well.
    tx = np.clip((uv[:, 0] * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
    ty = np.clip(((1.0 - uv[:, 1]) * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
    image = np.empty((*valid.shape, 3), dtype=np.uint8)
    image[:] = bg_color
    image[valid] = texture_bgr[ty, tx]

    face_points = vertices_world[triangles].astype(np.float64)
    normals = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 1e-15)
    normals[lengths <= 1e-15] = np.nan
    normal_world = np.full((*valid.shape, 3), np.nan, dtype=np.float32)
    normal_world[valid] = normals[buffers.triangle_index[valid]]
    depth = np.where(valid, buffers.depth, np.nan).astype(np.float32)
    return image, {
        "depth_z": depth,
        "triangle_id": buffers.triangle_index,
        "material_barycentric": bary,
        "normal_world": normal_world,
        "valid": valid,
    }


def query_surface_depth(
    uv: np.ndarray,
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    intr: CameraIntrinsics,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    *,
    high_precision: bool = False,
) -> np.ndarray:
    """Continuous target z-buffer at arbitrary subpixel image coordinates.

    This tests projected triangles at the actual query locations, never using
    nearest-pixel depth as a surrogate for visibility. Spatial query indexing
    accelerates candidate selection without altering the geometric test.
    UV must be finite. Queries with no rendered surface return infinity.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    result = np.full(len(uv), np.inf, dtype=np.float64)
    if len(uv) == 0:
        return result
    if not np.all(np.isfinite(uv)):
        raise ValueError("Surface depth queries must be finite")
    query_tree = cKDTree(uv)
    pc, projected, front = _camera_projection(vertices_world, intr, R_wc, t_wc,
                                             high_precision=high_precision)
    for tri in np.asarray(triangles):
        z = pc[tri, 2].astype(np.float64)
        if not np.all(front[tri]) or np.min(z) <= 1e-5:
            continue
        p = projected[tri].astype(np.float64)
        lo, hi = np.min(p, axis=0), np.max(p, axis=0)
        center = (lo + hi) / 2
        radius = float(np.linalg.norm(hi - lo) / 2) + 1e-9
        ids = np.asarray(query_tree.query_ball_point(center, radius), dtype=np.int64)
        if len(ids) == 0:
            continue
        q = uv[ids]
        bbox = np.all((q >= lo - 1e-7) & (q <= hi + 1e-7), axis=1)
        ids, q = ids[bbox], q[bbox]
        if len(ids) == 0:
            continue
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-8:
            continue
        w0 = ((y1 - y2) * (q[:, 0] - x2) + (x2 - x1) * (q[:, 1] - y2)) / den
        w1 = ((y2 - y0) * (q[:, 0] - x2) + (x0 - x2) * (q[:, 1] - y2)) / den
        w2 = 1 - w0 - w1
        # Small edge tolerance covers float32 endpoint projection roundoff.
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        reciprocal = w0 / z[0] + w1 / z[1] + w2 / z[2]
        inside &= reciprocal > 0
        ids = ids[inside]
        result[ids] = np.minimum(result[ids], 1 / reciprocal[inside])
    return result


def flow_to_frame(
    source_truth: dict[str, np.ndarray],
    target_vertices_world: np.ndarray,
    triangles: np.ndarray,
    intr: CameraIntrinsics,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    *,
    depth_atol: float = 1e-5,
    depth_rtol: float = 1e-5,
    high_precision: bool = False,
) -> dict[str, np.ndarray]:
    """Advect source pixels by fixed material identity; compute exact visibility.

    Flow is destination minus source in (u,v) pixels. It remains defined for
    occluded/out-of-image points in front of the target camera. ``flow_valid``
    means a finite positive-depth projection, not visibility. ``visible`` also
    requires image bounds and agreement with continuous target surface depth.
    ``unresolved`` records in-image endpoints not consistent with the rendered
    target surface (e.g. a discarded near-plane triangle); these are not called
    occlusions. ``depth_atol`` is in mesh units; ``depth_rtol`` is dimensionless.
    Use ``high_precision=True`` when source truth was rendered with that mode;
    it applies float64 camera arithmetic both to endpoints and visibility rays.
    """
    valid = source_truth["valid"]
    h, w = valid.shape
    if (h, w) != (intr.height, intr.width):
        raise ValueError("Source and destination images must have matching size")
    tri_ids = source_truth["triangle_id"][valid]
    face_vertices = np.asarray(triangles)[tri_ids]
    bary = source_truth["material_barycentric"][valid].astype(np.float64)
    points = np.sum(bary[:, :, None] * np.asarray(target_vertices_world)[face_vertices], axis=1)
    pc, target_uv, front = _camera_projection(points, intr, R_wc, t_wc,
                                             high_precision=high_precision)
    front &= np.all(np.isfinite(target_uv), axis=1) & np.isfinite(pc[:, 2])
    rows, cols = np.nonzero(valid)
    source_uv = np.column_stack((cols + .5, rows + .5))
    flow = np.full((h, w, 2), np.nan, dtype=np.float32)
    values = target_uv - source_uv
    values[~front] = np.nan
    flow[valid] = values
    inside = front & (target_uv[:, 0] >= 0) & (target_uv[:, 0] < w) & (target_uv[:, 1] >= 0) & (target_uv[:, 1] < h)
    z = pc[:, 2].astype(np.float64)
    target_depth = np.full(len(points), np.inf)
    target_depth[inside] = query_surface_depth(
        target_uv[inside], target_vertices_world, triangles, intr, R_wc, t_wc,
        high_precision=high_precision,
    )
    tolerance = depth_atol + depth_rtol * np.abs(z)
    visible = inside & np.isfinite(target_depth) & (np.abs(target_depth - z) <= tolerance)
    occluded = inside & np.isfinite(target_depth) & (z > target_depth + tolerance)
    result = {"flow_uv": flow}
    masks = {
        "flow_valid": front,
        "visible": visible,
        "occluded": occluded,
        "out_of_frame": front & ~inside,
        "behind_camera": ~front,
        "unresolved": inside & ~visible & ~occluded,
    }
    for key, values in masks.items():
        mask = np.zeros((h, w), dtype=bool)
        mask[valid] = values
        result[key] = mask
    return result
