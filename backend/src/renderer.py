from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import project_points, world_to_camera
from .types import CameraIntrinsics


_PREVIEW_LIGHT_DIR = np.asarray([-0.25, -0.45, -0.86], dtype=np.float32)
_PREVIEW_LIGHT_DIR = _PREVIEW_LIGHT_DIR / max(float(np.linalg.norm(_PREVIEW_LIGHT_DIR)), 1e-8)


@dataclass(frozen=True)
class MeshRasterBuffers:
    """Geometry buffers produced with the generator's exact raster convention.

    ``screen_barycentric`` contains affine screen-space weights. Those weights are not
    physical surface barycentrics when triangle vertex depths differ; use
    :func:`perspective_correct_barycentric` for ray-consistent geometry.
    """

    depth: np.ndarray
    triangle_index: np.ndarray
    screen_barycentric: np.ndarray


def _camera_oriented_vertex_normals(pts_cam: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(pts_cam, dtype=np.float32)
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0 = pts_cam[i0]
        p1 = pts_cam[i1]
        p2 = pts_cam[i2]
        normal = np.cross(p1 - p0, p2 - p0).astype(np.float32)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            continue
        normal = normal / norm
        centroid = (p0 + p1 + p2) / 3.0
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 1e-8 and float(np.dot(normal, centroid / centroid_norm)) > 0.0:
            normal = -normal
        normals[i0] += normal
        normals[i1] += normal
        normals[i2] += normal

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    safe = norms[:, 0] > 1e-8
    normals[safe] = normals[safe] / norms[safe]
    normals[~safe] = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return normals.astype(np.float32)


def _shade_map(
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    n0: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
) -> np.ndarray:
    normal = w0[:, :, None] * n0 + w1[:, :, None] * n1 + w2[:, :, None] * n2
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(norm, 1e-8)
    diffuse = np.maximum(0.0, normal @ _PREVIEW_LIGHT_DIR)
    return np.clip(0.62 + 0.38 * diffuse, 0.55, 1.08).astype(np.float32)


def _barycentric_grid(
    px: np.ndarray,
    py: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    high_precision: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-8:
        shape = px.shape
        z = np.zeros(shape, dtype=np.float64 if high_precision else np.float32)
        return z, z, z, np.zeros(shape, dtype=bool)

    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    dtype = np.float64 if high_precision else np.float32
    return w0.astype(dtype), w1.astype(dtype), w2.astype(dtype), inside


def perspective_correct_barycentric(
    screen_barycentric: np.ndarray, vertex_depths: np.ndarray
) -> np.ndarray:
    """Convert projected affine weights to barycentrics on a 3D triangle."""

    screen = np.asarray(screen_barycentric, dtype=np.float64)
    depths = np.asarray(vertex_depths, dtype=np.float64)
    if screen.shape[-1] != 3 or depths.shape[-1] != 3:
        raise ValueError("screen_barycentric and vertex_depths must end in length 3")
    if np.any(depths <= 0.0):
        raise ValueError("perspective correction requires positive vertex depths")
    weights = screen / depths
    denominator = np.sum(weights, axis=-1, keepdims=True)
    if np.any(np.abs(denominator) <= 1e-15):
        raise ValueError("degenerate perspective-correction denominator")
    return (weights / denominator).astype(np.float64)


def _perspective_correct_grid(
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    z0: float,
    z1: float,
    z2: float,
    *,
    high_precision: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical barycentrics and reciprocal-interpolated camera depth."""

    q0 = np.asarray(w0, dtype=np.float64) / z0
    q1 = np.asarray(w1, dtype=np.float64) / z1
    q2 = np.asarray(w2, dtype=np.float64) / z2
    reciprocal_depth = q0 + q1 + q2
    valid = np.isfinite(reciprocal_depth) & (reciprocal_depth > 1e-15)
    depth = np.full(reciprocal_depth.shape, np.inf, dtype=np.float64)
    np.divide(1.0, reciprocal_depth, out=depth, where=valid)
    p0 = np.zeros_like(depth)
    p1 = np.zeros_like(depth)
    p2 = np.zeros_like(depth)
    np.multiply(q0, depth, out=p0, where=valid)
    np.multiply(q1, depth, out=p1, where=valid)
    np.multiply(q2, depth, out=p2, where=valid)
    dtype = np.float64 if high_precision else np.float32
    return p0.astype(dtype), p1.astype(dtype), p2.astype(dtype), depth.astype(dtype)


def _camera_projection(vertices_world, intr, R_wc, t_wc, *, high_precision=False):
    """Optional accurate arithmetic on exact stored vertex/camera values.

    The float32 rotation is not orthogonalized: camera coordinates still obey
    the declared transpose transform exactly. High precision promotes inputs
    before subtraction/multiplication and avoids float32 projected coordinates.
    """
    if not high_precision:
        pts_cam = world_to_camera(vertices_world, R_wc, t_wc)
        uv, valid = project_points(pts_cam, intr)
        return pts_cam, uv, valid
    pts_cam = ((np.asarray(vertices_world, dtype=np.float64)-np.asarray(t_wc, dtype=np.float64))
               @ np.asarray(R_wc, dtype=np.float64))
    valid = pts_cam[:, 2] > 1e-6
    uv = np.full((len(pts_cam), 2), -1., dtype=np.float64)
    uv[valid, 0] = intr.fx*pts_cam[valid, 0]/pts_cam[valid, 2]+intr.cx
    uv[valid, 1] = intr.cy-intr.fy*pts_cam[valid, 1]/pts_cam[valid, 2]
    return pts_cam, uv, valid


def rasterize_mesh_buffers(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    intr: CameraIntrinsics,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    *,
    perspective_correct: bool = False,
    high_precision: bool = False,
) -> MeshRasterBuffers:
    """Rasterize depth, triangle identity, and screen-space barycentrics.

    Pixel samples are evaluated at ``(column + 0.5, row + 0.5)`` and depth is
    affine by default, preserving audits of historical images. Explicitly set
    ``perspective_correct=True`` for the current :func:`render_mesh` convention:
    reciprocal-interpolated camera z and ray-consistent triangle ownership.
    Returned screen barycentrics remain affine in both modes.
    ``high_precision=True`` computes camera coordinates, projected vertices,
    screen weights, perspective correction, and z-buffer comparisons in float64.
    Defaults preserve the historical float32 arithmetic and output types.
    """

    h = intr.height
    w = intr.width
    dtype = np.float64 if high_precision else np.float32
    z_buffer = np.full((h, w), np.inf, dtype=dtype)
    triangle_index = np.full((h, w), -1, dtype=np.int32)
    barycentric = np.full((h, w, 3), np.nan, dtype=dtype)

    pts_cam, uv_pix, valid = _camera_projection(vertices_world, intr, R_wc, t_wc,
                                               high_precision=high_precision)

    for triangle_id, tri in enumerate(triangles):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue

        z0, z1, z2 = float(pts_cam[i0, 2]), float(pts_cam[i1, 2]), float(pts_cam[i2, 2])
        if min(z0, z1, z2) <= 1e-5:
            continue

        x0, y0 = float(uv_pix[i0, 0]), float(uv_pix[i0, 1])
        x1, y1 = float(uv_pix[i1, 0]), float(uv_pix[i1, 1])
        x2, y2 = float(uv_pix[i2, 0]), float(uv_pix[i2, 1])
        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), w - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), h - 1)
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=dtype) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=dtype) + 0.5
        px, py = np.meshgrid(xs, ys)
        w0, w1, w2, inside = _barycentric_grid(px, py, x0, y0, x1, y1, x2, y2,
                                             high_precision=high_precision)
        if not np.any(inside):
            continue
        if perspective_correct:
            _, _, _, depth = _perspective_correct_grid(w0, w1, w2, z0, z1, z2,
                                                       high_precision=high_precision)
        else:
            depth = w0 * z0 + w1 * z1 + w2 * z2
        patch_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        depth_mask = inside & (depth < patch_z)
        if not np.any(depth_mask):
            continue

        patch_z[depth_mask] = depth[depth_mask]
        patch_triangle = triangle_index[min_y : max_y + 1, min_x : max_x + 1]
        patch_triangle[depth_mask] = triangle_id
        patch_bary = barycentric[min_y : max_y + 1, min_x : max_x + 1]
        patch_bary[depth_mask] = np.stack(
            (w0[depth_mask], w1[depth_mask], w2[depth_mask]), axis=1
        )

    return MeshRasterBuffers(
        depth=z_buffer,
        triangle_index=triangle_index,
        screen_barycentric=barycentric,
    )


def render_mesh(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    uv_vertices: np.ndarray,
    texture_bgr: np.ndarray,
    intr: CameraIntrinsics,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    bg_color: tuple[int, int, int] = (12, 12, 12),
    rigid_weight_vertices: np.ndarray | None = None,
    rigid_tint_bgr: tuple[int, int, int] | None = None,
    rigid_tint_strength: float = 0.0,
    shade_faces: bool = False,
) -> np.ndarray:
    h = intr.height
    w = intr.width

    image = np.empty((h, w, 3), dtype=np.uint8)
    image[:, :] = np.asarray(bg_color, dtype=np.uint8)
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)

    pts_cam = world_to_camera(vertices_world, R_wc, t_wc)
    uv_pix, valid = project_points(pts_cam, intr)
    vertex_normals_cam = _camera_oriented_vertex_normals(pts_cam, triangles) if shade_faces else None
    tint_enabled = (
        rigid_weight_vertices is not None
        and rigid_tint_bgr is not None
        and float(rigid_tint_strength) > 1e-6
    )
    if tint_enabled:
        tint = np.asarray(rigid_tint_bgr, dtype=np.float32).reshape(1, 1, 3)
        rigid_w = rigid_weight_vertices.astype(np.float32)

    tex_h, tex_w = texture_bgr.shape[:2]
    flat_texture = tex_h == 1 and tex_w == 1
    if flat_texture:
        flat_color_u8 = texture_bgr[0, 0].astype(np.uint8)
        flat_color = flat_color_u8.astype(np.float32).reshape(1, 1, 3)

    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue

        z0, z1, z2 = float(pts_cam[i0, 2]), float(pts_cam[i1, 2]), float(pts_cam[i2, 2])
        if min(z0, z1, z2) <= 1e-5:
            continue

        x0, y0 = float(uv_pix[i0, 0]), float(uv_pix[i0, 1])
        x1, y1 = float(uv_pix[i1, 0]), float(uv_pix[i1, 1])
        x2, y2 = float(uv_pix[i2, 0]), float(uv_pix[i2, 1])

        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), w - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), h - 1)

        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys)

        w0, w1, w2, inside = _barycentric_grid(px, py, x0, y0, x1, y1, x2, y2)
        if not np.any(inside):
            continue

        p0, p1, p2, depth = _perspective_correct_grid(
            w0, w1, w2, z0, z1, z2
        )

        patch_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        depth_mask = inside & (depth < patch_z)
        if not np.any(depth_mask):
            continue

        patch_img = image[min_y : max_y + 1, min_x : max_x + 1]
        shade = None
        if vertex_normals_cam is not None:
            shade = _shade_map(
                p0,
                p1,
                p2,
                vertex_normals_cam[i0],
                vertex_normals_cam[i1],
                vertex_normals_cam[i2],
            )

        if flat_texture and not tint_enabled:
            if shade is not None:
                shaded = flat_color * shade[:, :, None]
                patch_img[depth_mask] = np.clip(shaded[depth_mask], 0, 255).astype(np.uint8)
            else:
                patch_img[depth_mask] = flat_color_u8
            patch_z[depth_mask] = depth[depth_mask]
            continue

        if flat_texture:
            sampled = np.empty((*depth.shape, 3), dtype=np.float32)
            sampled[:, :] = flat_color
        else:
            tri_uv = uv_vertices[[i0, i1, i2]]
            uu = p0 * tri_uv[0, 0] + p1 * tri_uv[1, 0] + p2 * tri_uv[2, 0]
            vv = p0 * tri_uv[0, 1] + p1 * tri_uv[1, 1] + p2 * tri_uv[2, 1]

            tx = np.clip((uu * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
            ty = np.clip(((1.0 - vv) * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
            sampled = texture_bgr[ty, tx].astype(np.float32)

        if tint_enabled:
            rw = p0 * rigid_w[i0] + p1 * rigid_w[i1] + p2 * rigid_w[i2]
            alpha = np.clip(rw * float(rigid_tint_strength), 0.0, 1.0)[:, :, None]
            sampled = sampled * (1.0 - alpha) + tint * alpha

        if shade is not None:
            sampled = sampled * shade[:, :, None]

        patch_img[depth_mask] = sampled[depth_mask].astype(np.uint8)
        patch_z[depth_mask] = depth[depth_mask]

    return image
