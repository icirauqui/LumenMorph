"""Array contracts and a separate all-triangle ray check for appearance data."""
import numpy as np


def independent_ray_check(vertices, faces, camera, R, C, dense, count=121):
    """Float64 Moller–Trumbore over every triangle; no renderer intersections."""
    h, w = dense['valid'].shape
    indices = np.unique(np.linspace(0, h*w-1, count, dtype=int))
    tri = np.asarray(vertices, dtype=float)[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    offset = np.asarray(C)-tri[:, 0]
    q = np.cross(offset, e1)
    numerator = np.einsum('ij,ij->i', e2, q)
    misses, largest, hits = 0, 0., 0
    for index in indices:
        row, col = divmod(int(index), w)
        # Independent analytic pinhole ray, not camera_models.rays().
        ray = np.array([(col+.5-camera.cx)/camera.fx, -(row+.5-camera.cy)/camera.fy, 1.])
        ray = R@ray
        ray /= np.linalg.norm(ray)
        cross = np.cross(ray, e2)
        det = np.einsum('ij,ij->i', e1, cross)
        safe = np.abs(det) > 1e-15
        inverse = np.zeros_like(det)
        inverse[safe] = 1/det[safe]
        u = inverse*np.einsum('ij,ij->i', offset, cross)
        v = inverse*(q@ray)
        distance = inverse*numerator
        valid = safe & (u >= 0) & (v >= 0) & (u+v <= 1) & (distance > 0)
        actual = bool(dense['valid'][row, col])
        misses += int(bool(valid.any()) != actual)
        if valid.any() and actual:
            point = C+float(distance[valid].min())*ray
            largest = max(largest, float(np.linalg.norm(point-dense['position_world'][row, col])))
            hits += 1
    return {'sampled_rays': len(indices), 'mutual_hits': hits,
            'hit_miss_disagreements': misses, 'max_position_error_meter': largest}


def validate_dense(vertices, faces, camera, R, C, dense):
    good = np.asarray(dense['valid'])
    ids = np.asarray(dense['triangle_id'])
    b = np.asarray(dense['material_barycentric'])
    depth = np.asarray(dense['depth_z'])
    if (good.shape != (camera.height, camera.width) or good.dtype != np.bool_
            or ids.shape != good.shape or not np.issubdtype(ids.dtype, np.integer)
            or b.shape != good.shape+(3,) or depth.shape != good.shape
            or not np.array_equal(good, ids >= 0) or np.any(ids[~good] != -1)
            or np.any(ids[good] >= len(faces)) or not good.any()):
        raise ValueError('Invalid image support or triangle identities')
    if (not np.isfinite(b[good]).all() or np.any(b[good] < -2e-7) or np.any(b[good] > 1+2e-7)
            or np.any(np.abs(b[good].sum(axis=1)-1) > 2e-7)
            or not np.isnan(b[~good]).all() or not np.isnan(depth[~good]).all()
            or not np.isfinite(depth[good]).all() or np.any(depth[good] <= 0)):
        raise ValueError('Invalid barycentrics/depth/background')
    points = np.sum(b[good, :, None]*np.asarray(vertices, dtype=float)[faces[ids[good]]], axis=1)
    world_error = float(np.linalg.norm(points-dense['position_world'][good], axis=1).max())
    pc = (points-C)@R
    z_error = float(np.abs(pc[:, 2]-depth[good]).max())
    uv = np.c_[camera.fx*pc[:, 0]/pc[:, 2]+camera.cx, camera.cy-camera.fy*pc[:, 1]/pc[:, 2]]
    yy, xx = np.nonzero(good)
    projection = float(np.linalg.norm(uv-np.c_[xx+.5, yy+.5], axis=1).max())
    # Float32 grazing intersections can exceed 0.001 px; the complete first-hit
    # census and worst-ray audit justify a 0.01 px numerical representation gate.
    if world_error > 1e-12 or z_error > 1e-12 or projection > .01:
        raise ValueError(f'Geometry/reprojection contract failed: {world_error}, {z_error}, {projection}')
    rays = independent_ray_check(vertices, faces, camera, R, C, dense)
    if rays['hit_miss_disagreements'] or rays['max_position_error_meter'] > 1e-6:
        raise ValueError(f'Independent ray contract failed: {rays}')
    return {'support_fraction': float(good.mean()), 'max_position_readback_error_meter': world_error,
            'max_depth_error_meter': z_error, 'max_reprojection_pixels': projection,
            'independent_rays': rays}
