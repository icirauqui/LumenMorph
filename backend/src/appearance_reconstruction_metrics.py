"""Posthoc evaluator for the retained fork's binary models; never inference input.

Camera-center Sim(3) fixes only monocular gauge. Metric error below refers to
the assigned render-world scale, not independently estimated physical scale.
"""
from pathlib import Path
import struct

import numpy as np

F = np.diag([1., -1., 1.])


def rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid quaternion')
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _unpack(stream, fmt):
    size = struct.calcsize(fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated fork binary model')
    return struct.unpack(fmt, data)


def read_images(path):
    """Fork writer: uint32 image/camera IDs, float64 poses, 32-byte keypoint rows."""
    images, image_ids = {}, set()
    with Path(path).open('rb') as stream:
        count, = _unpack(stream, '<Q')
        for _ in range(count):
            image_id, *values = _unpack(stream, '<I7dI')
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError('Unterminated image filename')
                if char == b'\0':
                    break
                name.extend(char)
            name = name.decode('utf8')
            if name in images or image_id in image_ids:
                raise ValueError('Duplicate image filename or ID')
            image_ids.add(image_id)
            n, = _unpack(stream, '<Q')
            data = stream.read(n*32)
            if len(data) != n*32:
                raise ValueError('Truncated extended keypoint rows')
            R = rotation(values[:4])
            t = np.asarray(values[4:7])
            if not np.isfinite(t).all():
                raise ValueError('Nonfinite translation')
            images[name] = {'image_id': image_id, 'camera_id': values[7], 'C': -R.T@t, 'R_wc': R.T}
        if stream.read(1):
            raise ValueError('Trailing bytes or wrong fork binary image format')
    return images


def read_points(path):
    """Read rigid retained points3D, not the separate discarded-points export."""
    records, seen = [], set()
    with Path(path).open('rb') as stream:
        n, = _unpack(stream, '<Q')
        for _ in range(n):
            point_id, x, y, z, red, green, blue, error = _unpack(stream, '<Q3d3Bd')
            if point_id in seen or not np.isfinite([x, y, z, error]).all():
                raise ValueError('Duplicate/nonfinite model point')
            seen.add(point_id)
            length, = _unpack(stream, '<Q')
            if len(stream.read(length*8)) != length*8:
                raise ValueError('Truncated point track')
            records.append((point_id, [x, y, z]))
        if stream.read(1):
            raise ValueError('Trailing point-model bytes')
    records.sort()
    return np.array([p[0] for p in records], dtype=np.uint64), np.asarray([p[1] for p in records], dtype=float).reshape(-1, 3)


def fit_similarity(estimated, target):
    x, y = np.asarray(estimated, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError('Need at least three paired centers')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite centers')
    xc, yc = x-x.mean(axis=0), y-y.mean(axis=0)
    sx, sy = np.linalg.svd(xc, compute_uv=False), np.linalg.svd(yc, compute_uv=False)
    if min(sx[0], sy[0]) < 1e-12:
        raise ValueError('Stationary centers do not fix a similarity')
    ratios = [float(sx[1]/sx[0]), float(sy[1]/sy[0])]
    if min(ratios) < 1e-6:
        raise ValueError('Near-collinear centers: orientation/gauge is ill-conditioned')
    U, singular, Vt = np.linalg.svd(yc.T@xc/len(x))
    sign = np.array([1., 1., 1. if np.linalg.det(U@Vt) > 0 else -1.])
    R = U@np.diag(sign)@Vt
    scale = float(np.dot(singular, sign)/(np.sum(xc*xc)/len(x)))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Invalid fitted scale')
    return {'scale': scale, 'R': R, 't': y.mean(axis=0)-scale*R@x.mean(axis=0),
            'noncollinearity_ratios': ratios}


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite score')
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'rmse': float(np.sqrt(np.mean(a*a))), 'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def pose_metrics(images, expected, states, support=None):
    names = sorted(set(images) & set(expected))
    if support is not None:
        names = sorted(set(names) & set(support))
    result = {'expected_count': len(expected), 'registered_expected': len(set(images)&set(expected)),
        'registered_fraction': len(set(images)&set(expected))/len(expected),
        'scored_names': names, 'missing_names': sorted(set(expected)-set(images)),
        'unexpected_names': sorted(set(images)-set(expected)),
        'gauge': 'Proper Sim(3) fitted on scored camera centers; scale comes from evaluator truth.'}
    idx = np.array([int(Path(n).stem) for n in names], dtype=int)
    estimated = np.asarray([images[n]['C'] for n in names]).reshape(-1, 3)
    target = np.asarray(states['C_world_meter'])[idx]@F.T
    try:
        fit = fit_similarity(estimated, target)
    except ValueError as exc:
        return {**result, 'unavailable_reason': str(exc)}, None
    predicted = fit['scale']*(estimated@fit['R'].T)+fit['t']
    error_mm = 1000*np.linalg.norm(predicted-target, axis=1)
    angles = []
    for name, i in zip(names, idx):
        truth_R = F@states['R_wc'][i]@F
        delta = truth_R.T@fit['R']@images[name]['R_wc']
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))))
    result.update(ate_mm=describe(error_mm), orientation_degrees=describe(angles),
        similarity={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()},
        per_frame=[{'image': n, 'ate_mm': float(e), 'orientation_degrees': a}
                   for n, e, a in zip(names, error_mm, angles)])
    return result, fit


def point_triangle_squared(point, triangles):
    """Exact candidates: plane projection inside triangle, otherwise its edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v, w = b-a, c-a, point-a
    n = np.cross(u, v)
    nn = np.einsum('ij,ij->i', n, n)
    if np.any(nn <= 0):
        raise ValueError('Degenerate truth triangle')
    signed = np.einsum('ij,ij->i', w, n)
    first = np.einsum('ij,ij->i', np.cross(w, v), n)/nn
    second = np.einsum('ij,ij->i', np.cross(u, w), n)/nn
    inside = (first >= 0) & (second >= 0) & (first+second <= 1)
    distance = np.where(inside, signed*signed/nn, np.inf)
    for begin, end in [(a, b), (b, c), (c, a)]:
        edge = end-begin
        fraction = np.clip(np.einsum('ij,ij->i', point-begin, edge)/np.einsum('ij,ij->i', edge, edge), 0, 1)
        residual = point-(begin+fraction[:, None]*edge)
        distance = np.minimum(distance, np.einsum('ij,ij->i', residual, residual))
    return distance


def surface_distances(points, vertices, faces):
    """Exact triangle distances after conservative sphere pruning.

    A nearest mesh vertex supplies an upper distance bound. A triangle can beat
    it only if its center lies within upper_bound+triangle_radius. The tree uses
    the maximum radius, then per-triangle spheres prune conservatively. No
    fixed-k nearest-centroid approximation is substituted for surface distance.
    """
    from scipy.spatial import cKDTree
    p, v = np.asarray(points, dtype=float), np.asarray(vertices, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Expected finite model points')
    triangles = v[faces]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles-centers[:, None], axis=2).max(axis=1)
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError('Invalid truth surface')
    vertex_tree, center_tree = cKDTree(v[np.unique(faces)]), cKDTree(centers)
    upper, _ = vertex_tree.query(p)
    padding = 1e-12*max(1., float(np.abs(v).max()), float(np.abs(p).max()) if len(p) else 0.)
    distances = []
    for point, bound in zip(p, upper):
        candidates = np.asarray(center_tree.query_ball_point(point, bound+float(radii.max())+padding), dtype=int)
        candidates = candidates[np.linalg.norm(centers[candidates]-point, axis=1) <= bound+radii[candidates]+padding]
        if not len(candidates):
            raise ValueError('Conservative nearest-surface candidate set is empty')
        distances.append(float(np.sqrt(point_triangle_squared(point, triangles[candidates]).min())))
    return np.asarray(distances)


def evaluate_study(dataset, study, output):
    """Score completed inference only; retain unavailable metrics and fixed support."""
    import hashlib
    import json
    from scipy.spatial import cKDTree
    dataset, study, output = Path(dataset).resolve(), Path(study).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite completed or partial evaluation')
    execution = json.loads((study/'EXECUTION_COMPLETE.json').read_text())
    if not execution.get('all_20_conditions_accounted_for') or not execution.get('repeat_control_completed'):
        raise ValueError('All fixed inference conditions must be accounted for before evaluation')
    release = json.loads((dataset/'RELEASE_VALIDATION.json').read_text())
    hashes = release['verified_file_sha256']
    consumed = {}

    def checked_npz(relative):
        path = dataset/relative
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != hashes[relative]:
            raise ValueError('Evaluator input differs from independent validation: '+relative)
        consumed[relative] = digest
        with np.load(path, allow_pickle=False) as a:
            return {k: a[k] for k in a.files}

    rows, pairs = [], []
    for family in ['f01', 'f04', 'f05', 'f06']:
        states = checked_npz(f'families/{family}/evaluation/states.npz')
        geometry = checked_npz(f'families/{family}/evaluation/geometry.npz')
        vertices, faces = states['vertices_meter'][60].astype(float), geometry['triangles']
        if not all(np.array_equal(vertices, states['vertices_meter'][i]) for i in range(60, 75)):
            raise ValueError('Declared local plateau is not geometrically constant')
        support = []
        for frame in range(60, 75):
            dense = checked_npz(f'families/{family}/evaluation/dense/{frame:06d}.npz')
            valid = dense['valid'][::16, ::16]
            ids = dense['triangle_id'][::16, ::16][valid]
            b = dense['material_barycentric'][::16, ::16][valid]
            support.append(np.sum(b[..., None]*vertices[faces[ids]], axis=1)@F.T)
        support = np.concatenate(support)
        if not len(support):
            raise ValueError('No fixed visible-surface support')
        eligible = {}
        conditions = ['diffuse_symmetric', 'diffuse_asymmetric', 'coated_symmetric',
                      'diffuse_exposure_half', 'diffuse_exposure_double']
        if family == 'f01':
            conditions.append('diffuse_symmetric_repeat')
        for condition in conditions:
            job = study/'jobs'/(family+'__'+condition)
            report = json.loads((job/'REPORT.json').read_text())
            if not report.get('execution_finished') or not report.get('inputs_unchanged'):
                raise ValueError('Interrupted or changed-input inference is not scoreable')
            for label in ['global', 'local']:
                expected = [f'{i:06d}.png' for i in (range(90) if label == 'global' else range(60, 75))]
                row = {'family': family, 'condition': condition, 'model': label,
                       'repeat_control': condition == 'diffuse_symmetric_repeat',
                       'expected_count': len(expected)}
                model = report['models'].get(label)
                if model is None or not model['mapper_completed']:
                    row.update(status='not_scoreable', reason=report.get('pipeline_stop', 'mapper_failed_or_timed_out'))
                    rows.append(row); continue
                candidates = model['candidates']
                if len(candidates) == 0:
                    row.update(status='completed_without_model', registered_fraction=0., registered_expected=0)
                    rows.append(row); continue
                if len(candidates) != 1 or not candidates[0]['converted'] or candidates[0]['parse_error']:
                    row.update(status='not_scoreable', reason='ambiguous_or_unreadable_model')
                    rows.append(row); continue
                candidate = candidates[0]
                path = job/candidate['path']
                for name, expected_hash in candidate['files'].items():
                    with (path/name).open('rb') as stream:
                        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_hash:
                            raise ValueError('Model changed after inference: '+str(path/name))
                images = read_images(path/'images.bin')
                score, fit = pose_metrics(images, expected, states)
                row.update(status='completed_model', poses=score,
                    registered_fraction=score['registered_fraction'], registered_expected=score['registered_expected'])
                eligible[(condition, label)] = images
                ids, points = read_points(path/'points3D.bin')
                row['point_count'] = len(points)
                if label == 'local' and fit is not None and len(points):
                    aligned = fit['scale']*(points@fit['R'].T)+fit['t']
                    precision = surface_distances(aligned, vertices@F.T, faces)
                    coverage, _ = cKDTree(aligned).query(support)
                    row['local_geometry'] = {
                        'surface_accuracy_mm': describe(1000*precision),
                        'fixed_visible_support_nearest_point_mm': describe(1000*coverage),
                        'support_rule': 'Every 16th pixel in both axes, all 15 local frames, shared geometric validity, regardless of method registration.',
                        'precision_rule': 'All retained rigid points3D; unsigned distance to the entire constant local truth mesh.',
                        'scope': 'Sparse point-to-surface accuracy and image-grid sampling coverage, not a complete reconstructed surface.'}
                elif label == 'local':
                    row['local_geometry'] = {'unavailable_reason': 'No identified gauge or no retained points'}
                rows.append(row)
        for label in ['global', 'local']:
            expected = [f'{i:06d}.png' for i in (range(90) if label == 'global' else range(60, 75))]
            baseline = eligible.get(('diffuse_symmetric', label))
            for condition in conditions[1:]:
                variant = eligible.get((condition, label))
                pair = {'family': family, 'model': label, 'contrast': condition+' minus diffuse_symmetric'}
                if baseline is None or variant is None:
                    pairs.append({**pair, 'unavailable_reason': 'One or both methods lack a completed readable model'})
                    continue
                common = sorted(set(baseline)&set(variant)&set(expected))
                a, _ = pose_metrics(baseline, expected, states, common)
                b, _ = pose_metrics(variant, expected, states, common)
                pair.update(common_names=common, baseline=a, variant=b)
                if 'ate_mm' in a and 'ate_mm' in b:
                    pair['ate_rmse_difference_mm'] = b['ate_mm']['rmse']-a['ate_mm']['rmse']
                    pair['orientation_mean_difference_degrees'] = b['orientation_degrees']['mean']-a['orientation_degrees']['mean']
                pairs.append(pair)
    output.mkdir(parents=True)
    result = {'scope': 'Fixed paired appearance reconstruction study; synthetic assigned geometry, no tissue/clinical transfer claim.',
        'rows': rows, 'paired_common_support': pairs, 'consumed_evaluator_inputs': consumed,
        'aggregation': 'Report all four families separately. f01 development; f04 validation; f05/f06 original final-test families. No frame-level anatomical sample inflation.',
        'method_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('dataset', 'study', 'output'):
        parser.add_argument('--'+argument, type=Path, required=True)
    args = parser.parse_args()
    evaluate_study(args.dataset, args.study, args.output)
