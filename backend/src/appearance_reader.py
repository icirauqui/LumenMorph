"""Portable public-only and explicitly privileged appearance-suite readers."""
from pathlib import Path
import json

import cv2
import numpy as np

from .camera_models import PinholeCamera


class AppearancePublic:
    """Open only a family's public sensor files, never ground-truth states."""

    def __init__(self, family):
        self.family = Path(family)
        public = self.family/'public'
        self.manifest = json.loads((public/'frames.json').read_text())
        self.intrinsics = json.loads((public/'intrinsics.json').read_text())
        self.camera = PinholeCamera(**self.intrinsics)
        self.conditions = tuple(self.manifest['conditions'])

    def _frame(self, frame):
        if (not isinstance(frame, (int, np.integer)) or isinstance(frame, (bool, np.bool_))
                or not 0 <= frame < len(self.manifest['frames'])):
            raise IndexError('Appearance frame outside sequence')
        return self.manifest['frames'][int(frame)]

    def image_path(self, frame, condition='diffuse_symmetric'):
        if condition not in self.conditions:
            raise ValueError('Unknown appearance condition')
        return self.family/'public'/condition/'rgb'/self._frame(frame)['image']

    def rgb(self, frame, condition='diffuse_symmetric'):
        image = cv2.imread(str(self.image_path(frame, condition)))
        if image is None:
            raise FileNotFoundError(self.image_path(frame, condition))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class AppearanceEvaluation(AppearancePublic):
    """Privileged geometry, original material identity and renderer-specific truth.

    Returned positions are meters in the declared render world. Rays originate
    at pixel centers. Flow projection does not imply visibility in another frame.
    """

    def __init__(self, family):
        super().__init__(family)
        with np.load(self.family/'evaluation/states.npz', allow_pickle=False) as a:
            self.states = {k: a[k] for k in a.files}
        with np.load(self.family/'evaluation/geometry.npz', allow_pickle=False) as a:
            self.geometry = {k: a[k] for k in a.files}

    def state(self, frame):
        self._frame(frame)
        return {key: value[frame] for key, value in self.states.items()}

    def dense(self, frame):
        name = self._frame(frame)['image'].replace('.png', '.npz')
        with np.load(self.family/'evaluation/dense'/name, allow_pickle=False) as a:
            result = {k: a[k] for k in a.files}
        return self.expand_dense(result, frame)

    def expand_dense(self, dense, frame):
        state = self.state(frame)
        result = dict(dense)
        good = result['valid']
        ids = result['triangle_id'][good]
        vertices = state['vertices_meter'].astype(float)
        faces = self.geometry['triangles']
        xyz = np.full(good.shape+(3,), np.nan)
        xyz[good] = np.sum(result['material_barycentric'][good, :, None]*vertices[faces[ids]], axis=1)
        all_normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                               vertices[faces[:, 2]]-vertices[faces[:, 0]])
        all_normals /= np.linalg.norm(all_normals, axis=1)[:, None]
        normals = np.full_like(xyz, np.nan)
        normals[good] = all_normals[ids]
        result.update(position_world=xyz, normal_world=normals, shading_normal_world=normals.copy(),
                      sensor_valid=np.ones(good.shape, dtype=bool))
        return result

    def material_points(self, frame, triangle_ids, barycentric):
        s = self.state(frame)
        ids, b = np.asarray(triangle_ids), np.asarray(barycentric, dtype=float)
        faces = self.geometry['triangles']
        if (not np.issubdtype(ids.dtype, np.integer) or b.shape != ids.shape+(3,)
                or np.any(ids < 0) or np.any(ids >= len(faces)) or not np.isfinite(b).all()
                or np.any(b < -2e-7) or np.any(b > 1+2e-7)
                or np.any(np.abs(b.sum(axis=-1)-1) > 2e-7)):
            raise ValueError('Invalid material triangle/barycentric query')
        # Preserve the tracer weights, including their explicitly bounded sum
        # rounding, rather than silently renormalizing the recorded truth.
        return np.sum(b[..., None]*s['vertices_meter'].astype(float)[faces[ids]], axis=-2)

    def project_material(self, frame, triangle_ids, barycentric):
        p = self.material_points(frame, triangle_ids, barycentric)
        s = self.state(frame)
        uv, forward = self.camera.project((p-s['C_world_meter'])@s['R_wc'])
        return {'uv': uv, 'forward': forward, 'inside_image': forward & self.camera.contains(uv),
                'visibility': 'Not evaluated: in-frame projection is not an occlusion test.'}

    def radiance(self, frame, condition='diffuse_symmetric'):
        if condition not in self.conditions:
            raise ValueError('Unknown appearance condition')
        source = self.manifest['conditions'][condition]['radiance_condition']
        name = self._frame(frame)['image'].replace('.png', '.npz')
        with np.load(self.family/'evaluation/radiance'/source/name, allow_pickle=False) as a:
            return a['linear_radiance_rgb']
