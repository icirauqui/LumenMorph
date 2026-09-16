"""Read-only baseline-state adapter for matched endoscopic appearance studies.

This module does not invoke the mechanical solver or alter the sealed producer.
Its states are evaluator/producer information, never image-derived inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .camera_models import PinholeCamera
from .endoscopic_renderer import AppearanceSettings, render_endoscopic, linear_to_srgb


GEOMETRY_LABEL_KEYS = (
    'depth_z', 'triangle_id', 'material_barycentric', 'normal_world',
    'valid', 'sensor_valid', 'position_world', 'shading_normal_world',
    'mesh_topology_sha256',
)


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def srgb_to_linear(rgb):
    """Decode finite normalized sRGB; this is assigned reflectance, not a fit."""
    a = np.asarray(rgb, dtype=float)
    if a.ndim < 1 or a.shape[-1] != 3 or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError('Expected finite normalized RGB in [0,1]')
    return np.where(a <= .04045, a/12.92, ((a+.055)/1.055)**2.4)


def exposed_rgb(radiance, exposure):
    a = np.asarray(radiance, dtype=float)
    if (a.ndim < 1 or a.shape[-1] != 3 or not np.isfinite(a).all() or np.any(a < 0)
            or not np.isscalar(exposure) or not np.isfinite(exposure) or exposure <= 0):
        raise ValueError('Require nonnegative finite RGB radiance and positive finite exposure')
    with np.errstate(over='ignore'):
        return np.rint(255*linear_to_srgb(a*exposure)).astype(np.uint8)


def proper_rotation(source):
    """Remove only recorded float32 pose rounding, rejecting larger defects."""
    R = np.asarray(source, dtype=float)
    if (R.shape != (3, 3) or not np.isfinite(R).all() or np.linalg.det(R) <= 0
            or np.max(np.abs(R.T@R-np.eye(3))) > 1e-5):
        raise ValueError('Source pose is not a rounded proper rotation')
    u, _, vt = np.linalg.svd(R)
    return u@vt


def assert_shared_geometry(first, second):
    for key in GEOMETRY_LABEL_KEYS:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        # String topology digest has no NaN representation.
        equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)
        if not equal:
            raise ValueError(f'Matched rendering changed geometry label {key}')


class BaselineAppearanceSequence:
    """Load an existing benchmark package once and explicitly convert its states.

    A uniform *assigned* rendering scale maps the family's median radius to
    ``median_radius_meter``. This does not rescale or requalify original FE
    stresses, loads or constitutive parameters. Source paths in ``provenance``
    are portable relative to the whole baseline suite.
    """

    def __init__(self, package, *, stream='moving', resolution_divisor=1,
                 median_radius_meter=.015):
        self.package = Path(package).resolve()
        if stream not in ('moving', 'stationary', 'reference'):
            raise ValueError('Unknown baseline stream')
        if (not isinstance(resolution_divisor, int) or isinstance(resolution_divisor, bool)
                or resolution_divisor < 1):
            raise ValueError('Resolution divisor must be a positive integer')
        if not np.isfinite(median_radius_meter) or median_radius_meter <= 0:
            raise ValueError('Assigned radius must be finite and positive')
        self.stream = stream
        self.descriptor = json.loads((self.package/'DATASET.json').read_text())
        if self.descriptor['schema'] != 'meshmeld_tubular_benchmark_v1':
            raise ValueError('Unsupported baseline schema')
        family = (self.package/'evaluation/family').resolve()
        state_dir = family.parent/'reference_truth' if stream == 'reference' else self.package/'evaluation'/stream
        paths = {'geometry': family/'geometry.npz', 'states': state_dir/'states.npz',
                 'texture': family/'texture.png', 'intrinsics': self.package/'public/intrinsics.json',
                 'descriptor': self.package/'DATASET.json'}
        with np.load(paths['geometry'], allow_pickle=False) as a:
            self.geometry = {k: a[k] for k in a.files}
        with np.load(paths['states'], allow_pickle=False) as a:
            self.states = {k: a[k] for k in a.files}
        radius = float(np.median(self.geometry['base_radius']))
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('Invalid source radius')
        self.scale = float(median_radius_meter)/radius
        intr = json.loads(paths['intrinsics'].read_text())
        if intr['width'] % resolution_divisor or intr['height'] % resolution_divisor:
            raise ValueError('Resolution divisor must exactly divide image dimensions')
        self.camera = PinholeCamera(
            int(intr['width']//resolution_divisor), int(intr['height']//resolution_divisor),
            *(float(intr[k])/resolution_divisor for k in ('fx', 'fy', 'cx', 'cy')))
        texture = cv2.imread(str(paths['texture']))
        if texture is None:
            raise ValueError('Unable to read source texture')
        self.albedo = srgb_to_linear(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)/255).astype(np.float32)
        self.reference_vertices_meter = (self.geometry['vertices'].astype(float)*self.scale).astype(np.float32)
        for a in [*self.geometry.values(), *self.states.values(), self.albedo, self.reference_vertices_meter]:
            a.setflags(write=False)
        suite = self.package.parent
        self.provenance = {
            'schema': 'baseline_appearance_adapter_v1', 'family': self.descriptor['family'],
            'baseline_condition': self.descriptor['condition'], 'baseline_split': self.descriptor['split'],
            'stream': stream, 'source_to_meter_scale': self.scale,
            'assigned_median_radius_meter': float(median_radius_meter),
            'source_files': {k: {'path': str(p.resolve().relative_to(suite)), 'sha256': sha256(p)} for k, p in paths.items()},
            'camera': self.camera.to_dict(),
            'mechanics_scope': 'Original FE annotations retain original units and qualification; assigned rendering scale is not tissue calibration.',
            'texture_scope': 'Inverse sRGB baseline texture assigned as linear albedo; bilinear periodic UV sampling in the new renderer.',
            'pose_conversion': 'Nearest proper orthogonal rotation by SVD; exact converted poses stored in shared evaluator truth.',
            'baseline_pixel_labels': 'Not reused: new ray-traced labels generated for converted geometry/poses.',
        }

    def __len__(self):
        return len(self.states['vertices'])

    def state(self, frame):
        if not isinstance(frame, (int, np.integer)) or isinstance(frame, (bool, np.bool_)) or not 0 <= frame < len(self):
            raise IndexError('Frame index outside baseline stream')
        exact = self.states['vertices'][frame].astype(float)*self.scale
        v = exact.astype(np.float32)
        original_R = self.states['R_wc'][frame].astype(float)
        R = proper_rotation(original_R)
        return {'vertices_meter': v, 'R_wc': R,
                'C_world_meter': self.states['C_world'][frame].astype(float)*self.scale,
                'time_seconds': float(self.states['time_seconds'][frame]),
                'vertex_quantization_max_meter': float(np.linalg.norm(v-exact, axis=1).max()),
                'rotation_max_change': float(np.abs(R-original_R).max())}

    def render(self, frame, settings: AppearanceSettings):
        s = self.state(frame)
        rgb, truth, meta = render_endoscopic(s['vertices_meter'], self.geometry['triangles'],
            self.geometry['uv'], self.albedo.copy(), self.camera, s['R_wc'], s['C_world_meter'], settings)
        meta['baseline_adapter'] = {**self.provenance, 'source_frame': int(frame),
            **{k: s[k] for k in ('time_seconds', 'vertex_quantization_max_meter', 'rotation_max_change')}}
        return rgb, truth, meta
