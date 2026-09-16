"""Explicit pinhole and OpenCV-fisheye projection for the extension pilot.

Camera axes: x right, y up, z forward. Pixel coordinates: u right, v down;
sample centers are supplied by callers, normally (column+.5, row+.5).
Projection validity is distinct from being inside the image. Fisheye support
is restricted to an explicitly monotone forward-facing angular domain.
No existing frozen-bank producer uses these opt-in models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


def _array(value, last):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != last:
        raise ValueError(f'Expected final dimension {last}')
    return a


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0
               for v in (self.width, self.height)):
            raise ValueError('Image dimensions must be positive integers')
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError('Focal lengths must be positive; intrinsics finite')

    def to_dict(self):
        return {'model': 'PINHOLE', **asdict(self),
                'camera_axes': 'x-right,y-up,z-forward', 'pixel_centers': 'column+.5,row+.5'}

    def contains(self, uv):
        uv = _array(uv, 2)
        return (np.isfinite(uv).all(axis=-1) & (uv[..., 0] >= 0) & (uv[..., 0] < self.width)
                & (uv[..., 1] >= 0) & (uv[..., 1] < self.height))

    def project(self, points_camera):
        p = _array(points_camera, 3)
        valid = np.isfinite(p).all(axis=-1) & (p[..., 2] > 0)
        uv = np.full(p.shape[:-1] + (2,), np.nan)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            uv[..., 0] = self.fx * p[..., 0] / p[..., 2] + self.cx
            uv[..., 1] = self.cy - self.fy * p[..., 1] / p[..., 2]
        valid &= np.isfinite(uv).all(axis=-1)
        uv[~valid] = np.nan
        return uv, valid

    def rays(self, uv):
        uv = _array(uv, 2)
        with np.errstate(over='ignore', invalid='ignore'):
            xy = (uv - [self.cx, self.cy]) / [self.fx, -self.fy]
            d = np.concatenate((xy, np.ones(uv.shape[:-1] + (1,))), axis=-1)
            d /= np.hypot(np.hypot(d[..., 0], d[..., 1]), d[..., 2])[..., None]
        valid = np.isfinite(d).all(axis=-1) & (d[..., 2] > 0)
        d[~valid] = np.nan
        return d, valid


@dataclass(frozen=True)
class OpenCVFisheyeCamera(PinholeCamera):
    distortion: tuple[float, float, float, float] = (0., 0., 0., 0.)
    max_theta: float = np.pi / 2 - 1e-6

    def __post_init__(self):
        super().__post_init__()
        k = np.asarray(self.distortion, dtype=float)
        if k.shape != (4,) or not np.isfinite(k).all():
            raise ValueError('OpenCV fisheye requires four finite coefficients')
        if not np.isfinite(self.max_theta) or not 0 < self.max_theta < np.pi / 2:
            raise ValueError('max_theta must lie in the forward-facing interval (0,pi/2)')
        object.__setattr__(self, 'distortion', tuple(float(v) for v in k))
        # q(s)=d(theta_distorted)/d(theta), s=theta**2. Check every
        # interior extremum, not just a coarse sample of potentially folded optics.
        q = np.r_[1., k * [3., 5., 7., 9.]]
        roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(q))
        real = roots[np.abs(roots.imag) < 1e-10].real
        candidates = np.r_[0., self.max_theta**2,
                           real[(real > 0) & (real < self.max_theta**2)]]
        if np.min(np.polynomial.polynomial.polyval(candidates, q)) <= 1e-8:
            raise ValueError('Fisheye radial map must be strictly monotone on declared angular domain')

    def to_dict(self):
        result = super().to_dict()
        result['model'] = 'OPENCV_FISHEYE'
        result['distortion'] = list(self.distortion)
        result['angular_domain_radians'] = [0., self.max_theta]
        return result

    def _distort_theta(self, theta):
        return theta * np.polynomial.polynomial.polyval(theta * theta, [1., *self.distortion])

    def project(self, points_camera):
        p = _array(points_camera, 3)
        valid = np.isfinite(p).all(axis=-1) & (p[..., 2] > 0)
        radial = np.hypot(p[..., 0], p[..., 1])
        theta = np.arctan2(radial, p[..., 2])
        valid &= theta <= self.max_theta
        scale = np.divide(self._distort_theta(theta), radial,
                          out=np.zeros_like(theta), where=radial > 0)
        uv = np.stack((self.fx * p[..., 0] * scale + self.cx,
                       self.cy - self.fy * p[..., 1] * scale), axis=-1)
        valid &= np.isfinite(uv).all(axis=-1)
        uv[~valid] = np.nan
        return uv, valid

    def rays(self, uv):
        uv = _array(uv, 2)
        xy = (uv - [self.cx, self.cy]) / [self.fx, -self.fy]
        rd = np.hypot(xy[..., 0], xy[..., 1])
        valid = np.isfinite(rd) & (rd <= self._distort_theta(self.max_theta))
        # Bisection stays in the proven monotone domain. Invalid pixels are
        # masked rather than clipped to a fabricated edge ray.
        lo = np.zeros_like(rd)
        hi = np.full_like(rd, self.max_theta)
        target = np.where(valid, rd, 0.)
        for _ in range(56):
            mid = (lo + hi) * .5
            lower = self._distort_theta(mid) < target
            lo = np.where(lower, mid, lo)
            hi = np.where(lower, hi, mid)
        theta = (lo + hi) * .5
        scale = np.divide(np.sin(theta), rd, out=np.zeros_like(rd), where=rd > 0)
        d = np.concatenate((xy * scale[..., None], np.cos(theta)[..., None]), axis=-1)
        d[~valid] = np.nan
        return d, valid


def pixel_center_rays(camera: PinholeCamera):
    rows, columns = np.mgrid[:camera.height, :camera.width]
    return camera.rays(np.stack((columns + .5, rows + .5), axis=-1))


def opencv_pose_with_reflected_world(R_wc, C_world):
    """Return an interoperable proper OpenCV pose AND its mandatory world map.

    Existing pilot axes use y-up; merely flipping the camera y row gives an
    improper rotation. Define X_cv_world=H@X_pilot_world, H=diag(1,-1,1), and
    x_cv_camera=R_cw@X_cv_world+t_cw. Applying only the returned pose to original
    world points is invalid. Vector/tensor/mesh orientation conversions must
    accompany any export; this helper does not convert a whole dataset.
    """
    R=np.asarray(R_wc,dtype=float);C=np.asarray(C_world,dtype=float)
    if (R.shape!=(3,3) or C.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(C).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0)
        or not np.isclose(np.linalg.det(R),1.,atol=1e-10,rtol=0)):
        raise ValueError('Expected proper finite pilot camera pose')
    H=np.diag([1.,-1.,1.])
    return {'world_transform':H,'R_cw':H@R.T@H,'t_cw':-H@R.T@C,'C_world':H@C}
