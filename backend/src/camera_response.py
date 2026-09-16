"""Explicit pointwise camera response and image-only exposure control.

No blur, resampling, denoising or learned translation is performed. The pixel
center geometry is unchanged. Aperture support must accompany downstream flow.
Parameters describe an assigned processing model, not identified tissue optics.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .endoscopic_renderer import linear_to_srgb


@dataclass(frozen=True)
class CameraResponseSettings:
    encoding: str = 'srgb'
    gamma: float = 2.2
    white_balance_rgb: tuple = (1.,1.,1.)
    exposure_mode: str = 'fixed'
    exposure_gain: float = 1.
    target_linear_luminance: float = .07
    metering_percentile: float = 50.
    metering_radius_fraction: float = .8
    gain_limits: tuple = (.125,64.)
    exposure_time_constant_seconds: float = .15
    radial_falloff_strength: float = 0.
    aperture_polygon_normalized: tuple | None = None

    def __post_init__(self):
        if self.encoding not in ('srgb','power_gamma') or self.exposure_mode not in ('fixed','automatic'):
            raise ValueError('Unsupported image response mode')
        scalars=[self.gamma,self.exposure_gain,self.target_linear_luminance,self.metering_percentile,
                 self.metering_radius_fraction,self.exposure_time_constant_seconds,self.radial_falloff_strength]
        wb=np.asarray(self.white_balance_rgb);limits=np.asarray(self.gain_limits)
        if (not np.isfinite(scalars).all() or min(self.gamma,self.exposure_gain,self.target_linear_luminance,self.metering_radius_fraction)<=0
            or self.exposure_time_constant_seconds<0 or self.radial_falloff_strength<0
            or self.target_linear_luminance>1
            or not 0<=self.metering_percentile<=100 or self.metering_radius_fraction>1
            or wb.shape!=(3,) or not np.isfinite(wb).all() or (wb<=0).any()
            or limits.shape!=(2,) or not np.isfinite(limits).all() or not 0<limits[0]<=limits[1]):
            raise ValueError('Invalid finite camera-response settings')
        if self.aperture_polygon_normalized is not None:
            p=np.asarray(self.aperture_polygon_normalized,dtype=float)
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
                raise ValueError('Aperture requires finite normalized polygon vertices')
            e=np.roll(p,-1,axis=0)-p;cross=e[:,0]*np.roll(e,-1,axis=0)[:,1]-e[:,1]*np.roll(e,-1,axis=0)[:,0]
            if not ((cross>0).all() or (cross<0).all()):raise ValueError('Aperture polygon must be strictly convex and ordered')
            # Equal local turn signs also occur in self-intersecting stars.
            # A convex ordered polygon puts every other vertex strictly on the
            # same side of every oriented edge (both windings are supported).
            delta=p[None,:,:]-p[:,None,:]
            side=e[:,None,0]*delta[:,:,1]-e[:,None,1]*delta[:,:,0]
            other=np.ones((len(p),len(p)),dtype=bool)
            ids=np.arange(len(p));other[ids,ids]=False;other[ids,(ids+1)%len(p)]=False
            if not (np.sign(cross[0])*side[other]>0).all():
                raise ValueError('Aperture polygon must be strictly convex and ordered')


def aperture_mask(width,height,polygon):
    """Discrete pixel-center support for a specified normalized convex aperture."""
    y,x=np.mgrid[:height,:width]
    if polygon is None:return np.ones((height,width),dtype=bool)
    points=np.stack(((x+.5)/width,(y+.5)/height),axis=-1)
    p=np.asarray(polygon,dtype=float);edges=np.roll(p,-1,axis=0)-p
    area=np.sum(p[:,0]*np.roll(p,-1,axis=0)[:,1]-p[:,1]*np.roll(p,-1,axis=0)[:,0])
    inside=np.ones((height,width),dtype=bool)
    for origin,edge in zip(p,edges):
        delta=points-origin;cross=edge[0]*delta[...,1]-edge[1]*delta[...,0]
        inside &= np.sign(area)*cross>=-1e-14
    return inside


def apply_camera_response(linear_radiance_rgb, sensor_valid, settings=CameraResponseSettings(),
                          *, previous_log_gain=None, elapsed_seconds=0., quantize=True):
    """Encode radiance and return explicit sensor support/exposure state.

    Automatic gain is based solely on current image radiance after fixed color
    gains/falloff and previous exposure state. First frame initializes directly;
    subsequent frames update exponentially in log-gain using actual elapsed time.
    Geometry/depth/material arrays are never inputs to exposure estimation.
    quantize=False returns continuous encoded RGB in [0,1] for response fitting;
    the default returns the final quantized uint8 RGB image.
    """
    image=np.asarray(linear_radiance_rgb,dtype=float);support=np.asarray(sensor_valid)
    if not isinstance(quantize,bool):raise ValueError('quantize must be boolean')
    if (image.ndim!=3 or image.shape[-1]!=3 or min(image.shape[:2])<1 or not np.isfinite(image).all()
        or (image<0).any() or support.shape!=image.shape[:2] or support.dtype!=bool):
        raise ValueError('Finite nonnegative HxWx3 radiance and matching boolean sensor support required')
    if not np.isfinite(elapsed_seconds) or elapsed_seconds<0 or (previous_log_gain is not None and not np.isfinite(previous_log_gain)):
        raise ValueError('Invalid exposure history/time')
    h,w=support.shape;y,x=np.mgrid[:h,:w]
    radius=np.hypot((x+.5-w/2)/(min(w,h)/2),(y+.5-h/2)/(min(w,h)/2))
    observed=support & aperture_mask(w,h,settings.aperture_polygon_normalized)
    processed=image*np.asarray(settings.white_balance_rgb)[None,None,:]*np.exp(-settings.radial_falloff_strength*radius**2)[...,None]
    if not np.isfinite(processed).all():raise ValueError('Camera color response overflow')
    metering=observed & (radius<=settings.metering_radius_fraction)
    y_linear=processed@np.array([.2126,.7152,.0722])
    statistic=float(np.percentile(y_linear[metering],settings.metering_percentile)) if metering.any() else None
    if settings.exposure_mode=='fixed':
        log_gain=float(np.log(settings.exposure_gain)); desired=settings.exposure_gain
    else:
        desired=float(np.clip(settings.target_linear_luminance/max(statistic or 0.,np.finfo(float).tiny),*settings.gain_limits))
        target=float(np.log(desired))
        alpha=1. if previous_log_gain is None or settings.exposure_time_constant_seconds==0 else -np.expm1(-elapsed_seconds/settings.exposure_time_constant_seconds)
        log_gain=target if previous_log_gain is None else float(previous_log_gain+alpha*(target-previous_log_gain))
        log_gain=float(np.clip(log_gain,np.log(settings.gain_limits[0]),np.log(settings.gain_limits[1])))
    gain=float(np.exp(log_gain));linear=processed*gain
    if not np.isfinite(linear).all():raise ValueError('Camera exposure overflow')
    encoded=linear_to_srgb(linear) if settings.encoding=='srgb' else np.clip(linear,0,1)**(1/settings.gamma)
    rgb=np.rint(255*encoded).astype(np.uint8) if quantize else encoded
    rgb[~observed]=0
    return rgb,observed,{'schema':'pointwise_camera_response_pilot_v1','settings':asdict(settings),
        'exposure_gain':gain,'next_log_gain':log_gain,'desired_gain':desired,'metering_luminance':statistic,
        'metering_pixels':int(metering.sum()),'elapsed_seconds':float(elapsed_seconds),
        'output_encoding':'uint8 rounded' if quantize else 'continuous encoded float [0,1]',
        'aperture_semantics':'pixel-center mask; downstream endpoints use containing target pixel support',
        'limitations':'assigned image-processing model; gain, illuminant, reflectance and sensor response are not separately identified'}
