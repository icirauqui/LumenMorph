"""Continuous assigned Fourier reflectance in frozen material reference coordinates.

Geometry vertices carry reference XYZ, never colors sampled only at the vertices.
Interpolate XYZ at a hit, then evaluate this field. Frequencies are reference-space
radians/meter; this removes geometric-mesh color interpolation aliasing but does
not remove image/pixel sampling aliasing or validate biological microstructure.
The optional Mitsuba adapter is forward-only, with no inverse-rendering contract.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class FourierMaterialField:
    origin_meter: tuple = (0., 0., 0.)
    base_albedo_rgb: tuple = (.48, .16, .13)
    wavevectors_radian_per_meter: tuple = ()
    phases_radian: tuple = ()
    amplitudes: tuple = ()

    def __post_init__(self):
        origin=np.asarray(self.origin_meter,dtype=float)
        color=np.asarray(self.base_albedo_rgb,dtype=float)
        wave=np.asarray(self.wavevectors_radian_per_meter,dtype=float)
        if wave.shape==(0,):wave=np.empty((0,3))
        phase=np.asarray(self.phases_radian,dtype=float)
        amplitude=np.asarray(self.amplitudes,dtype=float)
        if (origin.shape!=(3,) or color.shape!=(3,) or wave.ndim!=2 or wave.shape[1]!=3
            or phase.shape!=(len(wave),) or amplitude.shape!=phase.shape
            or not all(np.isfinite(x).all() for x in (origin,color,wave,phase,amplitude))
            or (color<0).any() or (color>1).any() or np.abs(amplitude).sum()>=1
            or color.max()*(1+np.abs(amplitude).sum())>1):
            raise ValueError('Finite bounded Fourier reflectance specification required')
        # Materialize immutable native Python values: external ndarray/list edits
        # must not silently alter an already declared spectrum or its identity.
        for name,array in [('origin_meter',origin),('base_albedo_rgb',color),
                           ('phases_radian',phase),('amplitudes',amplitude)]:
            object.__setattr__(self,name,tuple(map(float,array)))
        object.__setattr__(self,'wavevectors_radian_per_meter',tuple(tuple(map(float,row)) for row in wave))

    def to_dict(self):
        return {'schema':'continuous_reference_fourier_v1',
                'origin_meter':list(self.origin_meter),'base_albedo_rgb':list(self.base_albedo_rgb),
                'wavevectors_radian_per_meter':[list(row) for row in self.wavevectors_radian_per_meter],
                'phases_radian':list(self.phases_radian),'amplitudes':list(self.amplitudes)}

    @classmethod
    def from_dict(cls,data):
        expected={'schema','origin_meter','base_albedo_rgb','wavevectors_radian_per_meter','phases_radian','amplitudes'}
        if set(data)!=expected or data['schema']!='continuous_reference_fourier_v1':
            raise ValueError('Unsupported or incomplete Fourier field specification')
        return cls(**{k:v for k,v in data.items() if k!='schema'})

    def sha256(self):
        payload=json.dumps(self.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def evaluate_offsets(self,reference_offsets_meter):
        """Independent NumPy float64 evaluation at coordinates minus fixed origin."""
        xyz=np.asarray(reference_offsets_meter,dtype=float)
        if xyz.ndim<1 or xyz.shape[-1]!=3 or not np.isfinite(xyz).all():
            raise ValueError('Finite material reference coordinates ending in XYZ required')
        modulation=np.zeros(xyz.shape[:-1],dtype=float)
        for k,phase,amplitude in zip(self.wavevectors_radian_per_meter,self.phases_radian,self.amplitudes):
            with np.errstate(over='ignore',invalid='ignore'):
                angle=np.sum(xyz*np.asarray(k),axis=-1)+phase
            if not np.isfinite(angle).all():raise ValueError('Fourier material phase overflow')
            modulation+=amplitude*np.cos(angle)
        return (1+modulation[...,None])*np.asarray(self.base_albedo_rgb)

    def evaluate(self,reference_xyz_meter):
        """Evaluate using the frozen physical reference frame, not deformed XYZ."""
        xyz=np.asarray(reference_xyz_meter,dtype=float)
        if xyz.ndim<1 or xyz.shape[-1]!=3:raise ValueError('Material reference coordinates must end in XYZ')
        with np.errstate(over='ignore',invalid='ignore'):
            offsets=xyz-np.asarray(self.origin_meter)
        return self.evaluate_offsets(offsets)

    def renderer_reference_offsets(self,reference_xyz_meter):
        """Center before float32 conversion to preserve local physical precision."""
        xyz=np.asarray(reference_xyz_meter,dtype=float)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
            raise ValueError('Finite Nx3 material reference vertices required')
        with np.errstate(over='ignore',invalid='ignore'):
            offset=(xyz-np.asarray(self.origin_meter)).astype(np.float32)
            wave=np.asarray(self.wavevectors_radian_per_meter,dtype=np.float32).reshape(-1,3)
            phases=np.asarray(self.phases_radian,dtype=np.float32)
        if not np.isfinite(offset).all() or not np.isfinite(wave).all() or not np.isfinite(phases).all():
            raise ValueError('Material coordinates/spectrum exceed renderer float32 range')
        # Reject arithmetic overflow before handing unchecked values to the JIT.
        if len(offset) and len(wave):
            bound=np.max(np.abs(offset).astype(float),axis=0)@np.abs(wave.astype(float)).T+np.abs(phases.astype(float))
            if np.any(bound>np.finfo(np.float32).max):raise ValueError('Fourier phase exceeds renderer float32 range')
        return offset

    def sampling_mean_proxy(self):
        """Phase-ensemble RGB mean, not a measured spatial surface average.

        Mitsuba roughplastic uses texture mean only for its sampling mixture;
        eval/pdf still use actual local reflectance and matching mixture weights.
        This fixed proposal statistic avoids mesh-quadrature aliasing. It is exact
        for the zero-mode constant control, and independent of tessellation/state.
        """
        return float(np.mean(self.base_albedo_rgb))


def seeded_fourier_field(origin_meter, *, seed=20260911, base_albedo_rgb=(.48,.16,.13),
                         wavelengths_meter=(.001,.003,.012), relative_amplitudes=(.03,.08,.15),
                         modes_per_scale=6):
    """Build and return explicit spectrum once; save it rather than only its seed.

    Directions/phases follow the original vertex-field generator's RNG order.
    Origin is caller-supplied and must stay fixed even when a mesh is subdivided.
    """
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32:
        raise ValueError('Seed must be uint32')
    if not isinstance(modes_per_scale,int) or isinstance(modes_per_scale,bool) or modes_per_scale<1:
        raise ValueError('Positive integer modes_per_scale required')
    wave=np.asarray(wavelengths_meter,dtype=float);amplitude=np.asarray(relative_amplitudes,dtype=float)
    if (wave.ndim!=1 or not len(wave) or amplitude.shape!=wave.shape or not np.isfinite(wave).all()
        or not np.isfinite(amplitude).all() or (wave<=0).any() or (amplitude<0).any()
        or amplitude.sum()>=1):
        raise ValueError('Positive wavelengths and matching nonnegative amplitudes required')
    rng=np.random.default_rng(seed);vectors=[];phases=[];weights=[]
    for wavelength,a in zip(wave,amplitude):
        direction=rng.normal(size=(modes_per_scale,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
        phase=rng.uniform(0,2*np.pi,modes_per_scale)
        with np.errstate(over='ignore',invalid='ignore'):
            vectors.extend((direction*(2*np.pi/wavelength)).tolist())
        phases.extend(phase.tolist());weights.extend([float(a/modes_per_scale)]*modes_per_scale)
    return FourierMaterialField(origin_meter,base_albedo_rgb,tuple(map(tuple,vectors)),tuple(phases),tuple(weights))


def mitsuba_fourier_texture(mi,dr,field):
    """Evaluate frozen spectrum on the barycentric reference-coordinate attribute."""
    class ReferenceFourierTexture(mi.Texture):
        def __init__(self):
            super().__init__(mi.Properties())
            self.coordinates=mi.load_dict({'type':'mesh_attribute','name':'vertex_reference_offset'})

        def eval(self,si,active=True):
            q=self.coordinates.eval_3(si,active)
            modulation=mi.Float(0.)
            for k,phase,amplitude in zip(field.wavevectors_radian_per_meter,field.phases_radian,field.amplitudes):
                angle=q[0]*k[0]+q[1]*k[1]+q[2]*k[2]+phase
                modulation+=amplitude*dr.cos(angle)
            return mi.Color3f(*field.base_albedo_rgb)*(1+modulation)

        def eval_3(self,si,active=True):
            return self.eval(si,active)

        def eval_1(self,si,active=True):
            return mi.luminance(self.eval(si,active))

        def mean(self):
            return field.sampling_mean_proxy()

        def is_spatially_varying(self):
            return any(a!=0 and any(x!=0 for x in k) for a,k in zip(field.amplitudes,field.wavevectors_radian_per_meter))

        def to_string(self):
            return 'ContinuousReferenceFourierTexture'

    return ReferenceFourierTexture()
