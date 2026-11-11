"""Opt-in arc-length tube geometry and configurable acquisition/motion.

Legacy benchmark functions remain the default when no collection options occur.
"""
from dataclasses import replace
import numpy as np
from scipy.interpolate import PchipInterpolator
from .tube import build_tube_static_geometry, _interp_curve
from .camera import _rotation_from_lookat


def geometry(cfg, preset, rng, options):
    g = build_tube_static_geometry(cfg, preset, rng)
    # Parallel transport avoids the world-up singularity on fully 3D axes.
    t = g.tangent.astype(float)
    r = np.empty_like(t); u = np.empty_like(t)
    r[0] = g.frame_right[0]
    for i in range(1, len(t)):
        v = np.cross(t[i-1], t[i]); c = np.dot(t[i-1], t[i])
        if c < -0.99999:
            raise ValueError('Axis sampling cannot resolve a reversing tangent')
        K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        rr = (np.eye(3)+K+K@K/(1+c))@r[i-1]
        rr -= t[i]*np.dot(rr,t[i]); r[i] = rr/np.linalg.norm(rr)
    u[:] = np.cross(t,r)
    knots = np.asarray(options['radius_profile'],float)
    if knots.shape[1] != 2 or knots[0,0] != 0 or knots[-1,0] != 1 or np.any(np.diff(knots[:,0])<=0) or np.any(knots[:,1]<=0):
        raise ValueError('radius_profile requires increasing [0,1] positions and positive factors')
    radius = g.base_radius.astype(float) + cfg.base_radius*(PchipInterpolator(knots[:,0],knots[:,1])(g.s_norm)-1)[:,None]
    if radius.min() <= max(cfg.min_radius,cfg.shell_thickness):
        raise ValueError('Configured radius violates lumen/thickness margin')
    return replace(g,frame_right=r.astype(np.float32),frame_up=u.astype(np.float32),base_radius=radius.astype(np.float32))


def coefficients(family):
    n = family['action_frames']; waves = family['temporal_coefficients']
    if len(waves) != len(family['patches']):
        raise ValueError('One temporal coefficient specification required per basis')
    t=np.linspace(0,1,n); out=[]
    for w in waves:
        if w['kind']!='raised_cosine' or not 0<=w['start']<w['end']<=1:
            raise ValueError('Invalid raised-cosine support')
        x=np.clip((t-w['start'])/(w['end']-w['start']),0,1)
        out.append(w.get('amplitude',1.)*np.sin(np.pi*x)**2)
    return np.stack(out,axis=1)


def poses(g,family,cfg):
    nref,n=family['reference_frames'],family['action_frames']; opts=family['acquisition']
    def path(progress):
        rows=[]
        for i,s in enumerate(progress):
            phase=2*np.pi*opts['orbit_cycles']*i/max(len(progress)-1,1)+.27
            radius=float(np.min(_interp_curve(g.base_radius,s)))
            center=_interp_curve(g.centerline,s)
            up=_interp_curve(g.frame_up,s);right=_interp_curve(g.frame_right,s)
            eye=center+opts['offset_ratio']*radius*(np.cos(phase)*right+np.sin(phase)*up)
            target=_interp_curve(g.centerline,min(1,s+opts['lookahead']/cfg.tube.length))
            rows.append((_rotation_from_lookat(eye,target,up),eye.astype(np.float32)))
        return np.stack([p[0] for p in rows]),np.stack([p[1] for p in rows])
    a,b,c=family['reference_progress'];turn=round(nref*2/3)
    ref=np.r_[np.linspace(a,b,turn),np.linspace(b,c,nref-turn+1)[1:]]
    rr,rc=path(ref);mr,mc=path(np.linspace(*family['moving_progress'],n))
    return dict(reference_R=rr,reference_C=rc,moving_R=mr,moving_C=mc,
                stationary_R=np.repeat(rr[-1:],n,axis=0),stationary_C=np.repeat(rc[-1:],n,axis=0))
