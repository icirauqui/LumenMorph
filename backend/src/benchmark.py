"""Versioned, complete tubular benchmark producer (no inverse-model scoring).

Run from the generator checkout: python -m backend.src.benchmark --help.
All writes require fresh stage directories; completed frames are never replaced.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import cv2
import numpy as np

from .benchmark_mechanics import solve_benchmark_fields, prescribed_fields
from .camera import _rotation_from_lookat, _quat_from_rot
from .config import load_config_dict
from .dense_truth import render_rgb_and_truth, flow_to_frame
from .noise import apply_photometric_noise
from .texture import generate_texture_with_qa, orb_feature_count
from .tube import build_tube_static_geometry, tube_vertices_from_radius, _interp_curve
from .tube_fea import _continuum_fem_src


GENERATOR_LOGICAL_ROOT = 'third_party/LumenMorph'
LEGACY_GENERATOR_LOGICAL_ROOT = 'third_party/def-slam-dataset-generator'


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def seal(path):
    files={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name!='COMPLETE.json'}
    write_json(path/'COMPLETE.json',{'format':'tubular_stage_v1','files':files,'completed_unix':time.time()})


def waveform(n=90):
    if n!=90:raise ValueError('This frozen schedule requires 90 states')
    a=np.zeros(n,dtype=np.float64)
    smooth=lambda n: .5-.5*np.cos(np.linspace(0,np.pi,n))
    a[15:30]=.02*smooth(15); a[30:45]=.02
    a[45:60]=.02+.98*smooth(15);a[60:75]=1
    a[75:90]=1-smooth(15)
    # Same nominal/low plateaux; distinct dynamics during the transitions.
    b=a.copy();c=a.copy()
    b[15:30]=.02*smooth(15)**1.3;b[45:60]=.02+.98*smooth(15)**1.3
    c[15:30]=.02*smooth(15)**.8;c[45:60]=.02+.98*smooth(15)**.8
    return np.stack([a,b,c],axis=1)


def make_poses(geom,family,cfg):
    R=float(np.median(geom.base_radius));ref=[];moving=[]
    n=family['reference_frames'];start,turn,end=family['reference_progress']
    if n!=90 or family['action_frames']!=90:raise ValueError('Frozen benchmark schedule requires 90 reference and action frames')
    progress=np.r_[np.linspace(start,turn,60),np.linspace(turn,end,31)[1:]]
    def pose(s,phase):
        center=_interp_curve(geom.centerline,s)
        right=_interp_curve(geom.frame_right,s);up=_interp_curve(geom.frame_up,s)
        off=family['camera_offset_ratio']*R
        eye=(center+off*(np.cos(phase)*right+np.sin(phase)*up)).astype(np.float32)
        target=_interp_curve(geom.centerline,min(.98,s+3.5*R/float(cfg.tube.length)))
        rot=_rotation_from_lookat(eye,target,up)
        return rot,eye
    for i,s in enumerate(progress):ref.append(pose(s,2*np.pi*i/60+.27))
    for i,s in enumerate(np.linspace(*family['moving_progress'],family['action_frames'])):
        moving.append(pose(s,2*np.pi*i/65+.27))
    def arrays(rows):return np.stack([x[0] for x in rows]),np.stack([x[1] for x in rows])
    rr,rc=arrays(ref);mr,mc=arrays(moving)
    return {'reference_R':rr,'reference_C':rc,'moving_R':mr,'moving_C':mc,
            'stationary_R':np.repeat(rr[-1:],[family['action_frames']],axis=0),
            'stationary_C':np.repeat(rc[-1:],[family['action_frames']],axis=0)}


def _source_roots():
    """Logical archive paths remain portable; solver root matches the loader."""
    return {GENERATOR_LOGICAL_ROOT: Path(__file__).resolve().parents[2],
            'third_party/continuum-fem/src': _continuum_fem_src(),
            'third_party/continuum-fem': _continuum_fem_src().parent}


def _git_state(path):
    """Source archives are valid inputs even when Git metadata is unavailable."""
    def query(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        if Path(query('rev-parse', '--show-toplevel')).resolve() != Path(path).resolve():
            return {'head': None, 'status': None, 'available': False}
        return {'head': query('rev-parse', 'HEAD'), 'status': query('status', '--porcelain'), 'available': True}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'head': None, 'status': None, 'available': False}


def freeze(spec_path,out):
    roots = _source_roots()
    solver = roots['third_party/continuum-fem/src']
    if not solver.is_dir():
        raise RuntimeError(f'ContinuumFEM source directory not found: {solver}')
    spec=json.loads(spec_path.read_text())
    sources={}
    for logical, root in roots.items():
        paths = ([root/'backend/src', root/'tests', root/'configs', root/'docs']
                 if logical == GENERATOR_LOGICAL_ROOT else [root] if logical.endswith('/src') else [])
        for source_dir in paths:
            for p in source_dir.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.toml','.txt','.md'}:
                    sources[str(Path(logical)/p.relative_to(root))] = p
        if logical == GENERATOR_LOGICAL_ROOT:
            for relative in ['README.md', 'pytest.ini', 'backend/pyproject.toml', 'backend/requirements.txt', 'backend/requirements-benchmark.txt', 'backend/uv.lock']:
                p=root/relative
                if p.is_file(): sources[str(Path(logical)/relative)]=p
        for filename in ('LICENSE', 'LICENSE.md', 'COPYING', 'NOTICE'):
            p=root/filename
            if p.is_file(): sources[str(Path(logical)/filename)]=p
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'SPECIFICATION.json',spec)
    provenance=out/'provenance';provenance.mkdir()
    mapping={logical:sha(p) for logical,p in sorted(sources.items())}
    with tarfile.open(provenance/'sources.tar.gz','w:gz') as tar:
        for logical,p in sorted(sources.items()):tar.add(p,arcname=logical)
    states={'generator': _git_state(roots[GENERATOR_LOGICAL_ROOT]),
            'continuum_fem': _git_state(solver.parent)}
    write_json(provenance/'SOURCE_FREEZE.json',{'sources':mapping,'git':states,
               'source_layout':'logical_roots_v1', 'source_roots':{k:str(v) for k,v in roots.items()},
               'source_archive_sha256':sha(provenance/'sources.tar.gz')})
    import scipy
    write_json(provenance/'ENVIRONMENT.json',{'python':sys.version,'executable':sys.executable,'executable_sha256':sha(sys.executable),
               'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,
               'installed_distributions':sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
               'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}})
    write_json(out/'FROZEN.json',{'spec_sha256':sha(out/'SPECIFICATION.json'),'source_freeze_sha256':sha(provenance/'SOURCE_FREEZE.json'),
                'time':time.time(),'recovery_evaluated':False})


def get_spec(root,family_id):
    spec=json.loads((root/'SPECIFICATION.json').read_text())
    if sha(root/'SPECIFICATION.json')!=json.loads((root/'FROZEN.json').read_text())['spec_sha256']:raise ValueError('spec changed')
    return spec,next(f for f in spec['families'] if f['id']==family_id)


def verify_sources(root):
    if sha(root/'provenance'/'SOURCE_FREEZE.json')!=json.loads((root/'FROZEN.json').read_text())['source_freeze_sha256']:
        raise ValueError('Source freeze changed')
    freeze_record=json.loads((root/'provenance'/'SOURCE_FREEZE.json').read_text())
    recorded=freeze_record['sources']
    if freeze_record.get('source_layout') == 'logical_roots_v1':
        roots=_source_roots()
        # Existing sealed suites retain their original logical source paths.
        # Resolve those paths to this checkout for a meaningful hash comparison.
        roots[LEGACY_GENERATOR_LOGICAL_ROOT] = roots[GENERATOR_LOGICAL_ROOT]
        def resolve(logical):
            for prefix, directory in roots.items():
                if logical.startswith(prefix+'/'):
                    return directory/logical[len(prefix)+1:]
            raise ValueError(f'Unknown frozen source root: {logical}')
    else:
        # Original immutable suite uses workspace-relative paths.
        workspace=Path(__file__).resolve().parents[4]
        def resolve(logical): return workspace/logical
    changed=[p for p,h in recorded.items() if not resolve(p).is_file() or sha(resolve(p))!=h]
    if changed:raise ValueError(f'Frozen sources changed: {changed[:10]}')



def prepare(root,family_id):
    spec,family=get_spec(root,family_id)
    out=root/'families'/family_id;out.mkdir(parents=True,exist_ok=False)
    cfg=load_config_dict(family['generator_config']);preset=cfg.difficulty_presets['colon_benchmark']
    rng=np.random.default_rng(cfg.seed)
    geom=build_tube_static_geometry(cfg.tube,preset,rng)
    texture,features=generate_texture_with_qa(cfg.texture,np.random.default_rng(cfg.seed+1))
    v=tube_vertices_from_radius(geom,geom.base_radius)
    fields,fea,diag=solve_benchmark_fields(geom,cfg.tube,family['patches'],family['fem_peak_fractions'])
    nonfem=prescribed_fields(geom,family['patches'],family['nonfem_peak_fractions'])
    private=out/'evaluation';private.mkdir()
    np.savez_compressed(private/'geometry.npz',vertices=v,triangles=geom.triangles,uv=geom.uv,base_radius=geom.base_radius,
                        centerline=geom.centerline,frame_right=geom.frame_right,frame_up=geom.frame_up,tangent=geom.tangent,
                        s_norm=geom.s_norm,theta=geom.theta,vertex_material_id=np.arange(len(v)))
    cv2.imwrite(str(private/'texture.png'),texture)
    np.savez_compressed(private/'mechanics.npz',**fea)
    w=waveform(family['action_frames'])
    np.savez_compressed(private/'fields.npz',fem_basis=fields,nonfem_basis=nonfem,time_coefficients=w)
    poses=make_poses(geom,family,cfg);np.savez_compressed(private/'poses.npz',**poses)
    write_json(private/'MECHANICS.json',{'model':'linear isotropic C3D6 assembled on mapped curved shell nodes',
       'surface':'bilinear periodic interpolation of outer FE nodes; exact prescribed rendered kinematics',
       'boundary':'both end rings fixed','clinical_or_tissue_validation':False,'basis_diagnostics':diag,
       'maximum_superposed_engineering_strain_bound':float(np.sum(np.max(np.abs(fea['basis_gauss_strain']),axis=(1,2)))),
       'nonfem':'prescribed vector Gaussian fields with smooth fixed-end envelope; no equilibrium claim',
       'radius':float(np.median(geom.base_radius)), 'K_format':'CSR',
       'gauss_strain_components':['xx','yy','zz','engineering_xy','engineering_yz','engineering_zx'],
       'units':'abstract length; force and E consistently used numerically, not physical tissue calibration'})
    intr=asdict(cfg.intrinsics);write_json(out/'public'/'intrinsics.json',intr)
    write_json(out/'public'/'acquisition.json',{'reference_frames':90,'action_frames':90,'fps':spec['fps'],
        'stationarity':'camera physically stationary throughout stationary action, registered using terminal rigid scan image',
        'reference_bridge_image':'reference/rgb/000089.png','reference_is_unloaded_anatomy':False,
        'moving_action':'separate repositioned capture; no reference-to-moving continuity or cross-boundary local window',
        'clock':'logical acquisition timestamps; paired action streams share material-state times, not simultaneous physical cameras'})
    write_json(out/'PREPARATION.json',{'family':family_id,'split':family['split'],'texture_orb':features,'inference_fitted':False,
                'config':family,'surface_vertices':len(v),'triangles':len(geom.triangles),'prepared':time.time()})
    seal(private)


def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.symlink_to(os.path.relpath(src,dst.parent),target_is_directory=src.is_dir())


def create_package(root,family_id,condition):
    spec,family=get_spec(root,family_id)
    fi=spec['families'].index(family);ci=spec['conditions'].index(condition)
    version=3+3*fi+ci;out=root/f'v{version}';out.mkdir(exist_ok=False)
    common=root/'families'/family_id
    link(common/'public'/'intrinsics.json',out/'public'/'intrinsics.json')
    link(common/'public'/'acquisition.json',out/'public'/'acquisition.json')
    link(common/'public'/'reference',out/'public'/'reference')
    link(common/'evaluation',out/'evaluation'/'family')
    link(common/'reference_truth',out/'evaluation'/'reference')
    write_json(out/'DATASET.json',{'schema':spec['schema'],'version':version,'family':family_id,'condition':condition,
        'split':family['split'],'reference':'public/reference','streams':['moving','stationary'],
        'truth':'evaluation','inference_allowlist':['public','reconstruction'],
        'contrasts':spec['contrasts'],'paired_acquisitions':True,'independent_family_count':1,
        'ground_truth_at_inference':False,'generated_synthetic':True})
    return out


def render(root,family_id,condition,stream):
    spec,family=get_spec(root,family_id);common=root/'families'/family_id
    cfg=load_config_dict(family['generator_config']);private=common/'evaluation'
    geom=np.load(private/'geometry.npz');pose=np.load(private/'poses.npz');field=np.load(private/'fields.npz')
    base=geom['vertices'];tris=geom['triangles'];texture=cv2.imread(str(private/'texture.png'))
    if stream=='reference':
        out=common/'public'/'reference';truth=common/'reference_truth';n=family['reference_frames'];condition='static'
        verts=np.repeat(base[None],n,axis=0)
    else:
        version=3+spec['families'].index(family)*3+spec['conditions'].index(condition)
        package=root/f'v{version}'
        out=package/'public'/stream;truth=package/'evaluation'/stream;n=family['action_frames']
        if condition=='static':disp=np.zeros((n,*base.shape),dtype=np.float32)
        else:disp=np.einsum('tk,kijc->tijc',field['time_coefficients'],field[condition+'_basis']).reshape(n,-1,3).astype(np.float32)
        verts=(base[None]+disp).astype(np.float32)
    out.mkdir(parents=True,exist_ok=False);truth.mkdir(parents=True,exist_ok=False)
    (out/'rgb').mkdir();(truth/'dense').mkdir();(truth/'flow_forward').mkdir()
    rotations=pose[stream+'_R'];centers=pose[stream+'_C']
    prescribed=np.zeros_like(verts) if stream=='reference' else disp
    np.savez_compressed(truth/'states.npz',vertices=verts,displacement=(verts.astype(np.float64)-base.astype(np.float64)),
                         prescribed_displacement=prescribed,
                         R_wc=rotations,C_world=centers,action_time_seconds=np.arange(n)/spec['fps'],
                         time_seconds=(np.arange(n)+(0 if stream=='reference' else 90))/spec['fps'])
    records=[];quality=[];start=time.monotonic()
    for t in range(n):
        rgb,buffers=render_rgb_and_truth(verts[t],tris,geom['uv'],texture,cfg.intrinsics,rotations[t],centers[t],periodic_u=True,high_precision=True)
        rng=np.random.default_rng(cfg.seed+100000+(0 if stream=='reference' else 10000 if stream=='moving' else 20000)+t)
        rgb=apply_photometric_noise(rgb,rng,cfg.photometric)
        path=out/'rgb'/f'{t:06d}.png'
        if not cv2.imwrite(str(path),rgb):raise IOError(path)
        np.savez_compressed(truth/'dense'/f'{t:06d}.npz',**buffers)
        if t<n-1:
            flow=flow_to_frame(buffers,verts[t+1],tris,cfg.intrinsics,rotations[t+1],centers[t+1],high_precision=True)
            np.savez_compressed(truth/'flow_forward'/f'{t:06d}.npz',**flow)
        records.append({'image':path.name,'timestamp_s':(t if stream=='reference' else 90+t)/spec['fps'],
                        'state_index':t,'acquisition':stream})
        quality.append({'frame':t,'orb_features':orb_feature_count(rgb),'surface_fraction':float(buffers['valid'].mean())})
        if t%15==0:print(json.dumps({'family':family_id,'condition':condition,'stream':stream,'frame':t,'elapsed_s':time.monotonic()-start}),flush=True)
    write_json(out/'frames.json',{'frames':records,'intrinsics':'../intrinsics.json','uses_truth':False})
    write_json(truth/'QUALITY.json',quality)
    write_json(truth/'CONVENTIONS.json',{'camera':'x right, y up, z forward; p_c=R_wc.T@(p_w-C)',
        'high_precision':True,'arithmetic':'exact stored float32 geometry/camera values promoted before subtraction; float64 camera transforms, projection, raster weights, z-buffer and flow visibility; float32 stored depth/material barycentrics',
        'texture':'periodic angular U: unwrap crossing triangles before physical barycentric interpolation, modulo1; nearest floor(U*(width-1)); terminal fan caps use arbitrary inherited closure UV, not a physical or clinical texture map',
        'projection':'u=fx*x/z+cx; v=cy-fy*y/z; samples at col+.5,row+.5',
        'COLMAP_conversion':'camera/world reflection F=diag(1,-1,1); C_colmap=F C; R_cw_colmap=F R_wc.T F',
        'units':'abstract length units, no millimetre/tissue claim; depth is camera z, not ray distance',
        'state':'vertices are exact float32 raster inputs; displacement is float64 difference from float32 reference',
        'normal':'world-space triangle winding; no camera-facing reorientation','flow':'forward projected same material triangle/barycentric; final frame absent by definition',
        'clipping':'whole triangle discarded if any camera vertex depth <=1e-5; no polygon near clipping',
        'background':'valid=false; triangle=-1; invalid entries must be masked',
        'truth_excluded_from_inference':True})
    seal(out);seal(truth)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','prepare','package','render'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--spec',type=Path)
    p.add_argument('--family');p.add_argument('--condition',choices=['static','fem','nonfem']);p.add_argument('--stream',choices=['reference','moving','stationary'])
    args=p.parse_args();cv2.setNumThreads(1)
    if args.stage=='freeze':freeze(args.spec,args.root)
    else:
        verify_sources(args.root)
        if args.stage=='prepare':prepare(args.root,args.family)
        elif args.stage=='package':create_package(args.root,args.family,args.condition)
        else:render(args.root,args.family,args.condition,args.stream)


if __name__=='__main__':main()
