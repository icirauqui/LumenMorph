"""Regenerate family preparation and selected complete observations in a fresh root.

Invoke from the frozen producer checkout/runtime. Compares numerical arrays and
image pixels, separately from ZIP timestamps and execution metadata.
"""
import argparse,json,shutil,sys
from pathlib import Path
import cv2
import numpy as np
from . import benchmark
from .config import load_config_dict
from .dense_truth import render_rgb_and_truth,flow_to_frame
from .noise import apply_photometric_noise
from .file_texture import load_texture,image_from_material


def compare_npz(a,b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return {k:bool(k in x.files and k in y.files and np.array_equal(x[k],y[k],equal_nan=True)) for k in sorted(set(x.files)|set(y.files))}


def evaluate(root,work,output):
    root,work,output=map(Path,(root,work,output))
    if work.exists() or output.exists():raise FileExistsError('Fresh work and output required')
    work.mkdir(parents=True);spec=json.loads((root/'SPECIFICATION.json').read_text())
    for file in ['SPECIFICATION.json','FROZEN.json']:shutil.copyfile(root/file,work/file)
    shutil.copytree(root/'provenance',work/'provenance');shutil.copytree(root/'assets',work/'assets')
    benchmark.verify_sources(root);families=[]
    for family in spec['families']:
        name=family['id']
        if not (root/'families'/name).exists():continue
        benchmark.prepare(work,name);old=root/'families'/name/'evaluation';new=work/'families'/name/'evaluation'
        checks={p:compare_npz(old/p,new/p) for p in ['geometry.npz','fields.npz','mechanics.npz','poses.npz']}
        checks['texture']={'pixels':bool(np.array_equal(cv2.imread(str(old/'texture.png')),cv2.imread(str(new/'texture.png'))))}
        cfg=load_config_dict(family['generator_config']);texture=cv2.imread(str(new/'texture.png'))
        with np.load(new/'geometry.npz') as z:g=dict(z)
        with np.load(new/'fields.npz') as z:field=dict(z)
        with np.load(new/'poses.npz') as z:pose=dict(z)
        external={k:load_texture(work,v) for k,v in family['file_textures'].items()};frames=[]
        for condition,stream,t in [('static','reference',0),('fem','moving',family['action_frames']//2),('nonfem','moving',family['action_frames']//2)]:
            if stream=='reference':truth=root/'families'/name/'reference_truth';public=root/'families'/name/'public/reference'
            else:
                package=root/f"v{family['package_base']+spec['conditions'].index(condition)}";truth=package/'evaluation'/stream;public=package/'public'/stream
            # Match the producer's batched contraction order exactly.
            if condition!='static':
                displacements=np.einsum('tk,kijc->tijc',field['time_coefficients'],field[condition+'_basis']).reshape(family['action_frames'],-1,3).astype(np.float32)
                v=(g['vertices']+displacements[t]).astype(np.float32);vt=(g['vertices']+displacements[t+1]).astype(np.float32)
            else:v=vt=g['vertices']
            rgb,dense=render_rgb_and_truth(v,g['triangles'],g['uv'],texture,cfg.intrinsics,pose[stream+'_R'][t],pose[stream+'_C'][t],periodic_u=True,high_precision=True,**({'material_barycentric_dtype':family['material_barycentric_dtype']} if 'material_barycentric_dtype' in family else {}))
            with np.load(truth/'dense'/f'{t:06d}.npz') as z:dense_checks={k:bool(np.array_equal(dense[k],z[k],equal_nan=True)) for k in z.files}
            image_checks={}
            for variant in ['procedural']+list(external):
                image=rgb if variant=='procedural' else image_from_material(dense,g['triangles'],g['uv'],external[variant],family['file_textures'][variant])
                rng=np.random.default_rng(cfg.seed+100000+(0 if stream=='reference' else 10000)+t);image=apply_photometric_noise(image,rng,cfg.photometric)
                path=public/('rgb' if variant=='procedural' else 'rgb_'+variant)/f'{t:06d}.png'
                image_checks[variant]=bool(np.array_equal(image,cv2.imread(str(path))))
            flow=flow_to_frame(dense,vt,g['triangles'],cfg.intrinsics,pose[stream+'_R'][t+1],pose[stream+'_C'][t+1],high_precision=True)
            with np.load(truth/'flow_forward'/f'{t:06d}.npz') as z:flow_checks={k:bool(np.array_equal(flow[k],z[k],equal_nan=True)) for k in z.files}
            frames.append(dict(condition=condition,stream=stream,frame=t,dense=dense_checks,images=image_checks,flow=flow_checks,passed=all(dense_checks.values()) and all(image_checks.values()) and all(flow_checks.values())))
        families.append(dict(family=name,preparation=checks,observations=frames,passed=all(all(x.values()) for x in checks.values()) and all(x['passed'] for x in frames)))
        print(json.dumps({'family':name,'passed':families[-1]['passed']}),flush=True)
    result=dict(schema='collection_producer_replay_v1',source_sha256=benchmark.sha(__file__),producer_freeze_sha256=benchmark.sha(root/'provenance/SOURCE_FREEZE.json'),families=families,passed=all(x['passed'] for x in families),scope='Fresh preparation and selected FE/non-FE/static observations, both textures and temporal flow; exact arrays/pixels in the recorded runtime, not cross-platform equality.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','work','output']:p.add_argument('--'+name,required=True)
    sys.exit(0 if evaluate(**vars(p.parse_args()))['passed'] else 1)
