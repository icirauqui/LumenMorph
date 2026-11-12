"""All-pixel first-hit validation with independent native world-space BVH rays."""
import argparse,json,hashlib,time,platform,sys
from pathlib import Path
import numpy as np
from ray_backend import Rays
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--condition',required=True);p.add_argument('--stream',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--material-weight-policy',choices=['stored','unit_sum']);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((a.root/'SPECIFICATION.json').read_text());f=next(f for f in spec['families'] if f['id']==a.family);intr=f['generator_config']['intrinsics'];policy=a.material_weight_policy or f.get('material_weight_policy','stored');common=a.root/'families'/a.family
folder=common/'reference_truth' if a.stream=='reference' else a.root/f"v{f['package_base']+spec['conditions'].index(a.condition)}"/'evaluation'/a.stream
with np.load(common/'evaluation/geometry.npz') as z:tri=z['triangles']
with np.load(folder/'states.npz') as z:verts=z['vertices'];Rs=z['R_wc'];Cs=z['C_world']
h,w=intr['height'],intr['width'];rr,cc=np.indices((h,w));pixel=np.c_[cc.ravel()+.5,rr.ravel()+.5];camera=np.c_[(pixel[:,0]-intr['cx'])/intr['fx'],(intr['cy']-pixel[:,1])/intr['fy'],np.ones(h*w)];frames=[]
inputs={str(q.relative_to(a.root)):sha(q) for q in [a.root/'SPECIFICATION.json',common/'evaluation/geometry.npz',folder/'states.npz']}
for i in range(len(verts)):
 start=time.monotonic();v=verts[i].astype(float);R=Rs[i].astype(float);C=Cs[i].astype(float);ray=Rays(v,tri);rid,z,b=ray(C,camera@np.linalg.inv(R));ray.close()
 dp=folder/'dense'/f'{i:06d}.npz';inputs[str(dp.relative_to(a.root))]=sha(dp)
 with np.load(dp) as q:valid=q['valid'].ravel();ids=q['triangle_id'].ravel();bary=q['material_barycentric'].reshape(-1,3).astype(float);depth=q['depth_z'].ravel().astype(float)
 hit=rid>=0;both=hit&valid;ix=np.flatnonzero(both);bg=(~hit)&(~valid);failure=(hit!=valid)|(valid!=(ids>=0));failure|=bg&np.isfinite(depth)
 observed_ok=both&(ids>=0)&(ids<len(tri));failure|=both&~observed_ok;ii=np.flatnonzero(observed_ok)
 weights=bary[ii]/bary[ii].sum(1)[:,None] if policy=='unit_sum' else bary[ii]
 point=np.einsum('ni,nij->nj',weights,v[tri[ids[ii]]]);pc=(point-C)@R
 projected=np.c_[intr['fx']*pc[:,0]/pc[:,2]+intr['cx'],intr['cy']-intr['fy']*pc[:,1]/pc[:,2]]
 re=np.linalg.norm(projected-pixel[ii],axis=1);de=abs(depth[ii]-z[ii])/np.maximum(abs(z[ii]),1e-12);pe=abs(pc[:,2]-z[ii])/np.maximum(abs(z[ii]),1e-12)
 independent_point=C+z[ii,None]*(camera@np.linalg.inv(R))[ii];point_error=np.linalg.norm(point-independent_point,axis=1)
 different=ids[ii]!=rid[ii];ties=different&(de<=1e-5)&(b[ii].min(1)<=1e-5)
 badweight=(abs(bary[ii].sum(1)-1)>1e-5)|(bary[ii].min(1)<-1e-5)|~np.isfinite(bary[ii]).all(1)
 bad=(re>=1e-3)|(de>=1e-5)|(pe>=1e-5)|(different&~ties)|badweight|~np.isfinite(re+de+pe)
 failure[ii]|=bad
 failed=np.flatnonzero(failure)
 if len(failed):np.savez_compressed(a.output/f'failures_{i:06d}.npz',flat_pixels=failed,stored_id=ids[failed],independent_id=rid[failed],stored_depth=depth[failed],independent_depth=z[failed],stored_weights=bary[failed],independent_weights=b[failed])
 same=~different
 record=dict(frame=i,pixels=h*w,foreground=int(valid.sum()),independent_foreground=int(hit.sum()),background=int(bg.sum()),failures=len(failed),foreground_mismatches=int((hit!=valid).sum()),identity_mismatches=int((different&~ties).sum()),accepted_edge_ties=int(ties.sum()),near_edge_pixels=int((b[ii].min(1)<=1e-5).sum()),depth_relative_max=float(de.max(initial=0)),material_depth_relative_max=float(pe.max(initial=0)),reprojection_px_max=float(re.max(initial=0)),point_distance_max=float(point_error.max(initial=0)),same_face_barycentric_max=float(abs(bary[ii][same]-b[ii][same]).max(initial=0)),weight_sum_max=float(abs(bary[ii].sum(1)-1).max(initial=0)),seconds=time.monotonic()-start)
 frames.append(record);(a.output/'progress.json').write_text(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
report=dict(material_weight_policy=policy,passed=all(r['failures']==0 for r in frames),family=a.family,condition=a.condition,stream=a.stream,frames=frames,pixels=sum(r['pixels'] for r in frames),failures=sum(r['failures'] for r in frames),inputs=inputs,source_sha256=sha(Path(__file__)),native_source_sha256=sha(Path(__file__).with_name('native_rays.cpp')),native_library_sha256=sha(Path(__file__).with_name('native_rays.so')),platform=platform.platform(),numpy=np.__version__,method='All pixel-center rays; independent double-precision Moller-Trumbore with conservative world-space BVH; no producer interpolation',thresholds=dict(depth_relative=1e-5,material_depth_relative=1e-5,reprojection_pixels=1e-3,edge_barycentric=1e-5))
(a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')

sys.exit(0 if report["passed"] else 1)
