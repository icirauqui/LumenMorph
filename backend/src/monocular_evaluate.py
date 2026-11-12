"""One declared-component Sim(3) evaluation of frozen camera/material tracks.

NPZ: frame_ids[T], centers[T,3], rotations[T,3,3], reference_pixels[N,2]
(integer row,col), points[T,N,3], optional mask[T,N] and normals[T,N,3].
Reference pixels establish observed material tracks, not an inferred 3D gauge.
All rotations map camera axes x-right/y-up/z-forward into prediction world.
The JSON protocol must predeclare anchor_frame_ids and reference_frame.
No independent per-frame transforms or component merging are performed.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .benchmark_reader import EvaluationDataset
from .benchmark_evaluate import digest


def fit_similarity(source,target,ratio=1e-6):
    if not np.isfinite(ratio) or not 0<ratio<1:raise ValueError('Invalid anchor degeneracy threshold')
    source,target=np.asarray(source,float),np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('At least three finite paired anchors required')
    a=source-source.mean(0);b=target-target.mean(0)
    for x in (a,b):
        sv=np.linalg.svd(x,compute_uv=False)
        if sv[0]<=1e-15 or sv[1]/sv[0]<ratio:raise ValueError('Degenerate anchor configuration')
    u,s,vt=np.linalg.svd(b.T@a);D=np.eye(3);D[-1,-1]=np.linalg.det(u@vt);R=u@D@vt
    scale=float(np.sum(s*np.diag(D))/np.sum(a*a))
    if scale<=0:raise ValueError('Nonpositive alignment scale')
    return scale,R,target.mean(0)-scale*R@source.mean(0)


def transform(scale,R,t,*,centers,rotations,points,normals=None,displacements=None):
    out={'centers':scale*np.asarray(centers)@R.T+t,'rotations':R@np.asarray(rotations),'points':scale*np.asarray(points)@R.T+t}
    if normals is not None:
        n=np.asarray(normals)@R.T;length=np.linalg.norm(n,axis=-1,keepdims=True)
        out['normals']=np.divide(n,length,out=np.full_like(n,np.nan),where=length>0)
    if displacements is not None:out['displacements']=scale*np.asarray(displacements)@R.T
    return out


def summary(error,eligible,available):
    n=int(eligible.sum());m=eligible&available&np.isfinite(error);count=int(m.sum())
    return {'eligible':n,'scored':count,'coverage':count/n if n else None,
            'conditional_mean':float(error[m].mean()) if count else None,
            'conditional_max':float(error[m].max()) if count else None,
            'full_population_mean':float(error[m].mean()) if count==n and n else None}


def evaluate(package,predictions,protocol,output,stream='moving'):
    package,predictions,protocol,output=map(lambda p:Path(p).resolve(),(package,predictions,protocol,output))
    if output==package.parent or package.parent in output.parents:raise ValueError('Output must be outside immutable suite')
    if output.exists():raise FileExistsError(output)
    ph,ah=digest(predictions),digest(protocol);contract=json.loads(protocol.read_text())
    if contract['schema']!='material_tracks_alignment_v1' or not contract.get('component_id'):raise ValueError('Declare one component and schema')
    anchors=contract['anchor_frame_ids'];reference=contract['reference_frame']
    if len(anchors)!=len(set(anchors)) or any(type(x) is not int for x in anchors) or type(reference) is not int:raise ValueError('Invalid anchor/reference IDs')
    with np.load(predictions,allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
    required={'frame_ids','centers','rotations','reference_pixels','points'}
    if not required<=set(pred)<=required|{'normals','mask'}:raise ValueError('Invalid reconstruction fields')
    ids=pred['frame_ids'];pix=pred['reference_pixels'];T=len(ids);N=len(pix)
    if ids.dtype.kind not in 'iu' or ids.ndim!=1 or len(set(ids.tolist()))!=T or pix.shape!=(N,2) or pix.dtype.kind not in 'iu':raise ValueError('Invalid frame/pixel IDs')
    if pred['centers'].shape!=(T,3) or pred['rotations'].shape!=(T,3,3) or pred['points'].shape!=(T,N,3):raise ValueError('Invalid reconstruction shape')
    if contract.get('frame_ids')!=ids.tolist():raise ValueError('Keep every predeclared frame; use nonfinite arrays/masks for missing results')
    if T==0 or N==0 or len(np.unique(pix,axis=0))!=N:raise ValueError('Require nonempty unique reference samples')
    if 'normals' in pred and pred['normals'].shape!=(T,N,3):raise ValueError('Invalid normal shape')
    pixel_hash=hashlib.sha256(np.asarray(pix,dtype='<i8').tobytes()).hexdigest()
    if contract.get('reference_pixels_sha256')!=pixel_hash:raise ValueError('Reference observations must match the predeclared protocol hash')
    mask=pred.get('mask',np.ones((T,N),bool))
    if mask.dtype!=bool or mask.shape!=(T,N):raise ValueError('mask must be boolean T,N')
    result=dict(schema='monocular_component_score_v1',component_id=contract['component_id'],prediction_sha256=ph,protocol_sha256=ah,
                alignment='ground-truth-aligned positive-scale proper Sim(3); one transform for declared component',
                absolute_scale_recovered=False,independent_components_combined=False,evaluator_sha256=digest(__file__))
    # Truth opens only after all submitted arrays/protocol are read and hashed.
    reader=EvaluationDataset(package);dense=reader.dense(reference,stream)
    if np.any(pix<0) or np.any(pix[:,0]>=dense['valid'].shape[0]) or np.any(pix[:,1]>=dense['valid'].shape[1]):raise ValueError('Reference pixel outside image')
    result['reference_image_valid_pixels']=int(dense['valid'].sum())
    result['requested_reference_observations']=N
    row,col=pix.T;good=dense['valid'][row,col];
    result['reference_image_sampling_fraction']=float(good.sum()/dense['valid'].sum()) if dense['valid'].any() else None
    result['population']='predeclared reference observations across requested frames; image sampling fraction reported separately'
    tri=dense['triangle_id'][row,col][good];weights=dense['material_barycentric'][row,col][good]
    weights=reader.material_weights(weights,stream)
    # Decompress each required state array once, rather than once per query.
    for frame in ids:reader._frame(stream,int(frame))
    geometry=reader.geometry();faces=geometry['triangles'][tri]
    with np.load(reader.truth_path(stream)/'states.npz',allow_pickle=False) as archive:
        vertices=archive['vertices'][ids];cameras=archive['C_world'][ids];rotations=archive['R_wc'][ids]
    truth=np.full((T,N,3),np.nan)
    for i in range(T):truth[i,good]=np.sum(weights[...,None]*vertices[i,faces].astype(float),axis=1)
    try:
        if any(x not in ids for x in anchors):raise ValueError('Missing predeclared anchor')
        idx=[list(ids).index(x) for x in anchors]
        scale,R,t=fit_similarity(pred['centers'][idx],cameras[idx],contract.get('minimum_second_singular_ratio',1e-6))
        aligned=transform(scale,R,t,centers=pred['centers'],rotations=pred['rotations'],points=pred['points'],normals=pred.get('normals'))
        valid_camera=np.isfinite(pred['centers']).all(-1)&np.isfinite(pred['rotations']).all((1,2))
        valid_camera &= np.max(np.abs(np.swapaxes(pred['rotations'],1,2)@pred['rotations']-np.eye(3)),axis=(1,2))<1e-5
        valid_camera &= np.abs(np.linalg.det(pred['rotations'])-1)<1e-5
        result.update(status='aligned',scale=scale,rotation=R.tolist(),translation=t.tolist(),
            camera_position=summary(np.linalg.norm(aligned['centers']-cameras,axis=-1),np.ones(T,bool),valid_camera),
            material_position=summary(np.linalg.norm(aligned['points']-truth,axis=-1),np.broadcast_to(good,(T,N)),mask))
        angle=np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(rotations,1,2)@aligned['rotations'],axis1=1,axis2=2)-1)/2,-1,1)))
        result['camera_rotation_degrees']=summary(angle,np.ones(T,bool),valid_camera)
        if reference in ids:
            j=list(ids).index(reference);result['material_displacement']=summary(np.linalg.norm((aligned['points']-aligned['points'][j])-(truth-truth[j]),axis=-1),np.broadcast_to(good,(T,N)),mask&mask[j])
        if 'normals' in aligned:
            normals=np.full_like(truth,np.nan)
            for i,frame in enumerate(ids):
                v=vertices[i,faces].astype(float);n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);normals[i,good]=n/np.linalg.norm(n,axis=-1,keepdims=True)
            err=np.degrees(np.arccos(np.clip(np.sum(aligned['normals']*normals,axis=-1),-1,1)));result['normal_degrees']=summary(err,np.broadcast_to(good,(T,N)),mask)
    except ValueError as exc:
        result.update(status='alignment_failed',reason=str(exc),camera_position={'eligible':T,'scored':0,'coverage':0.},material_position={'eligible':int(T*good.sum()),'scored':0,'coverage':0.},scale=None)
    if ph!=digest(predictions) or ah!=digest(protocol):raise ValueError('Frozen inputs changed')
    result['truth_sha256']={str(p.relative_to(package)):digest(p) for p in [package/'DATASET.json',reader.truth_path(stream)/'states.npz',package/'evaluation/family/geometry.npz',reader.truth_path(stream)/'dense'/f'{reference:06d}.npz',package/'public/intrinsics.json',reader.public_paths(stream)['frames']]}
    conventions=reader.truth_path(stream)/'CONVENTIONS.json'
    if conventions.is_file():result['truth_sha256'][str(conventions.relative_to(package))]=digest(conventions)
    result['reader_sha256']=digest(Path(__file__).with_name('benchmark_reader.py'))
    result['material_decoder_sha256']=digest(Path(__file__).with_name('material_weights.py'))
    output.parent.mkdir(parents=True,exist_ok=True);
    with output.open('x') as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['package','predictions','protocol','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--stream',default='moving');evaluate(**vars(p.parse_args()))

if __name__=='__main__':main()
