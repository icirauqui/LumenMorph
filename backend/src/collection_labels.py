"""Census plus deterministic stratified independent-ray checks of one stream.

This verifier uses world-space Moller-Trumbore intersections and a 3D R-tree,
not the producer's screen-space rasterizer or its continuous visibility query.
Strata overlap deliberately; counts are reported separately, never summed into
an allegedly independent sample size. All image pixels enter the census.
"""
import argparse,json,platform,sys
from pathlib import Path
import cv2
import numpy as np
from .benchmark import sha
from .benchmark_validation import _directions,_project


def rays(vertices,triangles,origin,directions):
    import trimesh
    from trimesh.ray.ray_triangle import ray_triangle_candidates
    mesh=trimesh.Trimesh(vertices,triangles,process=False)
    candidate,ray_id=ray_triangle_candidates(np.broadcast_to(origin,directions.shape),directions,mesh.triangles_tree)
    n=len(directions);depth=np.full(n,np.inf);ids=np.full(n,-1,int);bary=np.full((n,3),np.nan)
    for start in range(0,len(candidate),100000):
        ix=candidate[start:start+100000];ri=ray_id[start:start+100000];v=vertices[triangles[ix]]
        a,b=v[:,1]-v[:,0],v[:,2]-v[:,0];d=directions[ri];h=np.cross(d,b);det=np.einsum('ij,ij->i',a,h)
        ok=np.abs(det)>1e-13*np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(d,axis=1),1e-30)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);s=origin-v[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,a)
        vv=np.einsum('ij,ij->i',d,q)*inv;z=np.einsum('ij,ij->i',b,q)*inv
        ok &= (u>=-1e-10)&(vv>=-1e-10)&(u+vv<=1+1e-10)&(z>1e-5)
        order=np.lexsort((ix[ok],z[ok],ri[ok]));good=np.flatnonzero(ok)[order]
        if not len(good):continue
        first=np.r_[True,np.diff(ri[good])!=0];good=good[first];better=z[good]<depth[ri[good]];good=good[better];r=ri[good]
        depth[r]=z[good];ids[r]=ix[good];bary[r]=np.column_stack((1-u[good]-vv[good],u[good],vv[good]))
    return ids,depth,bary


def material_uv(uv,triangles,ids,weights):
    q=uv[triangles[ids]].astype(float);u=q[:,:,0]%1
    q[:,:,0]=u+((np.ptp(u,axis=1)>.5)[:,None]&(u<.5))
    return np.einsum('ni,nij->nj',weights,q)


def external_sample(image,uv,spec):
    """Independent scalar linear-light periodic texture sampler for QA."""
    image=np.rot90(image,spec['orientation']//90);h,w=image.shape[:2];out=[]
    for coordinate in uv:
        u,v=(coordinate*np.array(spec['repeat'])+np.array(spec.get('offset',[0,0])))%1
        x,y=u*w-.5,(1-v)*h-.5;i,j=int(np.floor(x)),int(np.floor(y));a,b=x-i,y-j
        if spec['interpolation']=='nearest':color=image[int(np.floor(y+.5))%h,int(np.floor(x+.5))%w]
        else:
            color=np.zeros(3)
            for di,dj,weight in [(0,0,(1-a)*(1-b)),(1,0,a*(1-b)),(0,1,(1-a)*b),(1,1,a*b)]:
                c=image[(j+dj)%h,(i+di)%w].astype(float)/255
                color+=weight*np.array([v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in c])
            color=np.rint(np.clip([12.92*v if v<=.0031308 else 1.055*v**(1/2.4)-.055 for v in color],0,1)*255).astype(np.uint8)
        out.append(color)
    return np.array(out,np.uint8)


def evaluate(root,family,condition,stream,output,flow_root=None,material_weight_policy=None):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    spec=json.loads((root/'SPECIFICATION.json').read_text());f=next(x for x in spec['families'] if x['id']==family)
    policy=material_weight_policy or f.get('material_weight_policy','stored')
    if policy not in ('stored','unit_sum'):raise ValueError('Unknown material weight policy')
    def geometric_weights(values):
        # Independent decoder: keep texture arithmetic separate from geometry.
        weights=np.asarray(values,dtype=float)
        return weights/weights.sum(axis=-1,keepdims=True) if policy=='unit_sum' else weights
    common=root/'families'/family;private=common/'evaluation'
    if stream=='reference':public=common/'public/reference';truth=common/'reference_truth'
    else:
        package=root/f"v{f['package_base']+spec['conditions'].index(condition)}";public=package/'public'/stream;truth=package/'evaluation'/stream
    with np.load(private/'geometry.npz') as z:g={k:z[k] for k in z.files}
    with np.load(truth/'states.npz') as z:states={k:z[k] for k in z.files}
    intr=f['generator_config']['intrinsics'];tri=g['triangles'];uv=g['uv'];T=len(states['vertices']);Rnom=f['generator_config']['tube']['dimensions']['base_radius']
    tex=cv2.imread(str(private/'texture.png'));external={k:cv2.imread(str(root/v['file'])) for k,v in f['file_textures'].items()}
    # A triangle entirely on the first/last longitudinal ring is an artificial cap.
    longitudinal=uv[tri,1];cap=(np.max(longitudinal,axis=1)<1e-7)|(np.min(longitudinal,axis=1)>1-1e-7)
    inputs={str(p.relative_to(root)):sha(p) for p in [private/'geometry.npz',truth/'states.npz',root/'SPECIFICATION.json',public/'frames.json']}
    results=[]
    for frame in range(T):
        dp=truth/'dense'/f'{frame:06d}.npz'
        with np.load(dp) as z:d={k:z[k] for k in z.files}
        inputs[str(dp.relative_to(root))]=sha(dp)
        valid=d['valid'];row,col=np.nonzero(valid);ids=d['triangle_id'][valid];b=d['material_barycentric'][valid].astype(float)
        vertices=states['vertices'][frame].astype(float);C=states['C_world'][frame].astype(float);R=states['R_wc'][frame].astype(float)
        geometric_b=geometric_weights(b)
        points=np.einsum('ni,nij->nj',geometric_b,vertices[tri[ids]]);projected,z=_project(points,R,C,intr);pixel=np.column_stack((col+.5,row+.5))
        faces=vertices[tri];normal=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        reproject=float(np.max(np.linalg.norm(projected-pixel,axis=1)));depth=float(np.max(np.abs(z-d['depth_z'][valid])/np.maximum(np.abs(z),1e-12)))
        ne=float(np.max(np.linalg.norm(normal[ids]-d['normal_world'][valid],axis=1)));weight_error=float(np.max(np.abs(b.sum(1)-1)))
        direction=points-C;direction/=np.linalg.norm(direction,axis=1,keepdims=True);cosine=np.abs(np.sum(normal[ids]*direction,axis=1))
        def mask(values):
            a=np.zeros(valid.shape,bool);a[valid]=values;return a
        seam=np.ptp(uv[tri[ids],0],axis=1)>.5
        strata={'random':np.ones(valid.shape,bool),'background':~valid,'triangle_edge':mask(b.min(1)<.02),
                'grazing':mask(cosine<.1),'periodic_seam':mask(seam),'artificial_cap':mask(cap[ids])}
        flow=None
        if frame<T-1:
            fp=(Path(flow_root) if flow_root else truth/'flow_forward')/f'{frame:06d}.npz'
            with np.load(fp) as file:flow={k:file[k] for k in file.files}
            inputs[str(fp.relative_to(root)) if root in fp.parents else str(fp)]=sha(fp)
            categories=['visible','occluded','out_of_frame','behind_camera','unresolved']
            partition=sum(flow[k].astype(np.uint8) for k in categories)
            partition_ok=np.array_equal(partition,valid.astype(np.uint8))
            for k in categories:strata[k]=flow[k]
        else:partition_ok=True
        rng=np.random.default_rng(61111000+frame);samples={k:rng.choice(np.flatnonzero(m),min(64,int(m.sum())),replace=False) for k,m in strata.items()}
        selected=np.unique(np.concatenate(list(samples.values())));rr,cc=np.unravel_index(selected,valid.shape)
        pixels=np.column_stack((cc+.5,rr+.5));ray_ids,ray_depth,ray_bary=rays(vertices,tri,C,_directions(pixels,intr,np.linalg.inv(R.T)))
        observed=d['triangle_id'][rr,cc];both=(ray_ids>=0)&(observed>=0);mismatch=(ray_ids>=0)!=(observed>=0)
        ray_error=np.zeros(len(selected));ray_error[both]=np.abs(ray_depth[both]-d['depth_z'][rr[both],cc[both]])/np.maximum(np.abs(ray_depth[both]),1e-12)
        mismatch |= ray_error>1e-5
        different=both&(observed!=ray_ids);ties=different&(ray_error<=1e-5)&(np.min(ray_bary,axis=1)<=1e-5);mismatch |=different&~ties
        flow_error=np.zeros(len(selected));visibility_bad=np.zeros(len(selected),bool)
        if flow is not None:
            ii=np.flatnonzero(observed>=0);q=np.einsum('ni,nij->nj',geometric_weights(d['material_barycentric'][rr[ii],cc[ii]]),states['vertices'][frame+1].astype(float)[tri[observed[ii]]])
            tn=states['R_wc'][frame+1].astype(float);cn=states['C_world'][frame+1].astype(float);pq,zq=_project(q,tn,cn,intr)
            front=np.isfinite(pq).all(1)&np.isfinite(zq)&(zq>1e-5);inside=front&(pq[:,0]>=0)&(pq[:,0]<intr['width'])&(pq[:,1]>=0)&(pq[:,1]<intr['height'])
            target_depth=np.full(len(ii),np.inf)
            if inside.any():target_depth[inside]=rays(states['vertices'][frame+1].astype(float),tri,cn,_directions(pq[inside],intr,np.linalg.inv(tn.T)))[1]
            tol=1e-5+1e-5*np.abs(zq);v=inside&np.isfinite(target_depth)&(np.abs(zq-target_depth)<=tol);o=inside&np.isfinite(target_depth)&(zq>target_depth+tol)
            expected=dict(visible=v,occluded=o,out_of_frame=front&~inside,behind_camera=~front,unresolved=inside&~v&~o,flow_valid=front)
            for k,value in expected.items():visibility_bad[ii]|=flow[k][rr[ii],cc[ii]]!=value
            flow_error[ii[front]]=np.linalg.norm(flow['flow_uv'][rr[ii[front]],cc[ii[front]]].astype(float)-(pq[front]-pixels[ii[front]]),axis=1)
        texture_errors={};fg=observed>=0;mat_uv=material_uv(uv,tri,observed[fg],d['material_barycentric'][rr[fg],cc[fg]])
        noise_cfg=f['generator_config']['photometric'];noise_rng=np.random.default_rng(f['generator_config']['seed']+100000+(0 if stream=='reference' else 10000 if stream=='moving' else 20000)+frame)
        gain=noise_rng.uniform(1-noise_cfg['gain_jitter'],1+noise_cfg['gain_jitter']);noise=noise_rng.normal(0,noise_cfg['gaussian_sigma'],size=valid.shape+(3,)).astype(np.float32)[rr,cc]
        for name in ['procedural']+list(external):
            clean=np.full((len(selected),3),12,np.uint8)
            if name=='procedural':
                # Producer material UV arithmetic has the stored UV dtype.
                q=uv[tri[observed[fg]]].copy();u=q[:,:,0]%1;q[:,:,0]=u+((np.ptp(u,axis=1)>.5)[:,None]&(u<.5))
                puv=np.sum(d['material_barycentric'][rr[fg],cc[fg],:,None]*q,axis=1);puv[:,0]%=1;h,w=tex.shape[:2]
                clean[fg]=tex[np.clip(((1-puv[:,1])*(h-1)).astype(int),0,h-1),np.clip((puv[:,0]*(w-1)).astype(int),0,w-1)]
            else:clean[fg]=external_sample(external[name],mat_uv,f['file_textures'][name])
            expected=np.clip(clean.astype(np.float32)*gain+noise,0,255).astype(np.uint8)
            image_path=public/('rgb' if name=='procedural' else 'rgb_'+name)/f'{frame:06d}.png';image=cv2.imread(str(image_path))
            if image is None:raise FileNotFoundError(image_path)
            inputs[str(image_path.relative_to(root))]=sha(image_path);texture_errors[name]=np.max(np.abs(expected.astype(int)-image[rr,cc].astype(int)),axis=1)
        failed=mismatch|visibility_bad|(flow_error>1e-3)
        for error in texture_errors.values():failed|=error>1
        rows={}
        for k,sample in samples.items():
            index=np.searchsorted(selected,sample)
            rows[k]={'population':int(strata[k].sum()),'checked':len(index),'failures':int(failed[index].sum()),
                     'ray_depth_relative_max':float(ray_error[index].max()) if len(index) else None,
                     'visibility_mismatch':int(visibility_bad[index].sum()),'flow_error_px_max':float(flow_error[index].max()) if len(index) else None}
        displacement=np.linalg.norm(points-np.einsum('ni,nij->nj',geometric_b,g['vertices'].astype(float)[tri[ids]]),axis=1)/Rnom
        record=dict(frame=frame,pixels=int(valid.size),valid=int(valid.sum()),reprojection_px_max=reproject,relative_z_max=depth,normal_vector_max=ne,weight_sum_max=weight_error,
            cap_pixels=int(cap[ids].sum()),strata=rows,edge_identity_ties=int(ties.sum()),unique_ray_samples=len(selected),
            texture_error_max={k:int(v.max()) for k,v in texture_errors.items()},
            motion_over_nominal_R=dict(mean=float(displacement.mean()),p95=float(np.percentile(displacement,95)),max=float(displacement.max())),
            depth=dict(min=float(z.min()),median=float(np.median(z)),max=float(z.max())),
            passed=bool(partition_ok and not failed.any() and reproject<=1e-3 and depth<=1e-5 and ne<=1e-5 and weight_error<=1e-5))
        if failed.any():record['failed_sample_pixels']=selected[failed].tolist()
        results.append(record)
        if frame%15==0:print(json.dumps({'frame':frame,'passed':record['passed'],'source':str(truth)}),flush=True)
    report=dict(schema='collection_labels_audit_v1',family=family,condition=condition,stream=stream,source_sha256=sha(__file__),
        material_weight_policy=policy,flow_root=str(flow_root) if flow_root else None,
        python=sys.version,platform=platform.platform(),inputs_sha256=inputs,frames=results,passed=all(x['passed'] for x in results),
        coverage='all-pixel algebra and visibility partition; up to 64 independent rays per present overlapping stratum per frame; zero-sized strata unexercised')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:handle.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','family','condition','stream','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--flow-root')
    p.add_argument('--material-weight-policy',choices=['stored','unit_sum'])
    r=evaluate(**vars(p.parse_args()));sys.exit(0 if r['passed'] else 1)
