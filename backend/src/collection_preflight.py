"""Statewise surface, camera and discrete-mechanics audit of a frozen family."""
import argparse,json,platform,sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .benchmark import sha
from .benchmark_validation import audit_c3d6_mechanics,audit_mesh_fields,_wedge_derivatives
from .collection_mechanical_checks import affine_checks


def crossing_pairs(reference,triangles,states):
    """Conservative swept AABB candidates; excludes pairs sharing a vertex."""
    from rtree import index
    lo=np.minimum(reference,states.min(0))[triangles].min(1);hi=np.maximum(reference,states.max(0))[triangles].max(1)
    prop=index.Property();prop.dimension=3
    tree=index.Index(((i,tuple(np.r_[a,b]),None) for i,(a,b) in enumerate(zip(lo,hi))),properties=prop)
    pairs=[]
    for i,(a,b) in enumerate(zip(lo,hi)):
        candidates=np.array([j for j in tree.intersection(tuple(np.r_[a,b])) if j>i],int)
        if len(candidates):
            shared=np.any(triangles[i][None,:,None]==triangles[candidates][:,None,:],axis=(1,2))
            pairs.extend((i,int(j)) for j in candidates[~shared])
    return np.asarray(pairs,int).reshape(-1,2)


def triangle_crossings(points,pairs,tolerance=1e-9):
    """Segment/triangle intersections, plus coplanar overlap via 2D SAT."""
    count=0
    for first in range(0,len(pairs),20000):
        pair=pairs[first:first+20000];a=points[pair[:,0]];b=points[pair[:,1]];hit=np.zeros(len(a),bool)
        for x,y in ((a,b),(b,a)):
            e1=y[:,1]-y[:,0];e2=y[:,2]-y[:,0]
            for k in range(3):
                o=x[:,k];d=x[:,(k+1)%3]-o;p=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,p);ok=np.abs(det)>tolerance
                inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);s=o-y[:,0];u=np.einsum('ij,ij->i',s,p)*inv;q=np.cross(s,e1);v=np.einsum('ij,ij->i',d,q)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
                hit |= ok&(u>=-tolerance)&(v>=-tolerance)&(u+v<=1+tolerance)&(t>=-tolerance)&(t<=1+tolerance)
        normal=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);length=np.linalg.norm(normal,axis=1)
        plane=np.abs(np.einsum('nvc,nc->nv',b-a[:,:1],normal))/np.maximum(length[:,None],1e-30)
        cop=np.flatnonzero((plane.max(1)<tolerance)&~hit)
        for i in cop:
            axes=np.delete(np.arange(3),np.argmax(np.abs(normal[i])));aa=a[i][:,axes];bb=b[i][:,axes];separate=False
            for poly in (aa,bb):
                for k in range(3):
                    edge=poly[(k+1)%3]-poly[k];axis=np.array([-edge[1],edge[0]]);pa=aa@axis;pb=bb@axis
                    if pa.max()<pb.min()-tolerance or pb.max()<pa.min()-tolerance:separate=True;break
                if separate:break
            hit[i]=not separate
        count+=int(hit.sum())
    return count


def evaluate(root,family,output):
    import trimesh
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    spec=json.loads((root/'SPECIFICATION.json').read_text());f=next(x for x in spec['families'] if x['id']==family);d=root/'families'/family/'evaluation'
    def read(name):
        with np.load(d/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    g,fields,fea,poses=map(read,['geometry.npz','fields.npz','mechanics.npz','poses.npz']);R=float(np.median(g['base_radius']));base=g['vertices'];tri=g['triangles'];weights=fields['time_coefficients']
    report={'family':family,'source_sha256':sha(__file__),'python':sys.version,'platform':platform.platform(),'input_sha256':{n:sha(d/n) for n in ['geometry.npz','fields.npz','mechanics.npz','poses.npz']},'spec_sha256':sha(root/'SPECIFICATION.json'),'manufactured':affine_checks()}
    report['mechanics']=audit_c3d6_mechanics(fea,coefficients=weights)
    # Scaled Jacobian uses the reference parametric derivative columns.
    x=fea['nodes'][fea['elements_c3d6']];scaled=[]
    for r,s in [(1/6,1/6),(2/3,1/6),(1/6,2/3)]:
        for z in [-1/np.sqrt(3),1/np.sqrt(3)]:
            J=np.einsum('eic,ij->ecj',x,_wedge_derivatives(r,s,z));scaled.append(np.linalg.det(J)/np.prod(np.linalg.norm(J,axis=1),axis=1))
    report['minimum_scaled_reference_jacobian']=float(np.min(scaled))
    K=csr_matrix((fea['K_data'],fea['K_indices'],fea['K_indptr']),shape=fea['K_shape']);U=fea['basis_displacement'].reshape(len(weights[0]),-1);gram=U@(K@U.T)
    errors=[]
    for w in weights:
        u=w@U;en=.5*u@(K@u);composed=.5*w@gram@w;errors.append(abs(en-composed)/max(abs(en),1e-20))
    report['max_temporal_energy_relative_error']=float(max(errors));report['surface']={}
    reference=trimesh.Trimesh(base,tri,process=False)
    cp,dist,_=trimesh.proximity.closest_point(reference,poses['reference_C']);report['reference_camera_min_distance_over_R']=float(dist.min()/R)
    report['reference_camera_inside']=bool(reference.contains(poses['reference_C']).all())
    for condition in ['fem','nonfem']:
        disp=np.einsum('tk,kijc->tijc',weights,fields[condition+'_basis']).reshape(len(weights),-1,3).astype(np.float32);states=(base[None]+disp).astype(np.float32)
        audit=audit_mesh_fields(base,tri,states,R);candidates=crossing_pairs(base,tri,states);cross=[];clear=[];inside=[]
        for i,state in enumerate(states):
            cross.append(triangle_crossings(state[tri].astype(float),candidates))
            mesh=trimesh.Trimesh(state,tri,process=False);cam=np.stack([poses['moving_C'][i],poses['stationary_C'][i]])
            _,dist,_=trimesh.proximity.closest_point(mesh,cam);clear.append((dist/R).tolist());inside.append(mesh.contains(cam).tolist())
            if i%30==0:print(json.dumps({'family':family,'condition':condition,'preflight_state':i,'candidate_pairs':len(candidates)}),flush=True)
        audit.update(nonadjacent_intersections=cross,swept_candidate_pairs=len(candidates),camera_clearance_over_R=clear,camera_inside=inside)
        audit['passed'] &= not any(cross) and np.min(clear)>=.25 and np.all(inside) and min(x['min_triangle_area_ratio'] for x in audit['frames'])>=.1
        report['surface'][condition]=audit
    report['passed']=bool(report['manufactured']['passed'] and report['mechanics']['passed'] and report['minimum_scaled_reference_jacobian']>.01 and max(errors)<=1e-10 and report['reference_camera_inside'] and report['reference_camera_min_distance_over_R']>=.25 and all(x['passed'] for x in report['surface'].values()))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--family',required=True);p.add_argument('--output',required=True);r=evaluate(**vars(p.parse_args()));print('PASS' if r['passed'] else 'FAIL');sys.exit(0 if r['passed'] else 1)
