"""Independent FE-to-render transfer and temporal load/energy checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .benchmark import sha


def evaluate(root,family,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    spec=json.loads((root/'SPECIFICATION.json').read_text());f=next(x for x in spec['families'] if x['id']==family)
    p=root/'families'/family/'evaluation'
    with np.load(p/'geometry.npz') as z:g=dict(z)
    with np.load(p/'fields.npz') as z:fields=dict(z)
    with np.load(p/'mechanics.npz') as z:m=dict(z)
    fe=f['generator_config']['tube']['deformation']['fea'];na,nt=fe['axial_elements'],fe['theta_elements'];R=f['generator_config']['tube']['dimensions']['base_radius']
    s=g['s_norm'].astype(float)*na;u=g['theta'].astype(float)/(2*np.pi)*nt
    i=np.minimum(np.floor(s).astype(int),na-1);j=np.floor(u).astype(int)%nt;a=s-i;b=u-np.floor(u)
    grid=m['basis_displacement'].reshape(-1,na+1,nt,2,3)[:,:,:,1,:]
    expected=np.empty_like(fields['fem_basis'],dtype=float)
    for k in range(len(grid)):
        for si in range(len(s)):
            expected[k,si]=(1-a[si])*((1-b)[:,None]*grid[k,i[si],j]+b[:,None]*grid[k,i[si],(j+1)%nt])+a[si]*((1-b)[:,None]*grid[k,i[si]+1,j]+b[:,None]*grid[k,i[si]+1,(j+1)%nt])
    error=float(np.max(np.linalg.norm(expected-fields['fem_basis'],axis=-1))/R)
    K=csr_matrix((m['K_data'],m['K_indices'],m['K_indptr']),shape=m['K_shape']);U=m['basis_displacement'].reshape(len(grid),-1);F=m['basis_loads'].reshape(len(grid),-1);Q=m['basis_reactions'].reshape(len(grid),-1);free=~m['fixed'].ravel();gram=U@(K@U.T)
    rows=[]
    for k,w in enumerate(fields['time_coefficients']):
        disp=w@U;load=w@F;reaction=w@Q;residual=K@disp-load;scale=max(np.linalg.norm(load),1e-30)
        energy=.5*disp@(K@disp);cross=.5*w@gram@w;work=.5*disp@load
        rows.append(dict(frame=k,free_equilibrium=float(np.linalg.norm(residual[free])/scale),reaction=float(np.linalg.norm(residual-reaction)/scale),energy_cross_relative=float(abs(energy-cross)/max(abs(energy),1e-20)),work_relative=float(abs(energy-work)/max(abs(energy),1e-20))))
    passed=error<=1e-6 and all(max(x['free_equilibrium'],x['reaction'],x['work_relative'])<=1e-7 and x['energy_cross_relative']<=1e-10 for x in rows)
    result=dict(family=family,passed=passed,transfer_max_over_nominal_R=error,transfer_tolerance_over_R=1e-6,
        source_sha256=sha(__file__),inputs_sha256={str(q.relative_to(root)):sha(q) for q in [p/'geometry.npz',p/'fields.npz',p/'mechanics.npz',root/'SPECIFICATION.json']},states=rows,
        geometry_interpretation='Observation surface attached parametrically to outer FE-node envelope; auxiliary mechanical shell extends inward by assigned thickness; not an anatomical lumen-wall model.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as out:out.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','family','output']:p.add_argument('--'+k,required=True)
    sys.exit(0 if evaluate(**vars(p.parse_args()))['passed'] else 1)
