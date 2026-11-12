import json,sys,time,hashlib
from pathlib import Path
import numpy as np
from ray_backend import Rays
sys.path.insert(0,sys.argv[1])
from backend.src.benchmark_validation import _ray_hits
rng=np.random.default_rng(6111201);v=rng.normal(size=(240,3));v[:,2]+=4;tri=np.arange(240).reshape(-1,3);o=np.zeros(3);dirs=np.c_[rng.uniform(-1,1,(512,2)),np.ones(512)]
# Deliberately include rays through vertices and shared-edge midpoints.
dirs=np.vstack((dirs,v[:20],(v[:20]+v[1:21])/2));r=Rays(v,tri);a=r(o,dirs);b=_ray_hits(v,tri,o,dirs);r.close()
assert np.array_equal(a[0],b[0]);ok=a[0]>=0;assert np.allclose(a[1][ok],b[1][ok],atol=1e-12,rtol=1e-12);assert np.allclose(a[2][ok],b[2][ok],atol=1e-12,rtol=1e-12)
# Exact two-sided planar hit, nearest surface, background and near-plane exclusion.
v=np.array([[-2,-2,3],[2,-2,3],[0,2,3],[-2,-2,5],[2,-2,5],[0,2,5]],float);tri=np.array([[0,1,2],[3,4,5]]);r=Rays(v,tri);ids,z,w=r(o,np.array([[0,0,1],[3,0,1],[0,0,-1]],float));r.close();assert ids.tolist()==[0,-1,-1] and z[0]==3 and np.allclose(w[0],[.25,.25,.5])
print(json.dumps({'passed':True,'random_edge_vertex_rays':len(dirs),'analytic_rays':3,'max_depth_difference':float(np.max(np.abs(a[1][ok]-b[1][ok]))),'source_sha256':hashlib.sha256(Path(__file__).with_name('native_rays.cpp').read_bytes()).hexdigest()}))
