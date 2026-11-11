import json
from pathlib import Path
import numpy as np
import pytest
from backend.src.collection_geometry import coefficients,geometry,poses
from backend.src.collection_mechanical_checks import affine_checks
from backend.src.config import load_config_dict
from backend.src.file_texture import sample,load_texture
from backend.src.benchmark import waveform

SPEC=Path(__file__).parents[1]/'configs/tubular_collection.json'

def test_analytic_mechanics():
    assert affine_checks()['passed']

@pytest.mark.parametrize('index',[0,1])
def test_configured_geometry_and_time(index):
    f=json.loads(SPEC.read_text())['families'][index];cfg=load_config_dict(f['generator_config'])
    g=geometry(cfg.tube,cfg.difficulty_presets['colon_benchmark'],np.random.default_rng(cfg.seed),f['collection_geometry'])
    w=coefficients(f);p=poses(g,f,cfg)
    assert w.shape==(f['action_frames'],len(f['patches']))
    assert np.all(w[0]==0) and np.max(np.abs(w[-1]))<1e-20
    assert np.linalg.matrix_rank(w)>3
    assert len(g.centerline)>112 and cfg.tube.length>150
    assert np.allclose(p['stationary_C'],p['reference_C'][-1])
    assert np.allclose(np.linalg.det(p['moving_R']),1,atol=1e-6)

def test_periodic_texture_and_hash():
    f=json.loads(SPEC.read_text())['families'][0];s=f['file_textures']['tissue'];im=load_texture(SPEC.parent,s)
    uv=np.array([[0,.3],[1,.3],[-1,.3],[.2,.4]])
    out=sample(im,uv,s);assert np.array_equal(out[0],out[1]) and np.array_equal(out[0],out[2])
    with pytest.raises(ValueError):load_texture(SPEC.parent,dict(s,sha256='0'*64))

def test_legacy_schedule_still_frozen():
    w=waveform();assert w.shape==(90,3) and np.all(w[60:75]==1)
    with pytest.raises(ValueError):waveform(240)

@pytest.mark.parametrize('other,expected',[
    ([[.2,.2,-1],[.2,.2,1],[.8,.2,0]],1),
    ([[.1,.1,0],[.5,.1,0],[.1,.5,0]],1),
    ([[2,0,0],[3,0,0],[2,1,0]],0),
    ([[0,0,1],[1,0,1],[0,1,1]],0),
])
def test_triangle_intersection_cases(other,expected):
    from backend.src.collection_preflight import triangle_crossings
    p=np.array([[[0,0,0],[1,0,0],[0,1,0]],other],float)
    assert triangle_crossings(p,np.array([[0,1]]))==expected

def test_independent_external_sampling_matches_linear_periodic_contract():
    from backend.src.collection_labels import external_sample
    rng=np.random.default_rng(5);im=rng.integers(0,256,(4,5,3),dtype=np.uint8)
    uv=rng.uniform(-2,2,(100,2))
    for interpolation in ['nearest','bilinear']:
        for orientation in [0,90,180,270]:
            spec=dict(interpolation=interpolation,orientation=orientation,repeat=[3,7],offset=[.1,-.4])
            assert np.array_equal(external_sample(im,uv,spec),sample(np.rot90(im,orientation//90),uv,spec))


def test_accelerated_independent_rays_against_analytic_geometry():
    pytest.importorskip('trimesh');pytest.importorskip('rtree')
    from backend.src.collection_labels import rays
    v=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.],[-1.,-1.,4.],[1.,-1.,4.],[0.,1.,4.]])
    ids,z,b=rays(v,np.array([[0,1,2],[3,4,5]]),np.zeros(3),np.array([[0.,0.,1.],[3.,0.,1.]]))
    assert ids.tolist()==[0,-1] and z[0]==2 and np.isinf(z[1])
    assert np.allclose(b[0],[.25,.25,.5])
