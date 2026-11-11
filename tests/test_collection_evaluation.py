"""Analytic evaluator integration fixtures, not benchmark validation evidence."""
import json,hashlib
import numpy as np
import pytest
from backend.src.monocular_evaluate import fit_similarity,transform,evaluate as monocular
from backend.src.appearance_evaluate import evaluate as appearance


def save(path,**data):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**data)


def js(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def scene(tmp_path):
    vertices=np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
    centers=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    states=vertices[None]+np.arange(4)[:,None,None]*np.array([.1,.2,0.])
    rotations=np.repeat(np.eye(3)[None],4,axis=0)
    dense=dict(valid=np.array([[True,True]]),triangle_id=np.array([[0,0]]),
        material_barycentric=np.array([[[1.,0.,0.],[0.,1.,0.]]]),depth_z=np.array([[2.,2.]]))
    frames={'frames':[{'image':f'{i:06d}.png'} for i in range(4)]}
    intr=dict(width=2,height=1,fx=1.,fy=1.,cx=.5,cy=.5)
    p=tmp_path/'data/baseline/v1';js(p/'DATASET.json',dict(condition='fem',family='analytic'))
    js(p/'public/intrinsics.json',intr);js(p/'public/moving/frames.json',frames)
    save(p/'evaluation/family/geometry.npz',vertices=vertices,triangles=np.array([[0,1,2]]))
    save(p/'evaluation/moving/states.npz',vertices=states,C_world=centers,R_wc=rotations)
    save(p/'evaluation/moving/dense/000000.npz',**dense)
    a=tmp_path/'data/appearance/f01';js(a/'public/intrinsics.json',intr)
    js(a/'public/frames.json',dict(**frames,conditions={'diffuse_symmetric':{}}))
    save(a/'evaluation/geometry.npz',triangles=np.array([[0,1,2]]))
    save(a/'evaluation/states.npz',vertices_meter=states,C_world_meter=centers,R_wc=rotations)
    save(a/'evaluation/dense/000000.npz',**dense)
    return p,a,states,centers,rotations


def test_global_sim3_transforms_all_modalities():
    source=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,1]])
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);target=3*source@R.T+[2,3,4]
    scale,rot,t=fit_similarity(source,target)
    out=transform(scale,rot,t,centers=source,points=source,rotations=np.eye(3),normals=[[1.,0,0]],displacements=[[1.,0,0]])
    assert np.allclose(out['points'],target) and np.allclose(out['rotations'],R)
    assert np.allclose(out['normals'],[[0,1,0]]) and np.allclose(out['displacements'],[[0,3,0]])
    with pytest.raises(ValueError,match='Degenerate'):fit_similarity(source[:2].repeat(2,axis=0),target[:2].repeat(2,axis=0))
    with pytest.raises(ValueError):fit_similarity(source,target,ratio=0)


def test_monocular_exact_gauge_and_missing_coverage(scene,tmp_path):
    p,a,states,C,R=scene
    pred=tmp_path/'prediction.npz';protocol=tmp_path/'protocol.json'
    q=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);t=np.array([1,2,3])
    mask=np.ones((4,2),bool);mask[-1,-1]=False
    save(pred,frame_ids=np.arange(4),centers=(C-t)@q/2,rotations=q.T@R,
         points=(states[:,:2]-t)@q/2,reference_pixels=np.array([[0,0],[0,1]]),mask=mask,
         normals=np.broadcast_to(np.array([0.,0,1])@q,(4,2,3)))
    js(protocol,dict(schema='material_tracks_alignment_v1',component_id='whole',frame_ids=[0,1,2,3],anchor_frame_ids=[0,1,2],reference_frame=0,reference_pixels_sha256=hashlib.sha256(np.array([[0,0],[0,1]],dtype='<i8').tobytes()).hexdigest()))
    result=monocular(p,pred,protocol,tmp_path/'result.json')
    assert result['status']=='aligned' and np.isclose(result['scale'],2)
    assert result['camera_position']['conditional_max']<1e-12
    assert result['material_position']['conditional_max']<1e-12
    assert result['material_position']['coverage']==7/8
    assert result['material_position']['full_population_mean'] is None
    assert result['material_displacement']['conditional_max']<1e-12
    assert result['normal_degrees']['conditional_max']<1e-6
    with pytest.raises(ValueError,match='immutable'):monocular(p,pred,protocol,p/'bad.json')


def test_missing_anchor_is_retained_failure(scene,tmp_path):
    p,a,states,C,R=scene;pred=tmp_path/'pred.npz';protocol=tmp_path/'protocol.json'
    save(pred,frame_ids=np.arange(4),centers=C,rotations=R,points=states[:,:2],reference_pixels=np.array([[0,0],[0,1]]))
    js(protocol,dict(schema='material_tracks_alignment_v1',component_id='whole',frame_ids=[0,1,2,3],anchor_frame_ids=[0,1,99],reference_frame=0,reference_pixels_sha256=hashlib.sha256(np.array([[0,0],[0,1]],dtype='<i8').tobytes()).hexdigest()))
    result=monocular(p,pred,protocol,tmp_path/'fail.json')
    assert result['status']=='alignment_failed' and result['material_position']['coverage']==0


@pytest.mark.parametrize('task,pred,target,frame',[('depth',[[2.,2.]],None,None),
    ('normals',[[[0.,0,1],[0,0,1]]],None,'appearance-world-meters'),
    ('displacement',[[[.1,.2,0],[.1,.2,0]]],1,'appearance-world-meters')])
def test_appearance_dense_metrics(scene,tmp_path,task,pred,target,frame):
    p,a,*_=scene;prediction=tmp_path/'pred.npz';save(prediction,prediction=np.array(pred))
    r=appearance(a,prediction,tmp_path/'r.json',task,0,target,frame)
    assert r['primary']['full_population_error']['mean']<1e-12
    assert r['visibility']=='not classified'


def test_appearance_flow_has_no_occlusion_claim(scene,tmp_path):
    p,a,*_=scene;prediction=tmp_path/'pred.npz';save(prediction,prediction=np.zeros((1,2,2)))
    r=appearance(a,prediction,tmp_path/'r.json','projected_flow',0,1)
    assert r['source_valid_pixels']==2 and r['primary']['eligible_pixels']==2
    assert r['visibility']=='not classified'


def test_explicit_rgb_keeps_historical_channels_and_index(scene,tmp_path):
    import cv2
    from backend.src.benchmark_reader import PublicDataset
    from backend.src.appearance_reader import AppearancePublic
    from backend.src.collection_reader import Collection
    p,a,*_=scene;bgr=np.array([[[1,2,3],[4,5,6]]],np.uint8)
    for q in [p/'public/moving/rgb/000000.png',p/'public/moving/rgb_tissue/000000.png',a/'public/diffuse_symmetric/rgb/000000.png']:
        q.parent.mkdir(exist_ok=True,parents=True);cv2.imwrite(str(q),bgr)
    desc=json.loads((p/'DATASET.json').read_text());desc['texture_variants']=['procedural','tissue'];js(p/'DATASET.json',desc)
    reader=PublicDataset(p);assert np.array_equal(reader.rgb(0,'moving'),bgr)
    assert np.array_equal(reader.image_rgb(0,'moving','tissue'),bgr[:,:,::-1])
    assert np.array_equal(AppearancePublic(a).rgb(0),bgr[:,:,::-1])
    root=tmp_path/'data'
    js(root/'COLLECTION.json',dict(schema='tubular_collection_v2',streams=[dict(id='f07_fem_moving_tissue',family='f07',motion='fem',camera='moving',appearance='tissue',pipeline='raster',path='baseline/v1',acquisition='moving')]))
    collection=Collection(root)
    assert len(collection.select(family='f07'))==1 and not collection.select(family='f08')
    assert np.array_equal(collection.image_rgb('f07_fem_moving_tissue',0),bgr[:,:,::-1])
