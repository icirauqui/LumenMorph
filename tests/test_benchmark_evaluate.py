import json

import numpy as np
import pytest

from backend.src.benchmark_evaluate import evaluate, score_dense


def test_mask_cannot_cherry_pick_population():
    truth=np.zeros((1,2,2))
    prediction=np.array([[[0.,0.],[3.,4.]]])
    result=score_dense(prediction,truth,np.ones((1,2),bool),'flow',np.array([[True,False]]))
    assert result['eligible_pixels']==2 and result['valid_prediction_pixels']==1
    assert result['conditional_error']['mean']==0
    assert result['full_population_error']['mean'] is None
    assert not result['complete_prediction_support']


def test_vector_metrics_and_static_zero_are_well_defined():
    prediction=np.array([[[3.,4.,0.],[0.,0.,0.]]])
    result=score_dense(prediction,np.zeros_like(prediction),np.ones((1,2),bool),'displacement')
    assert result['full_population_error']['mean']==2.5
    assert result['full_population_error']['p95']==4.75
    assert result['complete_prediction_support']
    empty=score_dense(prediction,prediction,np.zeros((1,2),bool),'displacement')
    assert empty['prediction_coverage'] is None and not empty['complete_prediction_support']


def test_normals_direction_not_magnitude_and_zero_invalid():
    truth=np.array([[[0.,0.,1.],[0.,0.,1.],[0.,0.,1.]]])
    prediction=np.array([[[0.,0.,2.],[0.,0.,-1.],[0.,0.,0.]]])
    result=score_dense(prediction,truth,np.ones((1,3),bool),'normals')
    assert result['conditional_error']['mean']==90.
    assert result['valid_prediction_pixels']==2 and result['invalid_finite_pixels']==1
    assert result['full_population_error']['mean'] is None


def test_normals_are_stable_at_extreme_nonzero_scales():
    truth=np.array([[[0.,0.,1.],[0.,0.,1.]]])
    prediction=np.array([[[0.,0.,1e300],[0.,0.,1e-300]]])
    result=score_dense(prediction,truth,np.ones((1,2),bool),'normals')
    assert result['complete_prediction_support']
    assert result['full_population_error']['mean']==0.


def test_overflowed_metrics_rejected_explicitly():
    with pytest.raises(ValueError,match='floating-point range'):
        score_dense(np.full((1,1,3),1e300),np.zeros((1,1,3)),np.ones((1,1),bool),'displacement')


def test_depth_background_nonfinite_and_invalid_values():
    truth=np.array([[2.,3.,np.nan]])
    result=score_dense(np.array([[3.,-1.,np.nan]]),truth,np.array([[True,True,False]]),'depth')
    assert result['eligible_pixels']==2 and result['invalid_finite_pixels']==1
    assert result['conditional_error']['mean']==1.
    assert result['full_population_error']['mean'] is None


def test_malformed_mask_shape_and_nonfinite_eligible_truth():
    valid=np.ones((1,2),bool)
    with pytest.raises(ValueError):score_dense(np.zeros((1,2)),np.zeros((1,2)),valid,'flow')
    with pytest.raises(ValueError):score_dense(np.ones((1,2)),np.ones((1,2)),valid,'depth',np.ones((1,2)))
    with pytest.raises(ValueError):score_dense(np.ones((1,2)),np.array([[1.,np.nan]]),valid,'depth')


@pytest.fixture
def package(tmp_path):
    package=tmp_path/'suite'/'v3'
    (package/'public/moving').mkdir(parents=True)
    (package/'DATASET.json').write_text(json.dumps({'condition':'static','family':'f01','split':'development'}))
    (package/'public/intrinsics.json').write_text(json.dumps(dict(width=3,height=1,fx=1.,fy=1.,cx=0.,cy=0.)))
    (package/'public/moving/frames.json').write_text(json.dumps({'frames':[{},{}]}))
    private=package/'evaluation/moving'
    (private/'dense').mkdir(parents=True)
    (private/'flow_forward').mkdir()
    family=package/'evaluation/family';family.mkdir()
    vertices=np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
    np.savez(family/'geometry.npz',vertices=vertices,triangles=np.array([[0,1,2]]))
    np.savez(private/'states.npz',vertices=np.array([vertices,vertices+[1.,2.,2.]]))
    valid=np.array([[True,True,False]])
    np.savez(private/'dense/000000.npz',depth_z=np.array([[2.,2.,np.nan]]),valid=valid,
             normal_world=np.array([[[0.,0.,1.],[0.,0.,1.],[np.nan,np.nan,np.nan]]]),
             triangle_id=np.array([[0,0,-1]]),
             material_barycentric=np.array([[[1.,0.,0.],[0.,1.,0.],[np.nan,np.nan,np.nan]]]))
    false=np.zeros((1,3),bool)
    np.savez(private/'flow_forward/000000.npz',flow_uv=np.array([[[1.,0.],[2.,0.],[np.nan,np.nan]]]),
             flow_valid=valid,visible=np.array([[True,False,False]]),
             occluded=np.array([[False,True,False]]),out_of_frame=false,behind_camera=false,unresolved=false)
    return package


@pytest.mark.parametrize('task,values,target,frame_name,expected',[
    ('depth',[[3.,3.,0.]],None,None,1.),
    ('flow',[[[0.,0.],[0.,0.],[0.,0.]]],1,None,1.),
    ('normals',[[[0.,0.,1.],[0.,0.,-1.],[0.,0.,0.]]],None,'benchmark-world',90.),
    ('displacement',[[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]],1,'benchmark-world',3.),
])
def test_evaluator_uses_source_pixel_truth_without_population_selection(package,tmp_path,task,values,target,frame_name,expected):
    prediction=tmp_path/'prediction.npz';np.savez(prediction,prediction=np.asarray(values))
    result=evaluate(package,prediction,tmp_path/'result.json',task,0,'moving',target,frame_name)
    assert result['primary']['full_population_error']['mean']==expected
    assert result['source_valid_pixels']==2
    if task=='flow':
        assert result['primary']['eligible_pixels']==1
        assert result['secondary_strata']['occluded']['full_population_error']['mean']==2.


def test_evaluator_rejects_unknown_gauge_and_writes_inside_bank(package,tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,prediction=np.zeros((1,3,3)))
    with pytest.raises(ValueError,match='coordinate-frame'):evaluate(package,p,tmp_path/'a.json','displacement',0,target=1)
    with pytest.raises(ValueError,match='immutable'):evaluate(package,p,package/'a.json','depth',0)
    with pytest.raises(ValueError,match='target=source'):evaluate(package,p,tmp_path/'a.json','flow',0,target=0)
