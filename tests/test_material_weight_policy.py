"""Analytic storage-roundoff regression, separate from dataset validation."""
import json
import numpy as np
import pytest
from backend.src.material_weights import decode_weights
from backend.src.benchmark_reader import EvaluationDataset
from backend.src.dense_truth import flow_to_frame
from backend.src.types import CameraIntrinsics


def test_affine_partition_restores_translation_invariance_and_face_membership():
    weights=np.array([[.1,.2,.7]],dtype=np.float32)
    face=np.array([[0.,0.,2.],[2.,0.,2.],[0.,2.,2.]])
    offset=np.array([30.,-80.,100.])
    raw=decode_weights(weights)
    assert raw.sum()!=1.
    assert np.array_equal(raw,weights.astype(float))
    decoded=decode_weights(weights,'unit_sum')
    np.testing.assert_allclose(decoded@(face+offset)-offset,decoded@face,atol=2e-14,rtol=0)
    np.testing.assert_allclose((decoded@face)[0,2],2.,atol=1e-15,rtol=0)
    assert abs((raw@(face+offset)-offset)[0,2]-2.)>1e-7
    with pytest.raises(ValueError):decode_weights(weights,'typo')
    with pytest.raises(ValueError):decode_weights([[0.,0.,0.]],'unit_sum')


def test_float32_roundoff_flow_is_translation_invariant_with_explicit_policy():
    weights=np.array([[[.1,.2,.7]]],np.float32)
    truth={'valid':np.ones((1,1),bool),'triangle_id':np.zeros((1,1),int),'material_barycentric':weights}
    face=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]])
    intr=CameraIntrinsics(width=1,height=1,fx=.2,fy=.2,cx=.5,cy=.5)
    offset=np.array([10000.,-80000.,30000.])
    def run(v,c,**kw):return flow_to_frame(truth,v,[[0,1,2]],intr,np.eye(3),c,high_precision=True,**kw)
    a=run(face,np.zeros(3),material_weight_policy='unit_sum')
    b=run(face+offset,offset,material_weight_policy='unit_sum')
    np.testing.assert_allclose(a['flow_uv'],b['flow_uv'],atol=1e-8,rtol=0)
    assert a['visible'].all() and b['visible'].all()
    old=run(face+offset,offset)
    assert old['unresolved'].all()


def test_materialized_reference_and_legacy_reference_paths(tmp_path):
    package=tmp_path/'v1';package.mkdir();(package/'public').mkdir()
    (package/'DATASET.json').write_text('{}')
    (package/'public/intrinsics.json').write_text(json.dumps(dict(width=1,height=1,fx=1.,fy=1.,cx=.5,cy=.5)))
    reader=EvaluationDataset(package)
    assert reader.truth_path('reference')==package/'evaluation/reference_truth'
    reference=package/'evaluation/reference';reference.mkdir(parents=True)
    (reference/'CONVENTIONS.json').write_text(json.dumps(dict(material_weight_policy='unit_sum')))
    assert reader.truth_path('reference')==reference
    weights=np.array([[.1,.2,.7]],np.float32)
    np.testing.assert_array_equal(reader.material_weights(weights,'reference'),decode_weights(weights,'unit_sum'))
    np.testing.assert_array_equal(reader.material_weights(weights,'moving'),weights.astype(float))
