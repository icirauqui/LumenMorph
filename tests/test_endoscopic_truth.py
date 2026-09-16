import numpy as np
import pytest
pytest.importorskip('mitsuba')
from backend.src.camera_models import PinholeCamera, OpenCVFisheyeCamera
from backend.src.endoscopic_truth import material_flow
from backend.src.endoscopic_renderer import mesh_topology_sha256


@pytest.mark.parametrize('camera_type',[PinholeCamera,OpenCVFisheyeCamera])
def test_material_transport_with_continuous_subpixel_occlusion(camera_type):
    # Source identity at a target position projected strictly between pixel centers.
    camera=camera_type(8,8,4.,4.,4.,4.)
    vertices=np.array([[-2,-2,2],[2,-2,2],[0,2,2],
                       [.1,-.02,1],[.14,-.02,1],[.12,.02,1]],dtype=float)
    faces=np.array([[0,1,2],[3,4,5]])
    # Target point(.24,0,2), occulted only at the actual fractional endpoint.
    bary=np.array([.19,.31,.5])
    source={'valid':np.array([[True]]),'triangle_id':np.array([[0]]),'material_barycentric':bary.reshape(1,1,3)}
    source['mesh_topology_sha256']=mesh_topology_sha256(len(vertices),faces)
    answer=material_flow(source,vertices,faces,camera,np.eye(3),np.zeros(3),threads=1)
    uv,_=camera.project(np.array([[.24,0,2]]))
    np.testing.assert_allclose(answer['flow_uv'][0,0],uv[0]-.5,atol=1e-7)
    assert answer['occluded'][0,0] and not answer['visible'][0,0]
    vertices[3:,0]+=1
    clear=material_flow(source,vertices,faces,camera,np.eye(3),np.zeros(3),threads=1)
    assert clear['visible'][0,0] and not clear['occluded'][0,0]


def test_optical_domain_is_not_mislabeled_behind_camera():
    camera=OpenCVFisheyeCamera(8,8,4,4,4,4,max_theta=.1)
    vertices=np.array([[1.,0,1],[2.,0,1],[1.,1,1]])
    source={'valid':np.array([[True,False]]),'triangle_id':np.array([[0,-1]]),
            'material_barycentric':np.array([[[1.,0,0],[np.nan]*3]])}
    source['mesh_topology_sha256']=mesh_topology_sha256(len(vertices),np.array([[0,1,2]]))
    answer=material_flow(source,vertices,np.array([[0,1,2]]),camera,np.eye(3),np.zeros(3),threads=1)
    assert answer['outside_optical_domain'][0,0] and not answer['behind_camera'].any()
    assert not answer['flow_valid'].any() and np.isnan(answer['flow_uv']).all()


def test_invalid_material_identity_rejected():
    source={'valid':np.array([[True]]),'triangle_id':np.array([[-1]]),
            'material_barycentric':np.array([[[1.,0,0]]])}
    source['mesh_topology_sha256']=mesh_topology_sha256(3,np.array([[0,1,2]]))
    with pytest.raises(ValueError,match='identities'):
        material_flow(source,np.array([[0,0,1],[1,0,1],[0,1,1]]),np.array([[0,1,2]]),
                      PinholeCamera(1,1,1,1,.5,.5),np.eye(3),np.zeros(3))


def test_changed_topology_and_collapsed_target_are_rejected():
    v=np.array([[0.,0,1],[1.,0,1],[0.,1,1],[1.,1,1]])
    f=np.array([[0,1,2],[1,3,2]])
    source={'valid':np.array([[True]]),'triangle_id':np.array([[0]]),
            'material_barycentric':np.array([[[1.,0,0]]]),
            'mesh_topology_sha256':mesh_topology_sha256(len(v),f)}
    c=PinholeCamera(1,1,1,1,.5,.5)
    with pytest.raises(ValueError,match='topology'):
        material_flow(source,v,f[::-1],c,np.eye(3),np.zeros(3))
    with pytest.raises(ValueError,match='topology'):
        material_flow({k:x for k,x in source.items() if k!='mesh_topology_sha256'},v,f,c,np.eye(3),np.zeros(3))
    v[1]=v[0]
    with pytest.raises(ValueError,match='Degenerate'):
        material_flow(source,v,f,c,np.eye(3),np.zeros(3))


def test_sensor_mask_is_separate_from_occlusion_and_source_geometric_support():
    v=np.array([[-1.,-1.,2.],[1.,-1.,2.],[1.,1.,2.],[-1.,1.,2.]])
    f=np.array([[0,1,2],[0,2,3]])
    source={'valid':np.array([[True,True]]),'observation_valid':np.array([[True,False]]),
            'triangle_id':np.array([[0,0]]),'material_barycentric':np.array([[[.5,0,.5],[.5,0,.5]]]),
            'mesh_topology_sha256':mesh_topology_sha256(len(v),f)}
    camera=PinholeCamera(4,4,4,4,2,2);mask=np.ones((4,4),bool);mask[2,2]=False
    answer=material_flow(source,v,f,camera,np.eye(3),np.zeros(3),target_sensor_mask=mask)
    assert answer['flow_valid'][0,0] and answer['masked_by_sensor'][0,0]
    assert not answer['visible'].any() and not answer['occluded'].any()
    assert np.isnan(answer['flow_uv'][0,1]).all()
