import numpy as np
import pytest
from backend.src.p2_state_sequence import P2RenderMapping

def fixture():
    X=np.array([[0,0,0],[1,0,0],[0,1,0.]])
    # Deliberately noninterleaved, permuted global component layout.
    I=np.arange(30).reshape(3,10).T[:,[2,0,1]]
    W=np.zeros((1,3,10));W[0,0,0]=W[0,1,1]=W[0,2,2]=1
    return P2RenderMapping(X,np.array([[0,1,2]]),W,np.arange(10)[None],I)

def test_explicit_layout_and_planar_composition():
    m=fixture();u=np.arange(30).reshape(10,3)*.001;q=np.empty(30);q[m.scalar_to_vector_dofs]=u/.015
    np.testing.assert_allclose(m.scalar_displacements_meter(q,.015),u,rtol=0,atol=1e-17)
    v=m.vertices_meter(q,.015);b=np.array([[.2,.3,.5],[1,0,0.]])
    np.testing.assert_allclose(m.material_points_meter(np.array([0,0]),b,q,.015),b@v,rtol=0,atol=4*np.finfo(float).eps)
    np.testing.assert_array_equal(m.vertices_meter(np.zeros(30),.015),m.reference_vertices_meter)

def test_not_curved_quadratic_evaluation():
    m=fixture();q=np.zeros(30);q[m.scalar_to_vector_dofs[0,2]]=1
    # Linear interpolation of nodal values gives1/3; curved quadratic vertex
    # shape function at barycentric1/3 gives-1/9. The API promises the former.
    p=m.material_points_meter(np.array([0]),np.array([[1/3]*3]),q,1.)
    assert p[0,2]==pytest.approx(1/3) and p[0,2]!=pytest.approx(-1/9)

@pytest.mark.parametrize('bad', [np.zeros(31),np.full(30,np.nan)])
def test_reject_state_layout_or_nonfinite(bad):
    with pytest.raises(ValueError):fixture().vertices_meter(bad,.015)

def test_invalid_component_map_and_cross_facet_triangle():
    m=fixture()
    with pytest.raises(ValueError):P2RenderMapping(m.reference_vertices_meter,m.triangles,m.weights,m.facet_scalar_dofs,np.zeros((10,3),int))
    with pytest.raises(ValueError):P2RenderMapping(np.tile(m.reference_vertices_meter,(2,1)),np.array([[0,1,3]]),np.tile(m.weights,(2,1,1)),np.tile(m.facet_scalar_dofs,(2,1)),m.scalar_to_vector_dofs)

def test_invalid_scale_barycentric_and_overflow():
    m=fixture()
    with pytest.raises(ValueError):m.vertices_meter(np.ones(30),0)
    with pytest.raises(ValueError):m.vertices_meter(np.full(30,1e308),1e308)
    with pytest.raises(ValueError):m.composed_material_weights(np.array([0]),np.array([[1,1,-1]]))
