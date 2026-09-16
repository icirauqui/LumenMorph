import numpy as np
import pytest
from backend.src.material_appearance import MaterialAppearanceSettings,material_vertex_albedo


def sheet():
    x,y=np.meshgrid(np.linspace(0,.04,21),np.linspace(0,.04,21));v=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    f=[]
    for j in range(20):
        for i in range(20):
            a=21*j+i;f.extend([[a,a+1,a+22],[a,a+22,a+21]])
    return v,np.asarray(f)


def test_reference_field_is_reproducible_bounded_and_has_surface_network():
    v,f=sheet();settings=MaterialAppearanceSettings(network_roots=3,network_width_meter=.002)
    a,n,meta=material_vertex_albedo(v,f,settings);b,m,_=material_vertex_albedo(v,f,settings)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(n,m)
    assert a.min()>0 and a.max()<1 and n.max()==1 and n.min()<.001
    assert meta['network_path_vertex_count']>3 and np.ptp(a[:,0])>.05
    shifted,shifted_network,_=material_vertex_albedo(v+[2.,3.,4.],f,settings)
    np.testing.assert_allclose(a,shifted,atol=1e-6)
    # Spatial coordinates are centered; rigid translation cannot change appearance.
    np.testing.assert_array_equal(n,shifted_network)


def test_disconnected_nearby_surface_does_not_receive_network_by_euclidean_distance():
    v,f=sheet();vv=np.r_[v,v+[0,0,1e-6]];ff=np.r_[f,f+len(v)]
    _,network,meta=material_vertex_albedo(vv,ff,MaterialAppearanceSettings(network_roots=1,network_width_meter=.002))
    root=meta['network_root_material_ids'][0];opposite=network[len(v):] if root<len(v) else network[:len(v)]
    assert not opposite.any()


def test_invalid_reflectance_and_geometry_rejected():
    with pytest.raises(ValueError):MaterialAppearanceSettings(base_albedo_rgb=(1,1,1))
    v,f=sheet();v[f[0,1]]=v[f[0,0]]
    with pytest.raises(ValueError,match='Degenerate'):material_vertex_albedo(v,f)
