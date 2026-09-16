import numpy as np
import pytest
from backend.src.anatomy_reference import (LumenMask,read_mha_lumen,binary_summary,
                                         lumen_surface,surface_summary,skeleton_path_diagnostics)


def write_mask(path,a,spacing=(2,3,4),origin=(10,20,30),direction='0 -1 0 1 0 0 0 0 1'):
    h=f'''ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = {direction}
Offset = {' '.join(map(str,origin))}
ElementSpacing = {' '.join(map(str,spacing))}
DimSize = {' '.join(map(str,a.shape[::-1]))}
ElementType = MET_UCHAR
ElementDataFile = LOCAL
'''
    path.write_bytes(h.encode()+a.tobytes())


def test_mha_affine_array_order_and_payload(tmp_path):
    a=np.zeros((7,8,9),dtype=np.uint8);a[1:5,2:6,3:7]=1
    p=tmp_path/'a.mha';write_mask(p,a)
    m=read_mha_lumen(p)
    np.testing.assert_array_equal(m.values_zyx,a)
    np.testing.assert_allclose(m.physical_xyz([1,2,3]),[4,26,34])
    q=binary_summary(m)
    assert q['volume_mm3']==64*24 and q['components_26']==1
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError,match='payload size'):read_mha_lumen(p)


def test_mask_rejects_guessed_scale_and_nonbinary(tmp_path):
    p=tmp_path/'a.mha';a=np.zeros((5,5,5),np.uint8)
    write_mask(p,a,spacing=(0,1,1))
    with pytest.raises(ValueError,match='spacing'):read_mha_lumen(p)
    a[2,2,2]=2;write_mask(p,a)
    with pytest.raises(ValueError,match='only background'):binary_summary(read_mha_lumen(p))


def test_surface_preserves_oblique_affine_and_is_closed(tmp_path):
    pytest.importorskip('skimage')
    a=np.zeros((12,12,12),np.uint8);a[3:9,3:9,3:9]=1
    p=tmp_path/'a.mha';write_mask(p,a)
    m=read_mha_lumen(p);v,f=lumen_surface(m)
    s=surface_summary(v,f)
    assert s['boundary_edges']==s['nonmanifold_edges']==s['zero_area_triangles']==0
    assert s['signed_volume_mm3']>0
    np.testing.assert_allclose(v.min(axis=0),[-15.5,25,40])
    np.testing.assert_allclose(v.max(axis=0),[2.5,37,64])
    assert abs(s['signed_volume_mm3']-216*24)/(216*24)<.05


def test_medial_diagnostic_reports_approximation_on_straight_tube():
    pytest.importorskip('skimage')
    z,y,x=np.indices((81,25,25));a=(((y-12)**2+(x-12)**2<=64)&(z>4)&(z<76)).astype(np.uint8)
    m=LumenMask(a,np.ones(3),np.zeros(3),np.eye(3),{})
    s,path,rad=skeleton_path_diagnostics(m,1.)
    assert not s['anatomical_centerline_validated']
    assert 50<s['path_length_mm']<72
    assert s['path_tortuosity']==pytest.approx(1,abs=.03)
    assert np.median(rad)==pytest.approx(np.sqrt(65),rel=.05)


def test_open_surface_cannot_report_enclosed_volume():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    s=surface_summary(v,np.array([[0,1,2]]))
    assert s['boundary_edges']==3 and s['enclosed_volume_mm3'] is None
    assert s['surface_components']==1
