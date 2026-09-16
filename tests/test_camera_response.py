import numpy as np
import pytest
from backend.src.camera_response import CameraResponseSettings,apply_camera_response


def test_automatic_gain_compensates_illumination_and_exposes_actual_state():
    image=np.full((10,10,3),.02);support=np.ones((10,10),dtype=bool)
    settings=CameraResponseSettings(exposure_mode='automatic',target_linear_luminance=.08)
    a,sa,ma=apply_camera_response(image,support,settings)
    b,sb,mb=apply_camera_response(image*2,support,settings)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(sa,sb)
    assert ma['exposure_gain']==pytest.approx(4.) and mb['exposure_gain']==pytest.approx(2.)
    assert np.all(image==.02)


def test_exposure_adaptation_obeys_elapsed_time_and_gain_bounds():
    image=np.full((4,4,3),.02);mask=np.ones((4,4),bool)
    settings=CameraResponseSettings(exposure_mode='automatic',target_linear_luminance=.08,exposure_time_constant_seconds=.2)
    _,_,zero=apply_camera_response(image,mask,settings,previous_log_gain=0.,elapsed_seconds=0.)
    _,_,half=apply_camera_response(image,mask,settings,previous_log_gain=0.,elapsed_seconds=.2*np.log(2))
    assert zero['exposure_gain']==1. and half['exposure_gain']==pytest.approx(2.)
    _,_,dark=apply_camera_response(image*0,mask,settings)
    assert dark['exposure_gain']==pytest.approx(settings.gain_limits[1])


def test_aperture_and_encoding_are_explicit_and_geometry_independent():
    polygon=((.2,.2),(.8,.2),(.8,.8),(.2,.8))
    settings=CameraResponseSettings(encoding='power_gamma',gamma=2.,aperture_polygon_normalized=polygon)
    rgb,support,meta=apply_camera_response(np.full((10,10,3),.25),np.ones((10,10),bool),settings)
    assert support.sum()==36 and np.all(rgb[support]==128) and not rgb[~support].any()
    assert meta['settings']['encoding']=='power_gamma'
    with pytest.raises(ValueError,match='convex'):
        CameraResponseSettings(aperture_polygon_normalized=((0,0),(1,1),(1,0),(0,1)))


def test_invalid_image_and_history_rejected():
    with pytest.raises(ValueError):apply_camera_response(np.full((3,3,3),np.nan),np.ones((3,3),bool))
    with pytest.raises(ValueError):apply_camera_response(np.ones((3,3,3)),np.ones((3,3),bool),previous_log_gain=np.inf)


def test_same_turn_self_intersecting_polygon_rejected_and_both_convex_windings_work():
    angles=np.linspace(0,2*np.pi,5,endpoint=False)
    polygon=.5+.45*np.c_[np.cos(angles),np.sin(angles)]
    with pytest.raises(ValueError,match='convex'):
        CameraResponseSettings(aperture_polygon_normalized=tuple(map(tuple,polygon[[0,2,4,1,3]])))
    image=np.ones((19,23,3));sensor=np.ones((19,23),bool)
    masks=[]
    for vertices in [polygon,polygon[::-1]]:
        _,mask,_=apply_camera_response(image,sensor,CameraResponseSettings(aperture_polygon_normalized=tuple(map(tuple,vertices))))
        masks.append(mask)
    np.testing.assert_array_equal(*masks)


def test_continuous_response_rounds_to_default_output():
    image=np.linspace(0,1,12*10*3).reshape(12,10,3)
    support=np.ones((12,10),bool);support[0]=False
    setting=CameraResponseSettings(exposure_mode='automatic')
    integer,mask,meta=apply_camera_response(image,support,setting)
    continuous,mask2,meta2=apply_camera_response(image,support,setting,quantize=False)
    np.testing.assert_array_equal(integer,np.rint(255*continuous).astype(np.uint8))
    np.testing.assert_array_equal(mask,mask2)
    assert meta['exposure_gain']==meta2['exposure_gain']
    assert continuous.dtype==float and np.max(abs(integer/255-continuous))<=.5/255
