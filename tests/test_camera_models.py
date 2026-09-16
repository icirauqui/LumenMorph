import cv2
import numpy as np
import pytest

from backend.src.camera_models import OpenCVFisheyeCamera, PinholeCamera, pixel_center_rays


def test_pinhole_axis_pixel_centers_and_independent_projection():
    c = PinholeCamera(20, 16, 13., 15., 10.25, 7.4)
    directions, valid = pixel_center_rays(c)
    assert valid.all()
    uv, ok = c.project(directions * 3.)
    rows, columns = np.mgrid[:16, :20]
    np.testing.assert_allclose(uv, np.stack((columns + .5, rows + .5), axis=-1), atol=4e-15)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1., atol=2e-16)
    p = np.array([[1., 2., 4.]])
    uv, _ = c.project(p)
    np.testing.assert_allclose(uv, [[13.5, -.1]], atol=2e-15)
    assert not c.contains(uv)[0]  # Mapping validity is not image containment.


@pytest.mark.parametrize('distortion', [(0., 0., 0., 0.),
    (-.1396589, -.0003139987, .001505504, -.0001316715), (.05, -.004, .001, .0001)])
def test_fisheye_matches_independent_opencv_projection_and_inverse(distortion):
    c = OpenCVFisheyeCamera(1472, 1104, 717.6911, 718.0214, 734.7278, 552.0724, distortion)
    rng = np.random.default_rng(372)
    points = rng.normal(size=(500, 3))
    points[:, 2] = rng.uniform(.4, 4., 500)
    actual, valid = c.project(points)
    opencv_points = points * [1., -1., 1.]
    K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1.]])
    expected, _ = cv2.fisheye.projectPoints(opencv_points.reshape(-1, 1, 3), np.zeros(3),
                                           np.zeros(3), K, np.array(distortion))
    assert valid.all()
    np.testing.assert_allclose(actual, expected[:, 0], rtol=0, atol=7e-13)
    rays, inverse_valid = c.rays(actual)
    assert inverse_valid.all()
    expected_ray = points / np.linalg.norm(points, axis=1)[:, None]
    np.testing.assert_allclose(rays, expected_ray, rtol=0, atol=2e-14)
    inverse = cv2.fisheye.undistortPoints(actual[:, None], K, np.array(distortion),
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-13))[:, 0]
    normalized = np.c_[inverse[:, 0], -inverse[:, 1], np.ones(len(inverse))]
    normalized /= np.linalg.norm(normalized, axis=1)[:, None]
    np.testing.assert_allclose(rays, normalized, rtol=0, atol=5e-13)


def test_fisheye_is_equidistant_at_zero_distortion_not_pinhole():
    c = OpenCVFisheyeCamera(100, 100, 20., 20., 50., 50.)
    uv, valid = c.project([[1., 0., 1.], [0., 0., 2.]])
    assert valid.all()
    np.testing.assert_allclose(uv, [[50 + 5*np.pi, 50.], [50., 50.]])
    rays, valid = c.rays([[50., 50.]])
    np.testing.assert_allclose(rays, [[0., 0., 1.]])


def test_fisheye_rejects_folded_calibration_and_outside_angular_domain():
    with pytest.raises(ValueError, match='monotone'):
        OpenCVFisheyeCamera(100, 100, 20, 20, 50, 50, (-1., 0, 0, 0))
    c = OpenCVFisheyeCamera(100, 100, 20, 20, 50, 50, max_theta=.8)
    uv, valid = c.project([[0, 0, -1], [10, 0, 1], [0, 0, 0]])
    assert not valid.any() and np.isnan(uv).all()
    rays, valid = c.rays([[1000, 50], [np.nan, 50]])
    assert not valid.any() and np.isnan(rays).all()


@pytest.mark.parametrize('args', [(0, 16, 10, 10, 0, 0), (10., 16, 10, 10, 0, 0),
                                 (10, 16, -1, 10, 0, 0), (10, 16, 10, 10, np.nan, 0)])
def test_invalid_intrinsics_are_rejected(args):
    with pytest.raises(ValueError):
        PinholeCamera(*args)


@pytest.mark.parametrize('model',[PinholeCamera,OpenCVFisheyeCamera])
def test_opencv_interoperability_requires_joint_world_and_pose_conversion(model):
    import cv2
    from backend.src.camera_models import opencv_pose_with_reflected_world
    R=cv2.Rodrigues(np.array([.2,-.3,.4]))[0];C=np.array([.12,-.07,.09])
    rng=np.random.default_rng(321); local=rng.uniform(-.3,.3,(100,3));local[:,2]+=2
    world=local@R.T+C
    camera=model(160,120,100,102,80,60)
    expected,valid=camera.project(local);assert valid.all()
    converted=opencv_pose_with_reflected_world(R,C);cv_world=world@converted['world_transform'].T
    np.testing.assert_allclose(np.linalg.det(converted['R_cw']),1.,atol=1e-12)
    k=np.array([[100.,0,80],[0,102,60],[0,0,1]])
    rvec=cv2.Rodrigues(converted['R_cw'])[0]
    if isinstance(camera,OpenCVFisheyeCamera):
        actual,_=cv2.fisheye.projectPoints(cv_world.reshape(-1,1,3),rvec,converted['t_cw'],k,np.zeros(4))
    else:
        actual,_=cv2.projectPoints(cv_world,rvec,converted['t_cw'],k,None)
    np.testing.assert_allclose(actual.reshape(-1,2),expected,atol=2e-12)
