import numpy as np
import pytest

pytest.importorskip('mitsuba')
from backend.src.camera_models import PinholeCamera, OpenCVFisheyeCamera
from backend.src.endoscopic_renderer import AppearanceSettings, render_endoscopic


def panel(camera, **kwargs):
    vertices = np.array([[-1., -1., 2.], [1., -1., 2.], [1., 1., 2.], [-1., 1., 2.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[.1, .1], [.4, .1], [.4, .9], [.1, .9]])
    settings = AppearanceSettings(material='diffuse', light_positions_camera=((0., 0., 0.),),
            light_intensities_rgb=((1., 1., 1.),), samples_per_pixel=1, max_depth=2, threads=1, **kwargs)
    return render_endoscopic(vertices, faces, uv, np.full((8, 8, 3), .5), camera, np.eye(3), np.zeros(3), settings)


@pytest.mark.parametrize('cls', [PinholeCamera, OpenCVFisheyeCamera])
def test_rgb_and_labels_share_physical_center_rays(cls):
    c = cls(8, 8, 12., 12., 4., 4.)
    rgb, truth, metadata = panel(c)
    assert truth['valid'].all()
    np.testing.assert_allclose(truth['depth_z'], 2., atol=3e-7)
    uv, valid = c.project(truth['position_world'])
    rows, cols = np.mgrid[:8, :8]
    np.testing.assert_allclose(uv, np.stack((cols+.5, rows+.5), axis=-1), atol=3e-6)
    np.testing.assert_allclose(truth['material_barycentric'].sum(axis=-1), 1., atol=1e-8)
    # Independent Lambertian irradiance: I*cos(theta)/(r**2), reflectance/pi.
    p = truth['position_world']
    radius = np.linalg.norm(p, axis=-1)
    expected = .5/np.pi * 2/radius**3
    np.testing.assert_allclose(truth['linear_radiance_rgb'], np.repeat(expected[..., None], 3, axis=-1), rtol=3e-6)
    assert rgb.dtype == np.uint8 and metadata['scene_length_unit'] == 'meter'


def test_fixed_center_geometry_is_independent_of_radiance_sampling():
    c = PinholeCamera(4, 4, 9., 9., 2., 2.)
    _, a, _ = panel(c, seed=7)
    _, b, _ = panel(c, seed=23)
    for key in ('depth_z', 'triangle_id', 'material_barycentric', 'position_world', 'normal_world'):
        np.testing.assert_array_equal(a[key], b[key])


def test_invalid_material_settings_rejected():
    with pytest.raises(ValueError):
        AppearanceSettings(light_intensities_rgb=((-1., 0., 0.),))
    with pytest.raises(ValueError):
        AppearanceSettings(roughness=0.)


def test_smooth_shading_changes_reflectance_but_preserves_material_intersections():
    from dataclasses import replace
    camera=PinholeCamera(8,8,12.,12.,4.,4.)
    vertices=np.array([[-1.,-1.,2.],[1.,-1.,2.],[1.,1.,2.],[-1.,1.,3.]])
    faces=np.array([[0,1,2],[0,2,3]])
    uv=np.array([[.1,.1],[.4,.1],[.4,.9],[.1,.9]])
    settings=AppearanceSettings(material='diffuse',light_positions_camera=((0.,0.,0.),),
              light_intensities_rgb=((1.,1.,1.),),samples_per_pixel=1,max_depth=2,threads=1)
    _, flat, _=render_endoscopic(vertices,faces,uv,np.full((8,8,3),.5),camera,np.eye(3),np.zeros(3),settings)
    _, smooth, _=render_endoscopic(vertices,faces,uv,np.full((8,8,3),.5),camera,np.eye(3),np.zeros(3),replace(settings,smooth_shading=True))
    for key in ('depth_z','triangle_id','material_barycentric','position_world','normal_world','valid'):
        np.testing.assert_array_equal(flat[key],smooth[key])
    good=smooth['valid']
    np.testing.assert_allclose(np.linalg.norm(smooth['shading_normal_world'][good],axis=1),1.,atol=1e-12)
    assert np.max(np.abs(flat['linear_radiance_rgb']-smooth['linear_radiance_rgb']))>1e-4
    assert np.max(np.abs(flat['normal_world'][good]-smooth['shading_normal_world'][good]))>.01


def test_atlas_endpoints_preserve_nonuniform_texture():
    camera=PinholeCamera(8,8,8.,8.,4.,4.)
    v=np.array([[-1.,-1.,2.],[1.,-1.,2.],[1.,1.,2.],[-1.,1.,2.]])
    f=np.array([[0,1,2],[0,2,3]])
    uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    texture=np.zeros((8,8,3));texture[:,:4,0]=.5;texture[:,4:,1]=.5
    settings=AppearanceSettings(material='diffuse',uv_mode='atlas',light_positions_camera=((0.,0.,0.),),
              light_intensities_rgb=((1.,1.,1.),),samples_per_pixel=1,max_depth=2,threads=1)
    _,truth,_=render_endoscopic(v,f,uv,texture,camera,np.eye(3),np.zeros(3),settings)
    radiance=truth['linear_radiance_rgb']
    assert radiance[:,:3,0].mean() > .02 and radiance[:,:3,1].max() < 1e-6
    assert radiance[:,5:,1].mean() > .02 and radiance[:,5:,0].max() < 1e-6
    assert truth['mesh_topology_sha256'].shape == ()


def test_material_vertex_albedo_matches_analytic_lambertian_radiance_after_motion():
    camera=PinholeCamera(8,8,12,12,4,4)
    faces=np.array([[0,1,2],[0,2,3]])
    reference=np.array([[-1.,-1.,2.],[1.,-1.,2.],[1.,1.,2.],[-1.,1.,2.]])
    colors=np.array([[.1,.2,.4],[.5,.2,.3],[.8,.3,.2],[.2,.4,.1]])
    settings=AppearanceSettings(material='diffuse',light_positions_camera=((0.,0.,0.),),
              light_intensities_rgb=((1.,1.,1.),),samples_per_pixel=1,max_depth=2,threads=1)
    for offset in [0., .3]:
        vertices=reference+[offset,0,offset]
        _,truth,meta=render_endoscopic(vertices,faces,None,None,camera,np.eye(3),np.zeros(3),settings,
                                     vertex_albedo_linear_rgb=colors)
        good=truth['valid'];p=truth['position_world'][good]
        interpolated=truth['diffuse_albedo_linear_rgb'][good]
        # Color interpolation uses material IDs, while radiometry follows moved geometry.
        expected=interpolated/np.pi*(2+offset)/np.linalg.norm(p,axis=1)[:,None]**3
        np.testing.assert_allclose(truth['linear_radiance_rgb'][good],expected,rtol=4e-6,atol=1e-9)
        assert 'vertex ID' in meta['albedo_representation']
    with pytest.raises(ValueError,match='only finite'):
        render_endoscopic(reference,faces,None,np.ones((8,8,3)),camera,np.eye(3),np.zeros(3),settings,
                          vertex_albedo_linear_rgb=colors)


def test_coated_vertex_color_matches_constant_bitmap_radiance():
    camera=PinholeCamera(8,8,12,12,4,4)
    v=np.array([[-1.,-1.,2.],[1.,-1.,2.],[1.,1.,2.],[-1.,1.,2.]])
    f=np.array([[0,1,2],[0,2,3]])
    uv=np.zeros((4,2));color=np.array([.48,.16,.13])
    settings=AppearanceSettings(light_positions_camera=((0.,0.,0.),),
        light_intensities_rgb=((1.,1.,1.),),samples_per_pixel=32,max_depth=3,threads=1)
    _,bitmap,_=render_endoscopic(v,f,uv,np.tile(color,(2,2,1)),camera,np.eye(3),np.zeros(3),settings)
    _,vertex,_=render_endoscopic(v,f,None,None,camera,np.eye(3),np.zeros(3),settings,
        vertex_albedo_linear_rgb=np.tile(color,(4,1)))
    np.testing.assert_allclose(vertex['linear_radiance_rgb'],bitmap['linear_radiance_rgb'],rtol=2e-6,atol=1e-9)
    np.testing.assert_array_equal(vertex['depth_z'],bitmap['depth_z'])


@pytest.mark.parametrize('batch,spp', [(1,1),(5,3),(8192,1)])
def test_fold_edge_truth_uses_path_integrator_coherent_first_hit(batch,spp):
    # Exact shared-edge rays can choose different faces with different Embree
    # traversal hints; the stored nonplanar normal must match the RGB path.
    v=np.array([[-1.,-1.,2.],[1.,-1.,2.2],[1.,1.,2.8],[-1.,1.,1.8]])
    f=np.array([[0,1,2],[0,2,3]])
    colors=np.array([[.2,.3,.4],[.5,.2,.1],[.7,.4,.2],[.2,.2,.5]])
    settings=AppearanceSettings(material='diffuse',samples_per_pixel=spp,
        ray_batch_pixels=batch,max_depth=2,threads=1,
        light_positions_camera=((0.,0.,0.),),light_intensities_rgb=((1.,1.,1.),))
    _,truth,_=render_endoscopic(v,f,None,None,PinholeCamera(6,6,8,8,3,3),
        np.eye(3),np.zeros(3),settings,vertex_albedo_linear_rgb=colors)
    valid=truth['valid'];p=truth['position_world'][valid]
    distance=np.linalg.norm(p,axis=1)
    cosine=np.abs(np.sum(truth['normal_world'][valid]*p/distance[:,None],axis=1))
    expected=truth['diffuse_albedo_linear_rgb'][valid]/np.pi*cosine[:,None]/distance[:,None]**2
    np.testing.assert_allclose(truth['linear_radiance_rgb'][valid],expected,rtol=5e-6,atol=1e-9)
