import json
import numpy as np
import pytest
from backend.src.continuous_material import FourierMaterialField,seeded_fourier_field


def plane(n=2):
    x,y=np.meshgrid(np.linspace(-.02,.02,n),np.linspace(-.02,.02,n))
    v=np.c_[x.ravel(),y.ravel(),np.full(x.size,.04)]
    faces=np.array([[a,b,c] for j in range(n-1) for i in range(n-1)
        for a,b,c in [(n*j+i,n*j+i+1,n*j+i+n+1),(n*j+i,n*j+i+n+1,n*j+i+n)]])
    return v,faces


def test_explicit_spectrum_serialization_and_independent_analytic_values():
    waves=np.array([[2*np.pi/.001,0,0],[0,2*np.pi/.003,0]])
    field=FourierMaterialField((10.,20.,30.),(.3,.2,.1),waves,[.3,-.2],[.2,.1])
    points=np.array([[.0011,.0023,0],[.0009,-.0022,.003]])+field.origin_meter
    expected=np.array([.3,.2,.1])[None,:]*(1+.2*np.cos(2*np.pi*(points[:,0]-10)/.001+.3)+.1*np.cos(2*np.pi*(points[:,1]-20)/.003-.2))[:,None]
    np.testing.assert_allclose(field.evaluate(points),expected,rtol=1e-12,atol=1e-12)
    restored=FourierMaterialField.from_dict(json.loads(json.dumps(field.to_dict())))
    assert field==restored and field.sha256()==restored.sha256()
    waves[:]=0;np.testing.assert_array_equal(field.evaluate(points),restored.evaluate(points))


def test_seeded_spectrum_matches_existing_vertex_modulation_at_vertices():
    from backend.src.material_appearance import MaterialAppearanceSettings,material_vertex_albedo
    v,f=plane(7);old,_,_=material_vertex_albedo(v,f,MaterialAppearanceSettings(network_roots=0))
    field=seeded_fourier_field(v.mean(axis=0))
    np.testing.assert_allclose(field.evaluate(v),old,atol=3e-8,rtol=0)
    assert seeded_fourier_field(v.mean(axis=0)).sha256()==field.sha256()


def test_spectrum_and_coordinate_contract_failures():
    with pytest.raises(ValueError):FourierMaterialField(base_albedo_rgb=(1,1,1),wavevectors_radian_per_meter=((1,0,0),),phases_radian=(0,),amplitudes=(.2,))
    with pytest.raises(ValueError):FourierMaterialField(phases_radian=(0,))
    with pytest.raises(ValueError):FourierMaterialField(wavevectors_radian_per_meter=((np.inf,0,0),),phases_radian=(0,),amplitudes=(.2,))
    with pytest.raises(ValueError):seeded_fourier_field((0,0,0),seed=True)
    with pytest.raises(ValueError):seeded_fourier_field((0,0,0),wavelengths_meter=(0,),relative_amplitudes=(.1,))
    field=seeded_fourier_field((0,0,0))
    with pytest.raises(ValueError):field.evaluate([[np.nan,0,0]])
    with pytest.raises(ValueError):field.renderer_reference_offsets([[1e100,0,0]])
    assert FourierMaterialField().evaluate(np.empty((0,3))).shape==(0,3)


def runtime():
    pytest.importorskip('mitsuba');pytest.importorskip('drjit')
    from backend.src.endoscopic_renderer import render_endoscopic,AppearanceSettings
    from backend.src.camera_models import PinholeCamera
    return render_endoscopic,AppearanceSettings,PinholeCamera


@pytest.mark.parametrize('material',['diffuse','roughplastic'])
def test_constant_field_matches_bitmap_and_vertex_controls(material):
    render,Settings,Camera=runtime();v,f=plane();color=(.48,.16,.13);camera=Camera(8,8,12,12,4,4)
    settings=Settings(material=material,light_positions_camera=((0.,0.,0.),),light_intensities_rgb=((.0005,.0005,.0005),),samples_per_pixel=32,max_depth=3,threads=1)
    _,bitmap,_=render(v,f,np.zeros((len(v),2)),np.tile(color,(2,2,1)),camera,np.eye(3),np.zeros(3),settings)
    _,vertex,_=render(v,f,None,None,camera,np.eye(3),np.zeros(3),settings,vertex_albedo_linear_rgb=np.tile(color,(len(v),1)))
    _,continuous,meta=render(v,f,None,None,camera,np.eye(3),np.zeros(3),settings,continuous_material=FourierMaterialField(base_albedo_rgb=color),material_reference_vertices_meter=v)
    np.testing.assert_allclose(continuous['linear_radiance_rgb'],bitmap['linear_radiance_rgb'],rtol=3e-6,atol=1e-9)
    np.testing.assert_allclose(continuous['linear_radiance_rgb'],vertex['linear_radiance_rgb'],rtol=3e-6,atol=1e-9)
    np.testing.assert_array_equal(continuous['depth_z'],bitmap['depth_z'])
    assert 'sampling proxy' in meta['texture_mean_semantics']


@pytest.mark.parametrize('material',['diffuse','roughplastic'])
def test_continuous_field_coarse_subdivided_plane_and_analytic_diffuse_radiometry(material):
    render,Settings,Camera=runtime();camera=Camera(9,7,12,13,4.3,3.1)
    settings=Settings(material=material,light_positions_camera=((0.,0.,0.),),light_intensities_rgb=((.0005,.0005,.0005),),samples_per_pixel=1,max_depth=2,threads=1)
    field=seeded_fourier_field((0,0,.04),seed=92,wavelengths_meter=(.0013,.003,.012),relative_amplitudes=(.3,.15,.1),modes_per_scale=3,base_albedo_rgb=(.3,.2,.1))
    answers=[]
    for n in [2,25]:
        v,f=plane(n);_,truth,_=render(v,f,None,None,camera,np.eye(3),np.zeros(3),settings,continuous_material=field,material_reference_vertices_meter=v)
        assert truth['valid'].all();p=truth['position_world'];distance=np.linalg.norm(p,axis=-1)
        expected=truth['diffuse_albedo_linear_rgb']/np.pi*.0005*.04/distance[...,None]**3
        if material=='diffuse':
            np.testing.assert_allclose(truth['linear_radiance_rgb'],expected,rtol=2e-5,atol=1e-8)
        answers.append(truth)
    np.testing.assert_allclose(answers[0]['material_reference_position_meter'],answers[1]['material_reference_position_meter'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(answers[0]['diffuse_albedo_linear_rgb'],answers[1]['diffuse_albedo_linear_rgb'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(answers[0]['linear_radiance_rgb'],answers[1]['linear_radiance_rgb'],rtol=3e-5,atol=1e-8)
    v,f=plane();_,discrete,_=render(v,f,None,None,camera,np.eye(3),np.zeros(3),settings,vertex_albedo_linear_rgb=field.evaluate(v))
    assert np.max(np.abs(discrete['diffuse_albedo_linear_rgb']-answers[0]['diffuse_albedo_linear_rgb']))>.01


def test_deformed_geometry_preserves_reference_material_point():
    render,Settings,Camera=runtime();reference,f=plane();field=seeded_fourier_field((0,0,.04),seed=8)
    bary=np.array([.23,.34,.43]);point=bary@reference[f[0]]
    moved=reference.copy();moved[:,0]*=1.35;moved[:,1]*=.8;moved[:,2]+=np.array([.001,-.002,.003,-.004])
    colors=[];settings=Settings(material='diffuse',samples_per_pixel=1,max_depth=2,threads=1)
    for geometry in [reference,moved]:
        hit=bary@geometry[f[0]];center=hit-[0,0,.03]
        _,truth,_=render(geometry,f,None,None,Camera(1,1,1,1,.5,.5),np.eye(3),center,settings,continuous_material=field,material_reference_vertices_meter=reference)
        assert truth['valid'][0,0]
        np.testing.assert_allclose(truth['material_reference_position_meter'][0,0],point,atol=1e-8,rtol=0)
        colors.append(truth['diffuse_albedo_linear_rgb'][0,0])
    np.testing.assert_allclose(colors[0],colors[1],atol=2e-6,rtol=0)
    np.testing.assert_allclose(colors[0],field.evaluate(point),atol=2e-6,rtol=0)


def test_exclusive_material_and_reference_mapping_contract():
    render,Settings,Camera=runtime();v,f=plane();field=FourierMaterialField();camera=Camera(1,1,1,1,.5,.5)
    with pytest.raises(ValueError):render(v,f,None,None,camera,np.eye(3),np.zeros(3),continuous_material=field)
    with pytest.raises(ValueError):render(v,f,None,None,camera,np.eye(3),np.zeros(3),continuous_material=field,material_reference_vertices_meter=v[:-1])
    with pytest.raises(ValueError):render(v,f,None,None,camera,np.eye(3),np.zeros(3),continuous_material=field,material_reference_vertices_meter=v,vertex_albedo_linear_rgb=np.ones_like(v)*.3)
    with pytest.raises(ValueError):render(v,f,None,None,camera,np.eye(3),np.zeros(3),material_reference_vertices_meter=v)


def test_roughplastic_sampling_proxy_does_not_change_direct_reflectance():
    # The proxy controls BSDF proposal weights, not local BRDF evaluation.
    class DifferentProxy(FourierMaterialField):
        def sampling_mean_proxy(self):return .9
    render,Settings,Camera=runtime();v,f=plane()
    original=seeded_fourier_field((0,0,.04),seed=72)
    values=original.to_dict();values.pop('schema');alternate=DifferentProxy(**values)
    settings=Settings(samples_per_pixel=1,max_depth=2,threads=1)
    radiances=[]
    for field in [original,alternate]:
        _,truth,_=render(v,f,None,None,Camera(7,5,10,10,3.2,2.3),np.eye(3),np.zeros(3),settings,
            continuous_material=field,material_reference_vertices_meter=v)
        radiances.append(truth['linear_radiance_rgb'])
    np.testing.assert_allclose(radiances[0],radiances[1],atol=1e-8,rtol=1e-6)
