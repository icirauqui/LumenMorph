"""Opt-in endoscopic forward-rendering pilot, independent of the sealed producer.

Mitsuba traces identical center rays for RGB and material truth, using arbitrary
camera models from camera_models. Radiance samples vary light paths, never the
primary pixel position. There is no spatial antialiasing, motion blur or learned
image translation in this pilot. Camera-mounted point sources illuminate a
diffuse or rough dielectric-coated diffuse surface, including cast shadows and
bounded interreflection. Parameters are assigned, not calibrated tissue optics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import numpy as np

from .camera_models import PinholeCamera, pixel_center_rays
from .continuous_material import FourierMaterialField, mitsuba_fourier_texture


@dataclass(frozen=True)
class AppearanceSettings:
    material: str = 'roughplastic'
    roughness: float = .18
    interior_ior: float = 1.38
    exterior_ior: float = 1.0
    light_positions_camera: tuple = ((-.003, 0., 0.), (.003, 0., 0.))
    light_intensities_rgb: tuple = ((.0005, .0005, .0005), (.0005, .0005, .0005))
    samples_per_pixel: int = 16
    max_depth: int = 4
    seed: int = 0
    threads: int = 4
    ray_batch_pixels: int = 8192
    exposure: float = 1.
    smooth_shading: bool = False
    uv_mode: str = 'periodic_u'

    def __post_init__(self):
        if self.material not in ('diffuse', 'roughplastic'):
            raise ValueError('Unknown material')
        if not isinstance(self.smooth_shading, bool):
            raise ValueError('smooth_shading must be boolean')
        if self.uv_mode not in ('periodic_u', 'atlas'):
            raise ValueError('uv_mode must be periodic_u or atlas')
        if not np.isfinite([self.roughness, self.interior_ior, self.exterior_ior, self.exposure]).all():
            raise ValueError('Nonfinite material/exposure parameter')
        if not 0 < self.roughness <= 1 or min(self.interior_ior, self.exterior_ior, self.exposure) <= 0:
            raise ValueError('Invalid roughness, refractive index or exposure')
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
               for x in (self.samples_per_pixel, self.max_depth, self.threads, self.ray_batch_pixels)):
            raise ValueError('Sampling/depth/thread/batch settings must be positive integers')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('Seed must be a uint32 integer')
        positions, intensities = map(np.asarray, (self.light_positions_camera, self.light_intensities_rgb))
        if (positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape != intensities.shape
                or not np.isfinite(positions).all() or not np.isfinite(intensities).all()
                or np.any(intensities < 0)):
            raise ValueError('Lights require matching finite Nx3 positions and nonnegative RGB intensities')


def linear_to_srgb(values):
    values = np.clip(np.asarray(values, dtype=float), 0., 1.)
    return np.where(values <= .0031308, 12.92*values, 1.055*values**(1/2.4)-.055)


def mesh_topology_sha256(vertex_count, triangles):
    """Bind material vertex IDs and oriented face connectivity/order, not positions."""
    faces = np.asarray(triangles)
    if (not isinstance(vertex_count, (int, np.integer)) or vertex_count < 1
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError('Invalid topology identity input')
    payload = np.asarray([vertex_count, len(faces)], dtype='<u8').tobytes()
    payload += np.asarray(faces, dtype='<u8', order='C').tobytes()
    return hashlib.sha256(b'material_triangle_topology_v1\0'+payload).hexdigest()


def _runtime(threads):
    try:
        import mitsuba as mi
        import drjit as dr
    except ImportError as exc:
        raise RuntimeError('Install the optional pinned rendering runtime') from exc
    if mi.variant() is None:
        mi.set_variant('llvm_ad_rgb')
    if mi.variant() != 'llvm_ad_rgb':
        raise RuntimeError('Pilot requires a separate process using llvm_ad_rgb')
    dr.set_thread_count(threads)
    return mi, dr


def _vertex_texture(mi, mean_reflectance):
    """Supply the mesh attribute's area mean for Mitsuba 3.9.1 BSDF sampling.

    The stock mesh_attribute evaluates barycentric color but has no mean().
    Roughplastic requires that scalar to choose its sampling mixture; actual
    reflectance evaluation remains delegated to the stock attribute texture.
    """
    class MeanAwareAttribute(mi.Texture):
        def __init__(self):
            super().__init__(mi.Properties())
            self.attribute = mi.load_dict({'type': 'mesh_attribute', 'name': 'vertex_albedo'})

        def eval(self, si, active=True):
            return self.attribute.eval(si, active)

        def eval_1(self, si, active=True):
            return self.attribute.eval_1(si, active)

        def eval_3(self, si, active=True):
            return self.attribute.eval_3(si, active)

        def mean(self):
            return mean_reflectance

        def is_spatially_varying(self):
            return True

        def to_string(self):
            return 'AreaMeanAwareMaterialVertexAttribute'

    return MeanAwareAttribute()


def render_endoscopic(vertices_world, triangles, uv_vertices, albedo_linear_rgb,
                      camera: PinholeCamera, R_wc, C_world,
                      settings: AppearanceSettings = AppearanceSettings(), *,
                      vertex_albedo_linear_rgb=None, continuous_material=None,
                      material_reference_vertices_meter=None):
    """Return RGB uint8, dense truth, and explicit rendering metadata.

    Geometry/light positions must use meters; material/texture albedo is linear
    RGB reflectance in [0,1]. No anatomical scale is inferred by this function.
    Vertices are quantized to float32 for this renderer, and returned material
    points refer to those exact quantized positions. Triangle identity/order is
    retained when vertices are expanded solely for periodic texture seams.
    Continuous Fourier appearance instead carries immutable reference XYZ per
    material vertex, interpolates those coordinates, then evaluates the field.
    Supply the same reference mapping and field across deformed geometry states.
    """
    vertices = np.asarray(vertices_world, dtype=np.float32)
    faces = np.asarray(triangles)
    vertex_albedo = None if vertex_albedo_linear_rgb is None else np.asarray(vertex_albedo_linear_rgb, dtype=np.float32)
    albedo = None if albedo_linear_rgb is None else np.asarray(albedo_linear_rgb, dtype=np.float32)
    R, C = np.asarray(R_wc, dtype=float), np.asarray(C_world, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Invalid finite triangular geometry')
    reference_offset = None
    if continuous_material is not None:
        if not isinstance(continuous_material, FourierMaterialField) or vertex_albedo is not None or albedo is not None:
            raise ValueError('Continuous material must be a FourierMaterialField with no bitmap or vertex albedo')
        reference_offset = continuous_material.renderer_reference_offsets(material_reference_vertices_meter)
        if reference_offset.shape != vertices.shape:
            raise ValueError('Material reference vertices must match geometry material IDs')
    elif material_reference_vertices_meter is not None:
        raise ValueError('Reference coordinates require a continuous material field')
    uv = np.zeros((len(vertices), 2)) if uv_vertices is None and (vertex_albedo is not None or continuous_material is not None) else np.asarray(uv_vertices, dtype=float)
    if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
        raise ValueError('Material UV coordinates must match vertices')
    if vertex_albedo is None and continuous_material is None:
        if (albedo is None or albedo.ndim != 3 or albedo.shape[2] != 3 or min(albedo.shape[:2]) < 2
                or not np.isfinite(albedo).all() or albedo.min() < 0 or albedo.max() > 1):
            raise ValueError('Albedo must be a finite HxWx3 linear reflectance texture in [0,1]')
    elif vertex_albedo is not None and (albedo is not None or vertex_albedo.shape != vertices.shape or not np.isfinite(vertex_albedo).all()
          or vertex_albedo.min() < 0 or vertex_albedo.max() > 1):
        raise ValueError('Provide only finite Nx3 vertex reflectances in [0,1], with bitmap albedo None')
    if (R.shape != (3, 3) or C.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(C).all()
            or not np.allclose(R.T@R, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-10, rtol=0)):
        raise ValueError('Pose requires a finite proper orthogonal camera-to-world rotation and center')
    points = vertices[faces]
    cross = np.cross(points[:, 1].astype(float)-points[:, 0], points[:, 2].astype(float)-points[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if (lengths <= 0).any():
        raise ValueError('Degenerate triangles after renderer quantization')
    normals = cross / lengths[:, None]
    vertex_normals = None
    if settings.smooth_shading:
        vertex_normals = np.zeros(vertices.shape, dtype=float)
        np.add.at(vertex_normals, faces.ravel(), np.repeat(cross, 3, axis=0))
        norm = np.linalg.norm(vertex_normals, axis=1)
        if np.any(norm[np.unique(faces)] <= 0):
            raise ValueError('Undefined vertex shading normal; check mesh orientation/topology')
        vertex_normals /= np.where(norm > 0, norm, 1.)[:, None]
    mi, dr = _runtime(settings.threads)
    triangle_uv = uv[faces].copy()
    if settings.uv_mode == 'periodic_u':
        triangle_uv[:, :, 0] %= 1.
        crossing = np.ptp(triangle_uv[:, :, 0], axis=1) > .5
        triangle_uv[:, :, 0] += (crossing[:, None] & (triangle_uv[:, :, 0] < .5))
    # Array images have top-down rows. This explicit flip retains v-up material UV.
    triangle_uv[:, :, 1] = 1. - triangle_uv[:, :, 1]
    if continuous_material is not None:
        texture = mitsuba_fourier_texture(mi, dr, continuous_material)
    elif vertex_albedo is not None:
        texture = _vertex_texture(mi, float(np.average(
            vertex_albedo[faces].astype(float).mean(axis=(1, 2)), weights=lengths)))
    else:
        texture = {'type': 'bitmap', 'bitmap': mi.Bitmap(albedo), 'raw': True,
                   'filter_type': 'bilinear', 'wrap_mode': 'repeat' if settings.uv_mode == 'periodic_u' else 'clamp'}
    material = ({'type': 'diffuse', 'reflectance': texture} if settings.material == 'diffuse'
                else {'type': 'roughplastic', 'distribution': 'ggx', 'alpha': settings.roughness,
                      'int_ior': settings.interior_ior, 'ext_ior': settings.exterior_ior,
                      'diffuse_reflectance': texture, 'nonlinear': True})
    props = mi.Properties()
    props['bsdf'] = mi.load_dict({'type': 'twosided', 'material': material})
    mesh = mi.Mesh('material_surface', len(faces)*3, len(faces), props,
                   has_vertex_texcoords=True, has_vertex_normals=settings.smooth_shading)
    if vertex_albedo is not None:
        mesh.add_attribute('vertex_albedo', 3, mi.Float(vertex_albedo[faces].reshape(-1)))
    if reference_offset is not None:
        mesh.add_attribute('vertex_reference_offset', 3, mi.Float(reference_offset[faces].reshape(-1)))
    params = mi.traverse(mesh)
    params['vertex_positions'] = mi.Float(points.reshape(-1))
    params['faces'] = mi.UInt(np.arange(len(faces)*3, dtype=np.uint32))
    params['vertex_texcoords'] = mi.Float(triangle_uv.astype(np.float32).reshape(-1))
    if vertex_normals is not None:
        params['vertex_normals'] = mi.Float(vertex_normals[faces].astype(np.float32).reshape(-1))
    params.update()
    scene_description = {'type': 'scene', 'surface': mesh,
                         'integrator': {'type': 'path', 'max_depth': settings.max_depth}}
    for i, (pos, intensity) in enumerate(zip(settings.light_positions_camera, settings.light_intensities_rgb)):
        scene_description[f'scope_light_{i}'] = {
            'type': 'point', 'position': (C+R@np.asarray(pos)).tolist(),
            'intensity': {'type': 'rgb', 'value': list(intensity)}}
    scene = mi.load_dict(scene_description)
    camera_rays, camera_valid = pixel_center_rays(camera)
    shape = camera_valid.shape
    directions = camera_rays.reshape(-1, 3) @ R.T
    sensor_valid = camera_valid.ravel()
    directions[~sensor_valid] = R[:, 2]
    origin = mi.Point3f(*map(float, C))
    ray = mi.Ray3f(origin, mi.Vector3f(directions.T))
    # Match Mitsuba's path integrator first-bounce traversal hint. In Embree,
    # coherent/incoherent traversal may select opposite faces at exact fold-edge
    # ties, changing the geometric normal even when the material point agrees.
    preliminary = scene.ray_intersect_preliminary(ray, coherent=True, active=mi.Bool(sensor_valid))
    valid = np.array(preliminary.is_valid()) & sensor_valid
    face_ids = np.array(preliminary.prim_index).astype(np.int64)
    face_ids[~valid] = -1
    bary = np.full((len(valid), 3), np.nan)
    hit_uv = np.array(preliminary.prim_uv).T[valid]
    bary[valid] = np.c_[1.-hit_uv.sum(axis=1), hit_uv]
    world = np.full((len(valid), 3), np.nan)
    world[valid] = np.sum(bary[valid, :, None] * vertices[faces[face_ids[valid]]].astype(float), axis=1)
    normal_world = np.full_like(world, np.nan)
    normal_world[valid] = normals[face_ids[valid]]
    shading_normal = normal_world.copy()
    if vertex_normals is not None:
        shading_normal[valid] = np.sum(bary[valid, :, None] * vertex_normals[faces[face_ids[valid]]], axis=1)
        shading_normal[valid] /= np.linalg.norm(shading_normal[valid], axis=1)[:, None]
    depth = ((world-C) @ R)[:, 2]
    radiance = np.zeros((len(valid), 3), dtype=float)
    spp = settings.samples_per_pixel
    for start in range(0, len(valid), settings.ray_batch_pixels):
        end = min(start+settings.ray_batch_pixels, len(valid))
        d = np.tile(directions[start:end], (spp, 1))
        active = np.tile(sensor_valid[start:end], spp)
        rays = mi.RayDifferential3f(mi.Ray3f(origin, mi.Vector3f(d.T)))
        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed((settings.seed+start) % 2**32, len(d))
        value, _, _ = scene.integrator().sample(scene, sampler, rays, active=mi.Bool(active))
        dr.eval(value)
        samples = np.array(value).T.astype(float).reshape(spp, end-start, 3)
        if not np.isfinite(samples).all() or samples.min() < -1e-7:
            raise RuntimeError('Renderer produced invalid radiance')
        radiance[start:end] = samples.mean(axis=0)
    rgb = np.rint(255*linear_to_srgb(settings.exposure*radiance)).astype(np.uint8).reshape(*shape, 3)
    truth = {'depth_z': depth.reshape(shape), 'triangle_id': face_ids.reshape(shape),
             'material_barycentric': bary.reshape(*shape, 3), 'normal_world': normal_world.reshape(*shape, 3),
             'valid': valid.reshape(shape), 'sensor_valid': camera_valid,
             'position_world': world.reshape(*shape, 3), 'linear_radiance_rgb': radiance.reshape(*shape, 3)}
    truth['shading_normal_world'] = shading_normal.reshape(*shape, 3)
    truth['mesh_topology_sha256'] = np.asarray(mesh_topology_sha256(len(vertices), faces))
    if vertex_albedo is not None:
        sampled_albedo = np.full_like(world, np.nan)
        sampled_albedo[valid] = np.sum(bary[valid, :, None]*vertex_albedo[faces[face_ids[valid]]].astype(float), axis=1)
        truth['diffuse_albedo_linear_rgb'] = sampled_albedo.reshape(*shape, 3)
    if continuous_material is not None:
        sampled_reference = np.full_like(world, np.nan)
        sampled_reference[valid] = np.sum(bary[valid, :, None]*reference_offset[faces[face_ids[valid]]].astype(float), axis=1)
        sampled_albedo = np.full_like(world, np.nan)
        sampled_albedo[valid] = continuous_material.evaluate_offsets(sampled_reference[valid])
        truth['material_reference_position_meter'] = (sampled_reference+np.asarray(continuous_material.origin_meter)).reshape(*shape, 3)
        truth['diffuse_albedo_linear_rgb'] = sampled_albedo.reshape(*shape, 3)
    appearance_metadata = ({
        'albedo_representation': 'continuous Fourier field at barycentrically interpolated material reference XYZ',
        'albedo_sha256': continuous_material.sha256(),
        'continuous_material': continuous_material.to_dict(),
        'material_reference_vertices_sha256': hashlib.sha256(np.asarray(material_reference_vertices_meter, dtype='<f8', order='C').tobytes()).hexdigest(),
        'material_reference_offsets_float32_sha256': hashlib.sha256(np.asarray(reference_offset, dtype='<f4', order='C').tobytes()).hexdigest(),
        'material_coordinate_precision': 'reference XYZ minus fixed field origin stored as float32 mesh attribute; returned reference positions/albedo use float64 barycentric reconstruction and NumPy field evaluation; rendering uses float32 field arithmetic',
        'texture_mean_semantics': 'phase-ensemble RGB mean as fixed BSDF sampling proxy, not measured spatial area mean',
        'texture_sampling_scope': 'continuous reference field removes vertex-color interpolation aliasing; no pixel footprint filtering or anatomical/vascular validation'}
        if continuous_material is not None else {
            'albedo_representation': 'bitmap material UV' if vertex_albedo is None else 'material vertex ID; barycentric linear RGB reflectance',
            'albedo_sha256': hashlib.sha256(np.asarray(albedo if vertex_albedo is None else vertex_albedo, dtype='<f4', order='C').tobytes()).hexdigest()})
    return rgb, truth, {'schema': 'endoscopic_forward_renderer_pilot_v1', 'camera': camera.to_dict(),
                        'settings': asdict(settings), 'mitsuba': mi.__version__, 'drjit': dr.__version__,
                        'variant': mi.variant(), 'scene_length_unit': 'meter',
                        'primary_sampling': 'fixed pixel-center central-camera rays; no spatial antialiasing or motion blur',
                        'primary_traversal': 'coherent=True, matching Mitsuba path integrator first bounce; shared-edge face choice follows this traversal',
                        'geometry_precision': 'float32 quantized vertices/intersections; float64 returned barycentric reconstruction',
                        'normal_convention': 'normal_world is oriented triangle normal; shading_normal_world is oriented interpolated area-weighted vertex normal when enabled; neither is camera-facing flipped',
                        **appearance_metadata,
                        'material_scope': 'two-sided diffuse or rough dielectric-coated diffuse approximation; no calibrated tissue scattering',
                        'light_scope': 'camera-mounted point intensities in renderer radiometric units; no measured scope photometry',
                        'image_conversion': 'fixed exposure, clip linear RGB to[0,1], standard sRGB encoding; RGB channel order'}
