"""Material transport and continuous visibility for the opt-in forward renderer."""
from __future__ import annotations
import numpy as np
from .endoscopic_renderer import _runtime, mesh_topology_sha256


def _visibility_distances(vertices, triangles, C_world, points, threads):
    """BVH queries at actual material endpoints, never interpolated depth pixels."""
    mi, dr = _runtime(threads)
    mesh = mi.Mesh('target_geometry', len(vertices), len(triangles))
    parameters = mi.traverse(mesh)
    parameters['vertex_positions'] = mi.Float(vertices.reshape(-1))
    parameters['faces'] = mi.UInt(triangles.astype(np.uint32).reshape(-1))
    parameters.update()
    scene = mi.load_dict({'type':'scene','surface':mesh})
    delta = points-C_world
    lengths = np.linalg.norm(delta,axis=1)
    directions = delta/lengths[:,None]
    ray = mi.Ray3f(mi.Point3f(*map(float,C_world)),mi.Vector3f(directions.T))
    hit = scene.ray_intersect_preliminary(ray)
    dr.eval(hit.t)
    return np.array(hit.t).astype(float), lengths


def material_flow(source_truth, target_vertices_world, triangles, target_camera,
                  target_R_wc, target_C_world, *, distance_atol_meter=1e-7,
                  distance_rtol=1e-5, threads=4, target_sensor_mask=None):
    """Transport exact triangle/barycentric identities into a central target camera.

    Pixel centers use u-right/v-down. Source/target image sizes may differ.
    Projection validity is separate from image bounds and continuous occlusion.
    The target mesh is quantized to float32 exactly as in render_endoscopic.
    Classification tolerance is an explicit numerical visibility tolerance, not
    a physical uncertainty estimate. Invalid source pixels have no classification.
    """
    vertices=np.asarray(target_vertices_world,dtype=np.float32)
    faces=np.asarray(triangles)
    R=np.asarray(target_R_wc,dtype=float); C=np.asarray(target_C_world,dtype=float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
        or faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer)
        or not len(faces) or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Invalid target mesh')
    points = vertices[faces].astype(float)
    cross = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
    if np.any(np.linalg.norm(cross, axis=1) <= 0):
        raise ValueError('Degenerate triangles after renderer quantization')
    identity = np.asarray(source_truth.get('mesh_topology_sha256', ''))
    if identity.shape != () or str(identity) != mesh_topology_sha256(len(vertices), faces):
        raise ValueError('Source material topology identity does not match target connectivity/order')
    if (R.shape!=(3,3) or C.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(C).all()
        or not np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0)
        or not np.isclose(np.linalg.det(R),1.,atol=1e-10,rtol=0)):
        raise ValueError('Invalid target camera pose')
    if not np.isfinite([distance_atol_meter,distance_rtol]).all() or min(distance_atol_meter,distance_rtol)<0:
        raise ValueError('Invalid visibility tolerances')
    valid=np.asarray(source_truth['valid'],dtype=bool)
    if 'observation_valid' in source_truth:
        observed=np.asarray(source_truth['observation_valid'])
        if observed.dtype!=bool or observed.shape!=valid.shape or np.any(observed & ~valid):
            raise ValueError('Observation support must be boolean subset of geometric validity')
        valid=observed
    triangle_ids=np.asarray(source_truth['triangle_id'])
    barycentric=np.asarray(source_truth['material_barycentric'],dtype=float)
    if (valid.ndim!=2 or triangle_ids.shape!=valid.shape or barycentric.shape!=(*valid.shape,3)
        or not np.issubdtype(triangle_ids.dtype,np.integer)):
        raise ValueError('Invalid source material arrays')
    ids=triangle_ids[valid]; bary=barycentric[valid]
    if (np.any(ids<0) or np.any(ids>=len(faces)) or not np.isfinite(bary).all()
        or np.any(bary < -1e-5) or not np.allclose(bary.sum(axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError('Invalid source material identities')
    points=np.sum(bary[:,:,None]*vertices[faces[ids]].astype(float),axis=1)
    local=(points-C)@R
    destination,projectable=target_camera.project(local)
    rows,cols=np.nonzero(valid)
    flow=destination-np.c_[cols+.5,rows+.5]
    inside=projectable & target_camera.contains(destination)
    masked=np.zeros(len(points),dtype=bool)
    if target_sensor_mask is not None:
        sensor=np.asarray(target_sensor_mask)
        if sensor.dtype!=bool or sensor.shape!=(target_camera.height,target_camera.width):
            raise ValueError('Target sensor support shape/type mismatch')
        indices=np.floor(destination[inside]).astype(int)
        masked[inside]=~sensor[indices[:,1],indices[:,0]]
    inside_sensor=inside & ~masked
    hit=np.full(len(points),np.inf); distances=np.linalg.norm(points-C,axis=1)
    if inside_sensor.any():
        hit[inside_sensor],distances[inside_sensor]=_visibility_distances(vertices,faces,C,points[inside_sensor],threads)
    tolerance=distance_atol_meter+distance_rtol*distances
    visible=inside_sensor & np.isfinite(hit) & (np.abs(hit-distances)<=tolerance)
    occluded=inside_sensor & np.isfinite(hit) & (hit < distances-tolerance)
    masks={'flow_valid':projectable,'visible':visible,'occluded':occluded,
           'out_of_frame':projectable & ~inside,'behind_camera':local[:,2]<=0,
           'outside_optical_domain':(local[:,2]>0) & ~projectable,
           'masked_by_sensor':masked,'unresolved':inside_sensor & ~visible & ~occluded}
    result={}
    for key,values in masks.items():
        result[key]=np.zeros(valid.shape,dtype=bool); result[key][valid]=values
    for key,values,n in [('flow_uv',flow,2),('target_uv',destination,2),('target_position_world',points,3)]:
        result[key]=np.full((*valid.shape,n),np.nan); result[key][valid]=values
    return result
