"""Seeded appearance fields on reference material vertices, independent of camera.

These are assigned procedural color/network hypotheses, not measured mucosa or
histological vessel annotations. Store the returned values once, then transport
them by unchanged material vertex IDs; do not recompute on deformed coordinates.
"""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .endoscopic_renderer import mesh_topology_sha256


@dataclass(frozen=True)
class MaterialAppearanceSettings:
    seed: int = 20260911
    base_albedo_rgb: tuple = (.48, .16, .13)
    wavelengths_meter: tuple = (.001, .003, .012)
    relative_amplitudes: tuple = (.03, .08, .15)
    modes_per_scale: int = 6
    network_roots: int = 48
    branches_per_root: int = 3
    branch_length_meter: float = .020
    network_width_meter: float = .00045
    network_rgb_attenuation: tuple = (.20, .40, .32)
    graph_metric_quantum_meter: float = 1e-9

    def __post_init__(self):
        if not isinstance(self.seed,int) or not 0<=self.seed<2**32:
            raise ValueError('Seed must be uint32')
        for x in [self.modes_per_scale,self.branches_per_root]:
            if not isinstance(x,int) or isinstance(x,bool) or x<1: raise ValueError('Positive integer mode/branch counts required')
        if not isinstance(self.network_roots,int) or isinstance(self.network_roots,bool) or self.network_roots<0:
            raise ValueError('Nonnegative integer network root count required')
        color=np.asarray(self.base_albedo_rgb);atten=np.asarray(self.network_rgb_attenuation)
        wave=np.asarray(self.wavelengths_meter);amp=np.asarray(self.relative_amplitudes)
        if (color.shape!=(3,) or atten.shape!=(3,) or not np.isfinite(color).all() or not np.isfinite(atten).all()
            or (color<0).any() or (color>1).any() or (atten<0).any() or (atten>1).any()
            or wave.ndim!=1 or not len(wave) or amp.shape!=wave.shape or not np.isfinite(wave).all()
            or not np.isfinite(amp).all() or (wave<=0).any() or (amp<0).any() or amp.sum()>=1
            or color.max()*(1+amp.sum())>1):
            raise ValueError('Invalid bounded reflectance field parameters')
        if not np.isfinite([self.branch_length_meter,self.network_width_meter,self.graph_metric_quantum_meter]).all() or min(self.branch_length_meter,self.network_width_meter,self.graph_metric_quantum_meter)<=0:
            raise ValueError('Network physical lengths must be positive finite')


def material_vertex_albedo(reference_vertices_meter, triangles, settings=MaterialAppearanceSettings()):
    """Return fixed linear vertex reflectances, network weights, and provenance.

    Dark network paths follow mesh-edge geodesics; diffusion distance is also
    geodesic, preventing Euclidean color bleeding across nearby disconnected
    lumen surfaces. The graph approximation depends on reference mesh resolution.
    No scene camera, lights, image coordinates or frame time enter this function.
    """
    vertices=np.asarray(reference_vertices_meter,dtype=float);faces=np.asarray(triangles)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Finite Nx3 reference vertices required')
    topology=mesh_topology_sha256(len(vertices),faces)
    p=vertices[faces];twice_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    if (twice_area<=0).any():raise ValueError('Degenerate reference triangle')
    rng=np.random.default_rng(settings.seed)
    origin=np.average(vertices,axis=0);xyz=vertices-origin
    modulation=np.zeros(len(vertices))
    for wavelength,amplitude in zip(settings.wavelengths_meter,settings.relative_amplitudes):
        direction=rng.normal(size=(settings.modes_per_scale,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
        phase=rng.uniform(0,2*np.pi,settings.modes_per_scale)
        modulation+=amplitude*np.cos(2*np.pi*(xyz@direction.T)/wavelength+phase).mean(axis=1)
    network=np.zeros(len(vertices));source_ids=[];root_ids=[]
    if settings.network_roots:
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
        if (length<=0).any():raise ValueError('Zero reference graph edge')
        # Fixed integer length units prevent floating addition order from changing
        # shortest-path ties after a rigid translation of a symmetric mesh.
        quantum=settings.graph_metric_quantum_meter
        length_units=np.rint(length/quantum)
        if (length_units<1).any() or length_units.max()*len(vertices)>=2**53:
            raise ValueError('Graph length quantum cannot represent this mesh safely')
        graph=coo_matrix((np.tile(length_units,2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
        mass=np.zeros(len(vertices));np.add.at(mass,faces.ravel(),np.repeat(twice_area/6,3));mass/=mass.sum()
        count=min(settings.network_roots,int((mass>0).sum()))
        root_ids=rng.choice(len(vertices),size=count,replace=False,p=mass)
        for root in root_ids:
            distance,pred=dijkstra(graph,directed=False,indices=int(root),limit=settings.branch_length_meter/quantum,return_predecessors=True)
            distance*=quantum
            candidates=np.flatnonzero(np.isfinite(distance)&(distance>=.65*settings.branch_length_meter))
            if not len(candidates):continue
            delta=vertices[candidates]-vertices[root]
            for _ in range(settings.branches_per_root):
                direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
                endpoint=int(candidates[np.argmax(delta@direction)])
                node=endpoint
                while node!=root:
                    source_ids.append(node);node=int(pred[node])
                    if node<0:raise RuntimeError('Disconnected geodesic predecessor')
                source_ids.append(int(root))
        if source_ids:
            distance=dijkstra(graph,directed=False,indices=np.unique(source_ids),min_only=True,limit=4*settings.network_width_meter/quantum)*quantum
            network=np.exp(-.5*(distance/settings.network_width_meter)**2)
    color=np.asarray(settings.base_albedo_rgb)[None,:]*(1+modulation[:,None])
    color*=1-network[:,None]*np.asarray(settings.network_rgb_attenuation)[None,:]
    return color.astype(np.float32),network.astype(np.float32),{
        'schema':'reference_material_appearance_pilot_v1','settings':asdict(settings),
        'topology_sha256':topology,
        'reference_vertices_sha256':hashlib.sha256(np.asarray(vertices,dtype='<f8',order='C').tobytes()).hexdigest(),
        'reference_origin_meter':origin.tolist(),'network_root_material_ids':np.asarray(root_ids,dtype=int).tolist(),
        'network_path_vertex_count':len(set(source_ids)),
        'scope':'assigned procedural material color; not measured histology; fixed vertex IDs required across states'}
