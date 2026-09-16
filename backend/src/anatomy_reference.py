"""Physical lumen-mask import and explicitly approximate anatomy diagnostics.

This is an anatomy import/QA component, not a tissue or camera-path generator.
MHA coordinate convention: physical_xyz = origin + direction @ (index_xyz*spacing).
Surface extraction and skeleton diagnostics require optional scikit-image.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

@dataclass(frozen=True)
class LumenMask:
    values_zyx: np.ndarray
    spacing_xyz_mm: np.ndarray
    origin_xyz_mm: np.ndarray
    direction: np.ndarray
    header: dict

    def physical_xyz(self, index_zyx):
        x = np.asarray(index_zyx, dtype=float)[..., ::-1]
        return (x * self.spacing_xyz_mm) @ self.direction.T + self.origin_xyz_mm


def read_mha_lumen(path):
    """Read uncompressed local binary uint8 MHA without guessing lost metadata."""
    path = Path(path)
    header = {}
    with path.open('rb') as f:
        for _ in range(128):
            line = f.readline(4096)
            if not line or len(line) >= 4096:
                raise ValueError('invalid or unsupported MHA header')
            key, sep, value = line.decode('ascii').partition('=')
            if not sep:
                raise ValueError('invalid MHA header field')
            header[key.strip()] = value.strip()
            if key.strip() == 'ElementDataFile':
                offset = f.tell()
                break
        else:
            raise ValueError('MHA header too long')
    for key, expected in {'NDims':'3','BinaryData':'True','CompressedData':'False',
                          'ElementType':'MET_UCHAR','ElementDataFile':'LOCAL'}.items():
        if header.get(key) != expected:
            raise ValueError(f'unsupported MHA {key}: {header.get(key)}')
    dims = np.fromstring(header.get('DimSize',''), sep=' ', dtype=int)
    spacing = np.fromstring(header.get('ElementSpacing',''), sep=' ')
    origin = np.fromstring(header.get('Offset',''), sep=' ')
    direction = np.fromstring(header.get('TransformMatrix',''), sep=' ')
    if dims.shape != (3,) or np.any(dims <= 0):
        raise ValueError('invalid MHA dimensions')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('invalid MHA spacing')
    if origin.shape != (3,) or not np.all(np.isfinite(origin)) or direction.shape != (9,):
        raise ValueError('invalid MHA affine')
    direction = direction.reshape(3,3)
    if not np.all(np.isfinite(direction)) or not np.allclose(direction.T@direction,np.eye(3),atol=1e-7):
        raise ValueError('MHA direction must be orthonormal')
    if path.stat().st_size != offset + int(np.prod(dims)):
        raise ValueError('MHA payload size disagrees with dimensions')
    a = np.memmap(path, mode='r', dtype=np.uint8, offset=offset, shape=tuple(dims[::-1]))
    return LumenMask(a,spacing,origin,direction,header)


def binary_summary(mask):
    """Report complete mask components; never silently discard fragments."""
    a = mask.values_zyx
    labels, counts = np.unique(a,return_counts=True)
    if not set(labels.tolist()).issubset({0,1}):
        raise ValueError('lumen mask must contain only background0 and foreground1')
    components, n = ndimage.label(a,structure=np.ones((3,3,3),dtype=bool))
    volumes = np.bincount(components.ravel())[1:]
    occupied = int(volumes.sum())
    if not occupied:
        raise ValueError('empty lumen mask')
    boundary = sum(np.count_nonzero(np.take(a,i,axis=k)) for k in range(3) for i in (0,-1))
    return dict(shape_zyx=list(a.shape),label_counts=dict(zip(map(str,labels.tolist()),counts.tolist())),
                foreground_voxels=occupied,volume_mm3=float(occupied*np.prod(mask.spacing_xyz_mm)),
                components_26=int(n),largest_component_fraction=float(volumes.max()/occupied),
                component_voxel_counts_descending=sorted(volumes.tolist(),reverse=True),
                boundary_face_foreground_count=int(boundary))


def lumen_surface(mask,step_size=1):
    """Marching cubes at0.5, no smoothing/capping; respect full physical affine."""
    from skimage.measure import marching_cubes
    if not isinstance(step_size,int) or step_size < 1:
        raise ValueError('step_size must be a positive integer')
    v,f,_,_ = marching_cubes(mask.values_zyx,level=.5,step_size=step_size,
                              allow_degenerate=False,gradient_direction='descent')
    v = mask.physical_xyz(v)
    # Coordinate permutation and reflected direction may reverse winding.
    signed = np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6
    if signed < 0:
        f = f[:,[0,2,1]]
    return v,f


def surface_summary(vertices,faces):
    v = np.asarray(vertices,dtype=float); f = np.asarray(faces,dtype=np.int64)
    tri=v[f]
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
    unique_edges,counts=np.unique(edges,axis=0,return_counts=True)
    adjacency=coo_matrix((np.ones(len(unique_edges)),(unique_edges[:,0],unique_edges[:,1])),shape=(len(v),len(v))).tocsr()
    component_count,component=connected_components(adjacency,directed=False)
    component_vertices=np.bincount(component)
    closed=not np.any(counts!=2)
    vol=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    return dict(vertices=len(v),triangles=len(f),area_mm2=float(area.sum()),signed_volume_mm3=float(vol),
                boundary_edges=int(np.count_nonzero(counts==1)),nonmanifold_edges=int(np.count_nonzero(counts>2)),
                enclosed_volume_mm3=float(vol) if closed else None,volume_interpretation="oriented surface integral; volume valid only for closed non-self-intersecting consistently oriented boundary",
                surface_components=int(component_count),largest_surface_component_vertex_fraction=float(component_vertices.max()/len(v)),
                euler_characteristic=int(len(v)-len(unique_edges)+len(f)),
                zero_area_triangles=int(np.count_nonzero(area==0)),bbox_min_mm=v.min(axis=0).tolist(),
                bbox_max_mm=v.max(axis=0).tolist(),self_intersection_checked=False)


def skeleton_path_diagnostics(mask,target_spacing_mm=2.5):
    """Approximate medial path; not a validated whole-colon centerline.

    Nearest-neighbor resampling, 3D Lee thinning, then two geodesic sweeps on
    the largest skeleton graph component. Branches/cycles and lost coverage
    are reported; a branched graph diameter is not claimed exact.
    """
    from skimage.morphology import skeletonize
    if not np.isfinite(target_spacing_mm) or target_spacing_mm<=0:
        raise ValueError('target spacing must be positive')
    old=np.asarray(mask.values_zyx.shape)
    new=np.maximum(2,np.rint((old-1)*mask.spacing_xyz_mm[::-1]/target_spacing_mm).astype(int)+1)
    factors=(new-1)/(old-1)
    # grid_mode=False aligns endpoint voxel centers; actual spacing is explicit.
    small=ndimage.zoom(mask.values_zyx,new/old,order=0,prefilter=False).astype(bool)
    actual=(old-1)*mask.spacing_xyz_mm[::-1]/(np.asarray(small.shape)-1)
    distance=ndimage.distance_transform_edt(small,sampling=actual)
    skeleton=skeletonize(small,method='lee')
    points=np.argwhere(skeleton)
    if len(points)<2:raise ValueError('insufficient skeleton support')
    ids=np.full(small.shape,-1,dtype=np.int32);ids[tuple(points.T)]=np.arange(len(points))
    rows=[];cols=[];weights=[]
    for dz in (-1,0,1):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                shift=np.array([dz,dy,dx])
                if not np.any(shift):continue
                q=points+shift;valid=np.all((q>=0)&(q<np.asarray(small.shape)),axis=1)
                src=np.flatnonzero(valid);dest=ids[tuple(q[valid].T)];ok=dest>=0
                rows.extend(src[ok]);cols.extend(dest[ok]);weights.extend([float(np.linalg.norm(shift*actual))]*int(ok.sum()))
    graph=coo_matrix((weights,(rows,cols)),shape=(len(points),len(points))).tocsr()
    n,groups=connected_components(graph,directed=False);sizes=np.bincount(groups);group=int(np.argmax(sizes));support=np.flatnonzero(groups==group)
    d=dijkstra(graph,directed=False,indices=int(support[0]));start=int(support[np.argmax(d[support])])
    d,prev=dijkstra(graph,directed=False,indices=start,return_predecessors=True);end=int(support[np.argmax(d[support])]);path=[end]
    while path[-1]!=start:
        parent=int(prev[path[-1]])
        if parent<0 or len(path)>len(points):raise ValueError('invalid medial predecessor path')
        path.append(parent)
    path=np.array(path[::-1]);p=points[path]
    xyz=(p[:,::-1]*actual[::-1])@mask.direction.T+mask.origin_xyz_mm
    ds=np.linalg.norm(np.diff(xyz,axis=0),axis=1);rad=distance[tuple(p.T)]
    degree=np.diff(graph.indptr)
    report=dict(method='nearest-resampled Lee skeleton; two-sweep geodesic path on largest26-neighbor component',
                anatomical_centerline_validated=False,target_spacing_mm=float(target_spacing_mm),actual_spacing_zyx_mm=actual.tolist(),
                skeleton_voxels=len(points),skeleton_components=int(n),largest_skeleton_fraction=float(len(support)/len(points)),
                degree1_endpoints=int(np.count_nonzero(degree[support]==1)),degree_gt2_branch_voxels=int(np.count_nonzero(degree[support]>2)),
                graph_cycle_rank=int(graph[support][:,support].nnz//2-len(support)+1),path_voxels=len(path),path_length_mm=float(ds.sum()),
                endpoint_distance_mm=float(np.linalg.norm(xyz[-1]-xyz[0])),
                path_tortuosity=float(ds.sum()/np.linalg.norm(xyz[-1]-xyz[0])),
                inscribed_radius_mm_quantiles=np.percentile(rad,[0,5,50,95,100]).tolist(),
                radius_interpretation='distance to nearest background, not section area-equivalent lumen radius',
                fold_quality='not resolved/validated by this coarse medial-path diagnostic')
    return report,xyz,rad
