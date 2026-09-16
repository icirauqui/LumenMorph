"""NumPy-only attachment of frozen TetP2 states to a fixed triangle surface.

A solver-specific exporter supplies the explicit scalar-to-vector DOF map. This
module never assumes component interleaving, solves equilibrium, interpolates
load steps, or equates planar render interpolation with curved FE evaluation.
All lengths/displacements are in metres after explicit state scaling.
"""
import numpy as np


class P2RenderMapping:
    def __init__(self, reference_vertices_meter, triangles, facet_vertex_weights,
                 facet_scalar_dofs, scalar_to_vector_dofs):
        X=np.array(reference_vertices_meter,dtype=float,copy=True)
        T=np.array(triangles,copy=True);W=np.array(facet_vertex_weights,dtype=float,copy=True)
        D=np.array(facet_scalar_dofs,copy=True);I=np.array(scalar_to_vector_dofs,copy=True)
        if (X.ndim!=2 or X.shape[1]!=3 or not len(X) or not np.isfinite(X).all()
            or W.ndim!=3 or W.shape[2]!=10 or W.shape[0]*W.shape[1]!=len(X)
            or not np.isfinite(W).all() or not np.allclose(W.sum(-1),1.,rtol=0,atol=1e-12)
            or D.shape!=(W.shape[0],10) or not np.issubdtype(D.dtype,np.integer)
            or I.ndim!=2 or I.shape[1]!=3 or not np.issubdtype(I.dtype,np.integer)
            or not len(I) or D.min()<0 or D.max()>=len(I)
            or I.min()<0 or not np.array_equal(np.sort(I.ravel()),np.arange(I.size))
            or T.ndim!=2 or T.shape[1]!=3 or not len(T)
            or not np.issubdtype(T.dtype,np.integer) or T.min()<0 or T.max()>=len(X)):
            raise ValueError('Invalid fixed P2 render mapping or explicit vector layout')
        parents=T//W.shape[1]
        if np.any(parents!=parents[:,:1]):
            raise ValueError('Each rendered triangle must retain one parent facet')
        for a in (X,T,W,D,I):a.flags.writeable=False
        self.reference_vertices_meter=X;self.triangles=T;self.weights=W
        self.facet_scalar_dofs=D;self.scalar_to_vector_dofs=I

    def scalar_displacements_meter(self, dimensionless_displacement, length_scale_meter):
        q=np.asarray(dimensionless_displacement,dtype=float)
        if (q.shape!=(self.scalar_to_vector_dofs.size,) or not np.isfinite(q).all()
            or not np.isfinite(length_scale_meter) or length_scale_meter<=0):
            raise ValueError('Exact displacement-only vector and positive physical scale required')
        with np.errstate(over='ignore',invalid='ignore'):
            u=q[self.scalar_to_vector_dofs]*length_scale_meter
        if not np.isfinite(u).all():raise ValueError('Physical displacement overflow')
        return u

    def vertices_meter(self, dimensionless_displacement, length_scale_meter):
        u=self.scalar_displacements_meter(dimensionless_displacement,length_scale_meter)
        with np.errstate(over='ignore',invalid='ignore'):
            moved=self.reference_vertices_meter+np.einsum('fvi,fij->fvj',self.weights,u[self.facet_scalar_dofs]).reshape(-1,3)
        if not np.isfinite(moved).all():raise ValueError('Mapped vertex overflow')
        return moved

    def composed_material_weights(self, triangle_ids, triangle_barycentrics):
        ids=np.asarray(triangle_ids);b=np.asarray(triangle_barycentrics,dtype=float)
        if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)
            or b.shape!=(len(ids),3) or not np.isfinite(b).all()
            or (len(ids) and (ids.min()<0 or ids.max()>=len(self.triangles)))
            or np.any(b < -1e-5) or not np.allclose(b.sum(1),1.,atol=1e-6,rtol=0)):
            raise ValueError('Valid triangle identities and admissible barycentrics required')
        tri=self.triangles[ids];facet=tri[:,0]//self.weights.shape[1]
        weights=np.einsum('pi,pij->pj',b,self.weights.reshape(-1,10)[tri])
        reference=np.einsum('pi,pij->pj',b,self.reference_vertices_meter[tri])
        return reference,self.facet_scalar_dofs[facet],weights

    def material_points_meter(self, triangle_ids, triangle_barycentrics,
                              dimensionless_displacement, length_scale_meter):
        """Float64 planar material points, before renderer float32 quantization."""
        X,dofs,W=self.composed_material_weights(triangle_ids,triangle_barycentrics)
        u=self.scalar_displacements_meter(dimensionless_displacement,length_scale_meter)
        with np.errstate(over='ignore',invalid='ignore'):
            p=X+np.einsum('pi,pij->pj',W,u[dofs])
        if not np.isfinite(p).all():raise ValueError('Mapped material point overflow')
        return p
