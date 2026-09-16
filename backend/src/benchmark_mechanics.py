"""Reproducible benchmark fields solved on a curved C3D6 mesh.

The historical generator transports straight-cylinder responses. This opt-in
path instead deforms the FE reference mesh BEFORE assembly. Rendered surface
fields are bilinear interpolants of the coarse FE outer nodes, explicitly
kinematic truth rather than an assertion of tissue realism.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import splu

from .tube import _interp_curve, _tube_basis_grids
from .tube_fea import _load_patch_solver, _interp_periodic_response


def solve_benchmark_fields(geom, cfg, patches: list[dict], peak_fractions: list[float]):
    """Return render-grid responses, full FE state, and numerical diagnostics."""
    Builder, Solver = _load_patch_solver()
    na, nt = cfg.deformation.fea.axial_elements, cfg.deformation.fea.theta_elements
    R, thickness = float(cfg.base_radius), float(cfg.shell_thickness)
    solid = Builder.build_hollow_cylinder_c3d6(
        length=float(cfg.length), inner_radius=R-thickness, outer_radius=R,
        n_axial=na, n_theta=nt, E=cfg.deformation.fea.young_modulus_pa,
        nu=cfg.deformation.fea.poisson_ratio, fix_ends=True)
    # Builder ordering: axial ring, angular node, inner/outer radial node.
    for i in range(na+1):
        s = i / na
        centre = _interp_curve(geom.centerline, s).astype(float)
        right = _interp_curve(geom.frame_right, s).astype(float)
        up = _interp_curve(geom.frame_up, s).astype(float)
        radius_ring = _interp_curve(geom.base_radius, s).astype(float)
        for j in range(nt):
            theta = 2*np.pi*j/nt
            pos = j/nt*len(radius_ring)
            j0 = int(np.floor(pos)) % len(radius_ring)
            a = pos-np.floor(pos)
            radius = (1-a)*radius_ring[j0] + a*radius_ring[(j0+1)%len(radius_ring)]
            radial = np.cos(theta)*right + np.sin(theta)*up
            for layer in (0, 1):
                solid.nodes[(i*nt+j)*2+layer] = centre + (radius-(1-layer)*thickness)*radial
    solid.validate()
    K = Solver.assemble_global_K(solid)
    free = ~solid.fixed.reshape(-1)
    factor = splu(K[free][:,free].tocsc())
    radius = float(np.median(geom.base_radius))
    fields, displacements, loads, reactions, strains, stresses, energies, diagnostics = [], [], [], [], [], [], [], []
    for patch, peak in zip(patches, peak_fractions, strict=True):
        force = np.zeros_like(solid.loads)
        for i in range(1,na):
            s = i/na
            tang = _interp_curve(geom.tangent,s).astype(float)
            right = _interp_curve(geom.frame_right,s).astype(float)
            up = _interp_curve(geom.frame_up,s).astype(float)
            for j in range(nt):
                angle=2*np.pi*j/nt
                dt = (j/nt-patch['theta']+.5)%1-.5
                weight=np.exp(-.5*((s-patch['s'])/patch['sigma_s'])**2-.5*(dt/patch['sigma_theta'])**2)
                radial=np.cos(angle)*right+np.sin(angle)*up
                circum=-np.sin(angle)*right+np.cos(angle)*up
                dr, da, dc=patch['direction']
                direction=dr*radial+da*tang+dc*circum
                force[(i*nt+j)*2+1]=weight*direction/np.linalg.norm(direction)
        force /= np.sum(np.linalg.norm(force,axis=1))
        u=np.zeros(force.size)
        u[free]=factor.solve(force.reshape(-1)[free])
        u=u.reshape(-1,3)
        grid=u.reshape(na+1,nt,2,3)[:,:,1,:]
        surface=_interp_periodic_response(grid,geom.s_norm_grid,geom.theta_norm_grid)
        scale=peak*radius/float(np.max(np.linalg.norm(surface,axis=-1)))
        u*=scale; force*=scale; surface*=scale
        residual=(K@u.reshape(-1)-force.reshape(-1)).reshape(-1,3)
        state=Solver.postprocess_fields(solid,u,reactions=residual)
        rel=float(np.linalg.norm(residual.reshape(-1)[free])/max(np.linalg.norm(force),1e-30))
        if not np.isfinite(rel) or rel>1e-8:
            raise ValueError(f'FE equilibrium failure {rel}')
        fields.append(surface.astype(np.float32)); displacements.append(u)
        loads.append(force); reactions.append(residual); strains.append(state.gauss_strain)
        stresses.append(state.gauss_stress); energies.append(state.elem_strain_energy)
        diagnostics.append({'relative_free_equilibrium_residual':rel,'peak_surface_displacement_over_radius':float(np.max(np.linalg.norm(surface,axis=-1))/radius),
                            'max_abs_gauss_engineering_strain':float(np.max(np.abs(state.gauss_strain))),
                            'applied_force_norm_sum':float(np.linalg.norm(force,axis=1).sum())})
    arrays={'nodes':solid.nodes,'elements_c3d6':solid.elements_c3d6,'fixed':solid.fixed,
            'E':solid.E_c3d6,'nu':solid.nu_c3d6,'basis_displacement':np.stack(displacements),
            'basis_loads':np.stack(loads),'basis_reactions':np.stack(reactions),
            'basis_gauss_strain':np.stack(strains),'basis_gauss_stress':np.stack(stresses),
            'basis_element_energy':np.stack(energies),'K_data':K.data,'K_indices':K.indices,
            'K_indptr':K.indptr,'K_shape':np.array(K.shape)}
    return np.stack(fields), arrays, diagnostics


def prescribed_fields(geom, patches, peaks):
    """Localized vector kinematics independent of the elastic solve."""
    radial,tangent,circum=_tube_basis_grids(geom)
    R=float(np.median(geom.base_radius)); out=[]
    for patch,peak in zip(patches,peaks,strict=True):
        dt=(geom.theta_norm_grid-patch['theta']+.5)%1-.5
        w=np.exp(-.5*((geom.s_norm_grid-patch['s'])/patch['sigma_s'])**2-.5*(dt/patch['sigma_theta'])**2)
        # Exactly fixed end rings, smooth window, with no equilibrium claim.
        w*=np.sin(np.pi*geom.s_norm_grid)**2
        dr,da,dc=patch['direction']; d=dr*radial+da*tangent+dc*circum
        d/=np.linalg.norm(d,axis=-1,keepdims=True)
        f=w[:,:,None]*d
        f*=peak*R/np.max(np.linalg.norm(f,axis=-1))
        f[0]=0; f[-1]=0; out.append(f.astype(np.float32))
    return np.stack(out)
