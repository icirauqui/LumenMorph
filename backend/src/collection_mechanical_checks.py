"""Manufactured C3D6 checks with analytic, implementation-independent targets."""
import numpy as np
from .tube_fea import _load_patch_solver


def affine_checks():
    _,solver=_load_patch_solver()
    x=np.array([[0,0,-1],[1.3,0,-1],[.2,1.1,-1],[.1,.05,1],[1.4,.05,1],[.3,1.15,1]],float)
    E,nu=900000.,.28
    lam=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu))
    cases={'affine':np.array([[.001,.002,-.001],[.003,-.002,.0005],[.001,.002,.003]]),
           'translation':np.zeros((3,3)),
           'infinitesimal_rotation':np.array([[0,-.002,.001],[.002,0,-.003],[-.001,.003,0]])}
    K=solver._element_stiffness_c3d6(x,E,nu);results={}
    for name,A in cases.items():
        u=(x@A.T+np.array([.1,-.2,.3])).ravel()
        strain,stress,_,energy,volume=solver._element_gauss_state(x,u,E,nu,6)
        eps=.5*(A+A.T);sig=2*mu*eps+lam*np.trace(eps)*np.eye(3)
        e=np.array([eps[0,0],eps[1,1],eps[2,2],2*eps[0,1],2*eps[1,2],2*eps[0,2]])
        t=np.array([sig[0,0],sig[1,1],sig[2,2],sig[0,1],sig[1,2],sig[0,2]])
        es=float(np.max(np.abs(strain-e)));ss=float(np.max(np.abs(stress-t)))
        if name=='affine':
            er=es/np.max(np.abs(e));sr=ss/np.max(np.abs(t));passed=max(er,sr)<=1e-10
            en=.5*u@K@u;energy_error=abs(en-float(energy.sum()))/float(energy.sum());passed &= energy_error<=1e-10
        else:
            er=es;sr=ss/E;energy_error=float(abs(u@K@u))/(E*np.dot(u,u));passed=max(er,sr,energy_error)<=1e-11
        results[name]={'strain_error':er,'stress_error':sr,'energy_error':energy_error,'passed':bool(passed)}
    symmetry=float(np.linalg.norm(K-K.T)/np.linalg.norm(K))
    return {'schema':'c3d6_manufactured_verification_v1','cases':results,'relative_symmetry':symmetry,
            'passed':all(x['passed'] for x in results.values()) and symmetry<1e-12,
            'scope':'Affine displacement/strain/stress on skew prism; rigid translation and infinitesimal rotation. Not finite-rotation mechanics.'}
