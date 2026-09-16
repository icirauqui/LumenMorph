"""Analytic coupled-basis checks: total energy is not additive over bases."""
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from backend.src.benchmark_reader import EvaluationDataset


@pytest.fixture
def reader(tmp_path):
    public=tmp_path/'public'; public.mkdir()
    (tmp_path/'DATASET.json').write_text(json.dumps({'condition':'fem'}))
    (public/'intrinsics.json').write_text(json.dumps(dict(width=2,height=2,fx=1.,fy=1.,cx=1.,cy=1.)))
    for stream in ('reference', 'stationary', 'moving'):
        (public/stream).mkdir()
        (public/stream/'frames.json').write_text(json.dumps({'frames':[{},{}]}))
    family=tmp_path/'evaluation/family'; family.mkdir(parents=True)
    # K is SPD; nonorthogonal displacement bases have nonzero energy cross terms.
    stiffness=csr_matrix(np.array([[2.,1.,0.],[1.,3.,0.],[0.,0.,4.]]))
    displacement=np.array([[[1.,0.,0.]], [[1.,1.,0.]]])
    loads=np.array([(stiffness@u.reshape(-1)).reshape(1,3) for u in displacement])
    strain=np.array([[[1.,2.,3.,4.,5.,6.]], [[2.,3.,4.,5.,6.,7.]]])
    np.savez(family/'mechanics.npz', nodes=np.zeros((1,3)), elements_c3d6=np.zeros((0,6),int),
             fixed=np.zeros((1,3),bool), E=np.array([1.]),nu=np.array([.3]),
             basis_displacement=displacement,basis_loads=loads,basis_reactions=np.zeros_like(loads),
             basis_gauss_strain=strain,basis_gauss_stress=2*strain,
             basis_element_energy=np.array([[1.],[3.5]]),
             K_data=stiffness.data,K_indices=stiffness.indices,K_indptr=stiffness.indptr,K_shape=stiffness.shape)
    np.savez(family/'fields.npz',time_coefficients=np.array([[0.,0.],[.5,2.]]))
    return EvaluationDataset(tmp_path)


def test_mechanics_composes_coupled_energy_and_linear_fields(reader):
    state=reader.mechanics(1)
    np.testing.assert_allclose(state['displacement'], [[2.5,2.,0.]])
    np.testing.assert_allclose(state['loads'], [[7.,8.5,0.]])
    np.testing.assert_allclose(state['gauss_strain'], [[4.5,7.,9.5,12.,14.5,17.]])
    np.testing.assert_allclose(state['gauss_stress'], 2*state['gauss_strain'])
    assert state['strain_energy'] == pytest.approx(17.25)
    assert state['strain_energy'] != pytest.approx(.5**2*1.+2.**2*3.5)
    assert state['strain_energy'] == pytest.approx(.5*np.sum(state['displacement']*state['loads']))
    np.testing.assert_array_equal(reader.mechanics(1,'moving')['displacement'],state['displacement'])


def test_static_reference_and_initial_state_zero_nonfem_rejected(reader):
    for frame,stream in ((0,'stationary'), (1,'reference')):
        assert reader.mechanics(frame,stream)['strain_energy'] == 0.
    reader.descriptor['condition']='static'
    state=reader.mechanics(1)
    for field in ('displacement','loads','reactions','gauss_strain','gauss_stress'):
        np.testing.assert_array_equal(state[field],0.)
    reader.descriptor['condition']='nonfem'
    with pytest.raises(ValueError,match='no applied FE mechanics'):
        reader.mechanics(1)
    assert reader.mechanics(1,'reference')['strain_energy']==0.
    with pytest.raises(IndexError):reader.mechanics(-1)
