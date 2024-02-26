"""
Test energy computations
"""
import quax
import pytest
import numpy as np

### Reference Energies from Psi4
"""
molecule = psi4.geometry(
0 1
O   -0.000007070942     0.125146536460     0.000000000000
H   -1.424097055410    -0.993053750648     0.000000000000
H    1.424209276385    -0.993112599269     0.000000000000
units bohr
)

psi4.set_options({'basis': basis_name,
                  'scf_type': 'pk',
                  'mp2_type':'conv',
                  'e_convergence': 1e-10,
                  'd_convergence':1e-10,
                  'puream': 0,
                  'points':5,
                  'fd_project':False})
"""

basis_name = 'sto-3g'

def test_hartree_fock(method='hf'):
    quax_e = quax.core.energy("geom.xyz", basis_name, method)
    assert np.allclose(-74.96329133394005, quax_e)

def test_mp2(method='mp2'):
    quax_e = quax.core.energy("geom.xyz", basis_name, method)
    assert np.allclose(-74.99898194997418, quax_e)

def test_ccsd(method='ccsd'):
    quax_e = quax.core.energy("geom.xyz", basis_name, method)
    assert np.allclose(-75.01290672128442, quax_e)

def test_ccsd_t(method='ccsd(t)'):
    quax_e = quax.core.energy("geom.xyz", basis_name, method)
    assert np.allclose(-75.01297302474235, quax_e)
