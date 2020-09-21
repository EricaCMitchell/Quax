import jax 
from jax.config import config; config.update("jax_enable_x64", True)
config.enable_omnistaging()
import jax.numpy as np
from jax.experimental import loops
import psi4
import numpy as onp

from .energy_utils import nuclear_repulsion, partial_tei_transformation, tei_transformation
from .hartree_fock import restricted_hartree_fock

def rccsd(geom, basis, nuclear_charges, charge, return_aux_data=False):
    # Do HF
    E_scf, C, eps, V = restricted_hartree_fock(geom, basis, nuclear_charges, charge, SCF_MAX_ITER=50, return_aux_data=True)

    nelectrons = int(np.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2
    nbf = V.shape[0]
    nvir = nbf - ndocc

    o = slice(0, ndocc)
    v = slice(ndocc, nbf)

    # Transform TEI's to MO basis
    V = tei_transformation(V,C)
    fock_Od = eps[o]
    fock_Vd = eps[v]

    # Save slices of two-electron repulsion integral
    V = np.swapaxes(V, 1,2)
    V = (V[o,o,o,o], V[o,o,o,v], V[o,o,v,v], V[o,v,o,v], V[o,v,v,v], V[v,v,v,v])

    # Oribital energy denominators 
    D = 1.0 / (fock_Od.reshape(-1,1,1,1) + fock_Od.reshape(-1,1,1) - fock_Vd.reshape(-1,1) - fock_Vd)
    d = 1.0 / (fock_Od.reshape(-1,1) - fock_Vd)

    # Initial Amplitudes
    T1 = np.zeros((ndocc,nvir))
    T2 = D*V[2]

    # Pre iterations
    CC_MAX_ITER = 30
    iteration = 0
    E_ccsd = 1.0
    E_old = 0.0
    while abs(E_ccsd - E_old)  > 1e-9:
        E_old = E_ccsd * 1
        T1, T2 = rccsd_iter(T1, T2, V, d, D, ndocc, nvir)
        E_ccsd = rccsd_energy(T1,T2,V[2])
        iteration += 1
        if iteration == CC_MAX_ITER:
            break

    #print("CCSD Correlation Energy:   ", E_ccsd)
    #print("CCSD Total Energy:         ", E_ccsd + E_scf)
    if return_aux_data:
        return E_scf + E_ccsd, T1, T2, V, fock_Od, fock_Vd
    else:
        return E_scf + E_ccsd

@jax.jit
def rccsd_energy(T1, T2, Voovv):
    E_ccsd = 0.0
    E_ccsd -= np.einsum('lc, kd, klcd -> ', T1, T1, Voovv, optimize = 'optimal')
    E_ccsd -= np.einsum('lkcd, klcd -> ', T2, Voovv, optimize = 'optimal')
    E_ccsd += 2.0*np.einsum('klcd, klcd -> ', T2, Voovv, optimize = 'optimal')
    E_ccsd += 2.0*np.einsum('lc, kd, lkcd -> ', T1, T1, Voovv, optimize = 'optimal')
    return E_ccsd

@jax.jit
def rccsd_iter(T1, T2, V, d, D, ndocc, nvir):
    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V

    newT1 = np.zeros(T1.shape)
    newT2 = np.zeros(T2.shape)

    # T1 equation
    newT1 -= np.einsum('kc, icka -> ia', T1, Vovov, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, kica -> ia', T1, Voovv, optimize = 'optimal')
    newT1 -= np.einsum('kicd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += 2.0*np.einsum('ikcd, kadc -> ia', T2, Vovvv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('klac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += np.einsum('lkac, klic -> ia', T2, Vooov, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, la, lkic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 -= np.einsum('kc, id, kadc -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += 2.0*np.einsum('kc, id, kacd -> ia', T1, T1, Vovvv, optimize = 'optimal')
    newT1 += np.einsum('kc, la, klic -> ia', T1, T1, Vooov, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, ilad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, liad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('kc, liad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('ic, lkad, lkcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('ic, lkad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('la, ikdc, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('la, ikcd, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')
    newT1 += np.einsum('kc, id, la, lkcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += -2.0*np.einsum('kc, id, la, klcd -> ia', T1, T1, T1, Voovv, optimize = 'optimal')
    newT1 += 4.0*np.einsum('kc, ilad, klcd -> ia', T1, T2, Voovv, optimize = 'optimal')

    # T2 equation
    newT2 += Voovv
    newT2 += np.einsum('ic, jd, cdab -> ijab', T1, T1, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('ijcd, cdab -> ijab', T2, Vvvvv, optimize = 'optimal')
    newT2 += np.einsum('ka, lb, ijkl -> ijab', T1, T1, Voooo, optimize = 'optimal')
    newT2 += np.einsum('klab, ijkl -> ijab', T2, Voooo, optimize = 'optimal')
    newT2 -= np.einsum('ic, jd, ka, kbcd -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 -= np.einsum('ic, jd, kb, kadc -> ijab', T1, T1, T1, Vovvv, optimize = 'optimal')
    newT2 += np.einsum('ic, ka, lb, lkjc -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += np.einsum('jc, ka, lb, klic -> ijab', T1, T1, T1, Vooov, optimize = 'optimal')
    newT2 += np.einsum('klac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('lkac, ijdb, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljdb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ikac, ljbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ikac, jlbd, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kiac, ljbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('kiac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijac, lkbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += -2.0*np.einsum('ijac, klbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('kjac, ildb, lkcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += 4.0*np.einsum('ikac, jlbd, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ijdc, lkab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ic, jd, ka, lb, klcd -> ijab', T1, T1, T1, T1, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ic, jd, lkab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    newT2 += np.einsum('ka, lb, ijdc, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO = -np.einsum('kb, jika -> ijab', T1, Vooov, optimize = 'optimal')
    P_OVVO += np.einsum('jc, icab -> ijab', T1, Vovvv, optimize = 'optimal')
    P_OVVO -= np.einsum('kiac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO -= np.einsum('ic, ka, kjcb -> ijab', T1, T1, Voovv, optimize = 'optimal')
    P_OVVO -= np.einsum('ic, kb, jcka -> ijab', T1, T1, Vovov, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('ikac, kjcb -> ijab', T2, Voovv, optimize = 'optimal')
    P_OVVO -= np.einsum('ikac, jckb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO -= np.einsum('kjac, ickb -> ijab', T2, Vovov, optimize = 'optimal')
    P_OVVO -= 2.0*np.einsum('lb, ikac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += np.einsum('lb, kiac, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= np.einsum('jc, ikdb, kacd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= np.einsum('jc, kiad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= np.einsum('jc, ikad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += np.einsum('jc, lkab, lkic -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += np.einsum('lb, ikac, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= np.einsum('ka, ijdc, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += np.einsum('ka, ilcb, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('jc, ikad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO -= np.einsum('kc, ijad, kbdc -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += 2.0*np.einsum('kc, ijad, kbcd -> ijab', T1, T2, Vovvv, optimize = 'optimal')
    P_OVVO += np.einsum('kc, ilab, kljc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO -= 2.0*np.einsum('kc, ilab, lkjc -> ijab', T1, T2, Vooov, optimize = 'optimal')
    P_OVVO += np.einsum('jkcd, ilab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')
    P_OVVO -= 2.0*np.einsum('kc, jd, ilab, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += np.einsum('kc, jd, ilab, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('kc, la, ijdb, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += np.einsum('kc, la, ijdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += np.einsum('ic, ka, ljbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('ic, ka, jlbd, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += np.einsum('ic, ka, ljdb, lkcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += np.einsum('ic, lb, kjad, klcd -> ijab', T1, T1, T2, Voovv, optimize = 'optimal')
    P_OVVO += -2.0*np.einsum('ikdc, ljab, klcd -> ijab', T2, T2, Voovv, optimize = 'optimal')

    newT2 += P_OVVO + np.transpose(P_OVVO, (1,0,3,2))

    newT1 *= d
    newT2 *= D
    return newT1, newT2

