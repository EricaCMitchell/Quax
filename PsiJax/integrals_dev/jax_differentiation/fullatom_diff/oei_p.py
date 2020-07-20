"""
All one-electron integrals over p functions
(p|s) (p|p)
Uses the following equations for promoting angular momentum:
(a + 1i | b) = 1/2alpha * (d/dAi (a|b) + ai (a - 1i | b))
(a | b + 1i) = 1/2beta  * (d/dBi (a|b) + bi (a | b - 1i))
where i is a cartesian component of the gaussian
"""
import jax
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=500)
from oei_s import * 

def overlap_ps(A, C, alpha_bra, alpha_ket, c1, c2):
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_bra = 1 / (2 * alpha_bra)
    first_term = jax.jacrev(overlap_ss, 0)(A, C, alpha_bra, alpha_ket, c1, c2)
    return oot_alpha_bra * first_term

def overlap_pp(A, C, alpha_bra, alpha_ket, c1, c2): 
    """
    Computes and returns the following integrals as a vector: 
    (px|s) (py|s) (pz|s)
    """
    oot_alpha_ket = 1 / (2 * alpha_ket)
    first_term = jax.jacrev(overlap_ps, 1)(A, C, alpha_bra, alpha_ket, c1, c2)
    return oot_alpha_ket * first_term


