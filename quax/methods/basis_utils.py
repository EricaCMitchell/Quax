import jax
import jax.numpy as jnp
from jax.lax import fori_loop
import functools

from .ints import compute_f12_oeints
from .energy_utils import symmetric_orthogonalization
from ..integrals import libint_interface

def build_CABS(geom, basis_set, cabs_set, xyz_path, deriv_order, options):
    """
    Builds and returns 
    CABS transformation matrix
    """
    # Make Thread Safe
    threads = libint_interface.get_num_threads()
    libint_interface.set_num_threads(1)

    # Orthogonalize combined basis set
    S_ao_ribs_ribs = compute_f12_oeints(geom, cabs_set, cabs_set, xyz_path, deriv_order, options, True)

    if options['spectral_shift']:
        convergence = 1e-10
        fudge = jnp.asarray(jnp.linspace(0, 1, S_ao_ribs_ribs.shape[0])) * convergence
        shift = jnp.diag(fudge)
        S_ao_ribs_ribs += shift

    C_ribs = symmetric_orthogonalization(S_ao_ribs_ribs, 1.0e-8)

    # Compute the overlap matrix between OBS and RIBS
    S_ao_obs_ribs = compute_f12_oeints(geom, basis_set, cabs_set, xyz_path, deriv_order, options, True)

    _, S, Vt = svd_full(S_ao_obs_ribs @ C_ribs)

    def loop_zero_vals(idx, count):
        count += jax.lax.cond(abs(S[idx]) < 1.0e-6, lambda: 1, lambda: 0)
        return count
    ncabs = fori_loop(0, S.shape[0], loop_zero_vals, S.shape[0])

    V_N = jnp.transpose(Vt[ncabs:, :])

    C_cabs = jnp.dot(C_ribs, V_N)

    libint_interface.set_num_threads(threads)

    return C_cabs

def F_ij(s, m):
    """
    Can be numerically unstable if singular values are degenerate
    """
    F_ij = lambda i, j: jax.lax.cond(i == j, lambda: 0., lambda: 1 / (s[j]**2 - s[i]**2))
    F_fun = jax.vmap(jax.vmap(F_ij, (None, 0)), (0, None))

    indices = jnp.arange(m)

    return F_fun(indices, indices)

@jax.custom_jvp
def svd_full(A):
    return jnp.linalg.svd(A)

@svd_full.defjvp
def svd_full_jvp(primals, tangents):
    A, = primals
    dA, = tangents

    m = A.shape[0]
    n = A.shape[1]

    U, S, Vt = svd_full(A)

    dP = U.T @ dA @ Vt.T

    dS = jnp.diagonal(dP)

    S1 = jnp.diag(S)

    dP1 = dP[:, :m]

    F = F_ij(S, m)

    dU = U @ (F * (dP1 @ S1 + S1 @ dP1.T))

    dD1 = F * (S1 @ dP1 + dP1.T @ S1)

    dD2 = jnp.linalg.inv(S1) @ dP[:, m:] # Can be numerically unstable due to inversion

    dD_left = jnp.concatenate((dD1, dD2.T))
    dD_right = jnp.concatenate((-dD2, jnp.zeros((n-m, n-m))))

    dD = jnp.concatenate((dD_left, dD_right), axis=1)

    dV = Vt.T @ dD

    return (U, S, Vt), (dU, dS, dV.T)
