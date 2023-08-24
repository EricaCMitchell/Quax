import jax 
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.lax import fori_loop, while_loop
from functools import partial

from .integrals_utils import gaussian_product, boys, binomial_prefactor, factorials, double_factorials, neg_one_pow, cartesian_product, am_leading_indices, angular_momentum_combinations
from .basis_utils import flatten_basis_data, get_nbf

# Useful resources: Fundamentals of Molecular Integrals Evaluation, Fermann, Valeev https://arxiv.org/abs/2007.12057
#                   Gaussian-Expansion methods of molecular integrals Taketa, Huzinaga, O-ohata 

def overlap(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor):
    """
    Computes a single overlap integral. Taketa, Huzinaga, Oohata 2.12
    P = gaussian product of aa,A; bb,B
    PA_pow, PB_pow
        All powers of Pi-Ai or Pi-Bi packed into an array
        [[(Px-Ax)^0, (Px-Ax)^1, ... (Px-Ax)^max_am]
         [(Py-Ay)^0, (Py-Ay)^1, ... (Py-Ay)^max_am]
         [(Pz-Az)^0, (Pz-Az)^1, ... (Pz-Az)^max_am]]
    prefactor = jnp.exp(-aa * bb * jnp.dot(A-B,A-B) / gamma)
    """
    gamma = aa + bb
    prefactor *= (jnp.pi / gamma)**1.5

    wx = overlap_component(la,lb,PA_pow[0],PB_pow[0],gamma)
    wy = overlap_component(ma,mb,PA_pow[1],PB_pow[1],gamma)
    wz = overlap_component(na,nb,PA_pow[2],PB_pow[2],gamma)
    return prefactor * wx * wy * wz

def overlap_component(l1,l2,PAx,PBx,gamma):
    """
    The 1d overlap integral component. Taketa, Huzinaga, Oohata 2.12
    """
    K = 1 + (l1 + l2) // 2  

    def loop_i(arr):
       i, total = arr
       return (i+1, total + binomial_prefactor(2*i,l1,l2,PAx,PBx) * double_factorials[2*i-1] / (2*gamma)**i)

    i_accu, total_sum = while_loop(lambda arr: arr[0] < K, loop_i, (0, 0)) # (i, total)

    return total_sum

def kinetic(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor):
    """
    Computes a single kinetic energy integral.
    """
    gamma = aa + bb
    prefactor *= (jnp.pi / gamma)**1.5
    wx = overlap_component(la,lb,PA_pow[0],PB_pow[0],gamma)
    wy = overlap_component(ma,mb,PA_pow[1],PB_pow[1],gamma)
    wz = overlap_component(na,nb,PA_pow[2],PB_pow[2],gamma)
    wx_plus2 = overlap_component(la,lb+2,PA_pow[0],PB_pow[0],gamma)
    wy_plus2 = overlap_component(ma,mb+2,PA_pow[1],PB_pow[1],gamma)
    wz_plus2 = overlap_component(na,nb+2,PA_pow[2],PB_pow[2],gamma)
    wx_minus2 = overlap_component(la,lb-2,PA_pow[0],PB_pow[0],gamma)
    wy_minus2 = overlap_component(ma,mb-2,PA_pow[1],PB_pow[1],gamma)
    wz_minus2 = overlap_component(na,nb-2,PA_pow[2],PB_pow[2],gamma)

    term1 = bb*(2*(lb+mb+nb)+3) * wx * wy * wz 

    term2 = -2 * bb**2 * (wx_plus2*wy*wz + wx*wy_plus2*wz + wx*wy*wz_plus2)

    term3 = -0.5 * (lb * (lb-1) * wx_minus2 * wy * wz \
                  + mb * (mb-1) * wx * wy_minus2 * wz \
                  + nb * (nb-1) * wx * wy * wz_minus2)
    return prefactor * (term1 + term2 + term3)

def A_array(l1,l2,PA,PB,CP,g,A_vals):

    def loop_i(arr0):
       i_0, r_0, u_0, A_0 = arr0
       Aterm_0 = neg_one_pow[i_0] * binomial_prefactor(i_0,l1,l2,PA,PB) * factorials[i_0]
       r_0 = i_0 // 2

       def loop_r(arr1):
          i_1, r_1, u_1, Aterm_1, A_1 = arr1
          u_1 = (i_1 - 2 * r_1) // 2

          def loop_u(arr2):
             i_2, r_2, u_2, Aterm_2, A_2 = arr2
             I = i_2 - 2 * r_2 - u_2
             tmp = I - u_2
             fact_ratio = 1 / (factorials[r_2] * factorials[u_2] * factorials[tmp])
             Aterm_2 *= neg_one_pow[u_2]  * CP[tmp] * (0.25 / g)**(r_2+u_2) * fact_ratio
             A_2 = A_2.at[I].set(Aterm_2)
             u_2 -= 1
             return (i_2, r_2, u_2, Aterm_2, A_2)

          i_1_, r_1_, u_1_, Aterm_1_, A_1_ = while_loop(lambda arr2: arr2[1] > -1, loop_u, (i_1, r_1, u_1, Aterm_1, A_1))
          r_1_ -= 1
          return (i_1_, r_1_, u_1_, Aterm_1_, A_1_)

       i_0_, r_0_, u_0_, Aterm_0_, A_0_ = while_loop(lambda arr1: arr1[1] > -1, loop_r, (i_0, r_0, u_0, Aterm_0, A_0))
       i_0_ -= 1
       return (i_0_, r_0_, u_0_, A_0_)

    i, r, u, A = while_loop(lambda arr0: arr0[0] > -1, loop_i, (l1 + l2, 0, 0, A_vals)) # (i, r, u, A)

    return A

def potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals):
    """
    Computes a single electron-nuclear attraction integral
    """
    gamma = aa + bb
    prefactor *= -2 * jnp.pi / gamma

    def loop_val(n, val):
      Ax = A_array(la,lb,PA_pow[0],PB_pow[0],Pgeom_pow[n,0,:],gamma,A_vals)
      Ay = A_array(ma,mb,PA_pow[1],PB_pow[1],Pgeom_pow[n,1,:],gamma,A_vals)
      Az = A_array(na,nb,PA_pow[2],PB_pow[2],Pgeom_pow[n,2,:],gamma,A_vals)

      I, J, K, total = 0, 0, 0, 0
      def loop_I(arr0):
         I_0, J_0, K_0, val_0, total_0 = arr0
         J_0 = 0

         def loop_J(arr1):
            I_1, J_1, K_1, val_1, total_1 = arr1
            K_1 = 0

            def loop_K(arr2):
               I_2, J_2, K_2, val_2, total_2 = arr2
               total_2 += Ax[I_2] * Ay[J_2] * Az[K_2] * boys_eval[I_2 + J_2 + K_2, n]
               K_2 += 1
               return (I_2, J_2, K_2, val_2, total_2)

            I_1_, J_1_, K_1_, val_1_, total_1_ = while_loop(lambda arr2: arr2[2] < na + nb + 1, loop_K, (I_1, J_1, K_1, val_1, total_1))
            J_1_ += 1
            return (I_1_, J_1_, K_1_, val_1_, total_1_)

         I_0_, J_0_, K_0_, val_0_, total_0_ = while_loop(lambda arr1: arr1[1] < ma + mb + 1, loop_J, (I_0, J_0, K_0, val_0, total_0))
         I_0_ += 1
         return (I_0_, J_0_, K_0_, val_0_, total_0_)

      I_, J_, K_, val_, total_ = while_loop(lambda arr0: arr0[0] < la + lb + 1, loop_I, (I, J, K, val, total))
      val_ += charges[n] * prefactor * total_
      return val_

    val = fori_loop(0, Pgeom_pow.shape[0], loop_val, 0)
    return val

def oei_arrays(geom, basis, charges):
    """
    Build one electron integral arrays (overlap, kinetic, and potential integrals)
    """
    coeffs, exps, atoms, ams, indices, dims = flatten_basis_data(basis)
    nbf = get_nbf(basis)
    nprim = coeffs.shape[0]
    max_am = jnp.max(ams)
    A_vals = jnp.zeros(2*max_am+1)

    # Save various AM distributions for indexing
    # Obtain all possible primitive quartet index combinations 
    primitive_duets = cartesian_product(jnp.arange(nprim), jnp.arange(nprim))
    STV = jnp.zeros((3,nbf,nbf))

    for n in range(primitive_duets.shape[0]):
       p1,p2 = primitive_duets[n]
       coef = coeffs[p1] * coeffs[p2]
       aa, bb = exps[p1], exps[p2]
       atom1, atom2 = atoms[p1], atoms[p2]
       am1, am2 = ams[p1], ams[p2]
       A, B = geom[atom1], geom[atom2]
       ld1, ld2 = am_leading_indices[am1], am_leading_indices[am2]

       gamma = aa + bb
       prefactor = jnp.exp(-aa * bb * jnp.dot(A-B,A-B) / gamma)
       P = (aa * A + bb * B) / gamma
       # Maximum angular momentum: hard coded
       #max_am = 3 # f function support
       # Precompute all powers up to 2+max_am of Pi-Ai, Pi-Bi.
       # We need 2+max_am since kinetic requires incrementing angluar momentum by +2
       PA_pow = jnp.power(jnp.broadcast_to(P-A, (max_am+3, 3)).T, jnp.arange(max_am+3))
       PB_pow = jnp.power(jnp.broadcast_to(P-B, (max_am+3, 3)).T, jnp.arange(max_am+3))

       # For potential integrals, we need the difference between
       # the gaussian product center P and ALL atoms in the molecule,
       # and then take all possible powers up to 2*max_am.
       # We pre-collect this into a 3d array, and then just pull out what we need via indexing in the loops, so they need not be recomputed.
       # The resulting array has dimensions (atom, cartesian component, power) so index (0, 1, 3) would return (Py - atom0_y)^3
       P_minus_geom = jnp.broadcast_to(P, geom.shape) - geom
       Pgeom_pow = jnp.power(jnp.transpose(jnp.broadcast_to(P_minus_geom, (2*max_am + 1,geom.shape[0],geom.shape[1])), (1,2,0)), jnp.arange(2*max_am + 1))
       # All possible jnp.dot(P-atom,P-atom)
       rcp2 = jnp.einsum('ij,ij->i', P_minus_geom, P_minus_geom)
       # All needed (and unneeded, for am < max_am) boys function evaluations
       boys_arg = jnp.broadcast_to(rcp2 * gamma, (2*max_am+1, geom.shape[0]))
       boys_nu = jnp.tile(jnp.arange(2*max_am+1), (geom.shape[0],1)).T
       boys_eval = boys(boys_nu,boys_arg)

       a, b = 0, 0
       def loop_a(arr0):
          a_0, b_0, oei_0 = arr0
          b_0 = 0

          def loop_b(arr1):
             a_1, b_1, oei_1 = arr1
             # Gather angular momentum and index
             la,ma,na = angular_momentum_combinations[a_1 + ld1]
             lb,mb,nb = angular_momentum_combinations[b_1 + ld2]
             # To only create unique indices, need to have separate indices arrays for i and j.
             i = indices[p1] + a_1
             j = indices[p2] + b_1
             # Compute one electron integrals and add to appropriate index
             overlap_int = overlap(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor) * coef
             kinetic_int = kinetic(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,prefactor) * coef
             potential_int = potential(la,ma,na,lb,mb,nb,aa,bb,PA_pow,PB_pow,Pgeom_pow,boys_eval,prefactor,charges,A_vals) * coef
             oei_1 = oei_1.at[([0,1,2],[i,i,i],[j,j,j])].set((overlap_int, kinetic_int, potential_int))
             b_1 += 1
             return (a_1, b_1, oei_1)

          a_0_, b_0_, oei_0_ = while_loop(lambda arr1: arr1[1] < dims[p2], loop_b, (a_0, b_0, oei_0))
          a_0_ += 1
          return (a_0_, b_0_, oei_0_)

       a_, b_, oei_ = while_loop(lambda arr0: arr0[0] < dims[p1], loop_a, (a, b, STV))

       return oei_

    return STV[0], STV[1], STV[2]

