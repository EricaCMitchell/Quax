import psi4
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from basis_utils import build_basis_set
from integrals_utils import cartesian_product, old_cartesian_product, find_unique_shells
from functools import partial
from jax.experimental import loops
from pprint import pprint
from eri import *

# Define molecule
molecule = psi4.geometry("""
                         0 1
                         H 0.0 0.0 -0.849220457955
                         H 0.0 0.0  0.849220457955
                         units bohr
                         """)

# Get geometry as JAX array
geom = np.asarray(onp.asarray(molecule.geometry()))

basis_name = 'cc-pvdz'
basis_set = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name, puream=0)
basis_dict = build_basis_set(molecule, basis_name)
# Pytree test
#value_flat, value_tree = jax.tree_util.tree_flatten(basis_dict)
#print(value_flat)


max_prim = basis_set.max_nprimitive()
print(max_prim)
biggest_K = max_prim**4
#pprint(basis_dict)
nbf = basis_set.nbf()
nshells = len(basis_dict)
#unique_shell_quartets = find_unique_shells(nshells)

shell_quartets = old_cartesian_product(np.arange(nshells), np.arange(nshells), np.arange(nshells), np.arange(nshells))

def transform_basisdict(basis_dict, max_prim):
    '''
    Make it so all contractions are the same size in the basis dict by padding exp and coef values to 0 and 0?
    Also create 'indices' key which says where along axis the integral should go, but padded with -1's to maximum angular momentum size
    This allows you to pack them neatly into an array, and then worry about redundant computation later.
    '''
    new_dict = basis_dict.copy()
    for i in range(len(basis_dict)):
        current_exp = onp.asarray(basis_dict[i]['exp'])
        new_dict[i]['exp'] = np.asarray(onp.pad(current_exp, (0, max_prim - current_exp.shape[0])))
        current_coef = onp.asarray(basis_dict[i]['coef'])
        new_dict[i]['coef'] = np.asarray(onp.pad(current_coef, (0, max_prim - current_coef.shape[0])))
        idx, size = basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        indices = onp.repeat(idx, size) + onp.arange(size)
    return new_dict

#TODO this is incorrect, mixes 0's and real values together, not what you want
basis_dict = transform_basisdict(basis_dict, max_prim)
pprint(basis_dict)

def preprocess(shell_quartets, basis_dict):
    coeffs = []
    exps = []
    atoms = []
    ams = []
    indices = []
    sizes = []
    for i in range(nshells):
        c1, exp1, atom1_idx, am1, idx1, size1 = onp.asarray(basis_dict[i]['coef']), onp.asarray(basis_dict[i]['exp']), basis_dict[i]['atom'], basis_dict[i]['am'], basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        coeffs.append(c1)
        exps.append(exp1)
        atoms.append(atom1_idx)
        ams.append(am1)
        indices.append(idx1)
        sizes.append(size1)
    return np.asarray(coeffs), np.asarray(exps), np.asarray(atoms), np.asarray(ams), np.asarray(indices), np.asarray(sizes)

coeffs, exps, atoms, am, indices, sizes = preprocess(shell_quartets, basis_dict)
print(am)

def get_indices(shell_quartets, basis_dict):
    '''
    Get all indices of ERIs in (nbf**4,4) array. 
    Record where each shell quartet starts and stops along the first axis of this index array.
    '''
    all_indices = []
    for i in range(nshells):
        idx1, size1 = basis_dict[i]['idx'], basis_dict[i]['idx_stride']
        indices1 = onp.repeat(idx1, size1) + onp.arange(size1)
        for j in range(nshells):
            idx2, size2 = basis_dict[j]['idx'], basis_dict[j]['idx_stride']
            indices2 = onp.repeat(idx2, size2) + onp.arange(size2)
            for k in range(nshells):
                idx3, size3 = basis_dict[k]['idx'], basis_dict[k]['idx_stride']
                indices3 = onp.repeat(idx3, size3) + onp.arange(size3)
                for l in range(nshells):
                    idx4, size4 = basis_dict[l]['idx'], basis_dict[l]['idx_stride']
                    indices4 = onp.repeat(idx4, size4) + onp.arange(size4)
                    indices = old_cartesian_product(indices1,indices2,indices3,indices4)
                    indices = onp.pad(indices, ((0, 81-indices.shape[0]),(0,0)), constant_values=-1)
                    all_indices.append(indices)
    return np.asarray(onp.asarray(all_indices))

indices = get_indices(shell_quartets, basis_dict)
print(indices.shape)

def compute(geom, coeffs, exps, atoms, am, indices):
    #dim_indices = np.repeat(indices, sizes) + np.arange(sizes)

    with loops.Scope() as s:
        @jax.jit
        def primitive(A, B, C, D, aa, bb, cc, dd, coeff, am):
            '''Geometry parameters, exponents, coefficients, angular momentum identifier'''
            # For different 
            # Must (legally) permute arguments so the right arguments appear at function-differentiated indices (argnums in jacfwd)  
            primitive = np.where(coeff == 0, 0.0, 
            np.where(np.allclose(am,np.array([0,0,0,0])), np.pad(eri_ssss(A,B,C,D,aa,bb,cc,dd,coeff).reshape(-1), (0,80)),
            np.where(np.allclose(am,np.array([1,0,0,0])), np.pad(eri_psss(A,B,C,D,aa,bb,cc,dd,coeff).reshape(-1), (0,78)), 
            np.where(np.allclose(am,np.array([0,1,0,0])), np.pad(eri_psss(B,A,C,D,bb,aa,cc,dd,coeff).reshape(-1), (0,78)),
            np.where(np.allclose(am,np.array([0,0,1,0])), np.pad(eri_psss(C,D,A,B,cc,dd,aa,bb,coeff).reshape(-1), (0,78)),
            np.where(np.allclose(am,np.array([0,0,0,1])), np.pad(eri_psss(D,C,B,A,dd,cc,bb,aa,coeff).reshape(-1), (0,78)), 
            np.where(np.allclose(am,np.array([1,1,0,0])), np.pad(eri_ppss(A,B,C,D,aa,bb,cc,dd,coeff).reshape(-1), (0,72)), 
            np.where(np.allclose(am,np.array([0,0,1,1])), np.pad(eri_ppss(C,D,A,B,cc,dd,aa,bb,coeff).reshape(-1), (0,72)), 
            np.where(np.allclose(am,np.array([1,0,1,0])), np.pad(eri_psps(A,B,C,D,aa,bb,cc,dd,coeff).reshape(-1), (0,72)), 
            np.where(np.allclose(am,np.array([0,1,1,0])), np.pad(eri_psps(B,A,C,D,bb,aa,cc,dd,coeff).reshape(-1), (0,72)), 
            np.where(np.allclose(am,np.array([1,0,0,1])), np.pad(eri_psps(A,B,D,C,aa,bb,dd,cc,coeff).reshape(-1), (0,72)), 
            np.where(np.allclose(am,np.array([0,1,0,1])), np.pad(eri_psps(B,A,D,C,bb,aa,dd,cc,coeff).reshape(-1), (0,72)),
            # PPPS is a weird case, it returns a (3,3,3), where each dim is differentiation w.r.t. 0, then 1, then 2
            # You must transpose the result when you permute. 
            np.where(np.allclose(am,np.array([1,1,1,0])), np.pad(eri_ppps(A,B,C,D,aa,bb,cc,dd,coeff).reshape(-1), (0,54)), 
            np.where(np.allclose(am,np.array([1,1,0,1])), np.pad(eri_ppps(A,B,D,C,aa,bb,dd,cc,coeff).reshape(-1), (0,54)), 
            np.where(np.allclose(am,np.array([1,0,1,1])), np.pad(np.transpose(eri_ppps(C,D,A,B,cc,dd,aa,bb,coeff),(2,0,1)).reshape(-1), (0,54)), 
            np.where(np.allclose(am,np.array([0,1,1,1])), np.pad(np.transpose(eri_ppps(D,C,B,A,dd,cc,bb,aa,coeff), (2,1,0)).reshape(-1), (0,54)),
            np.where(np.allclose(am,np.array([1,1,1,1])), eri_pppp(A,B,C,D,aa,bb,cc,dd,coeff).reshape(-1),  0.0
            )))))))))))))))))
            return primitive
        # Computes multiple primitives with same center, angular momentum 
        vectorized_primitive = jax.vmap(primitive, (None,None,None,None,0,0,0,0,0,None))
        ## Computes a contracted integral 
        #@partial(jax.jit, static_argnums=(9))
        @jax.jit
        def contraction(A, B, C, D, aa, bb, cc, dd, coeff, am):
            primitives = vectorized_primitive(A, B, C, D, aa, bb, cc, dd, coeff, am)
            contraction = np.sum(primitives, axis=0)
            return contraction

        indx_array = np.arange(nshells**4).reshape(nshells,nshells,nshells,nshells) 
        #s.G = np.zeros((nbf,nbf,nbf,nbf))
        s.G = np.zeros((nbf+1,nbf+1,nbf+1,nbf+1))
        idx_vec = np.arange(nbf)
        for i in s.range(nshells):
            A = geom[atoms[i]]
            aa = exps[i]
            c1 = coeffs[i]
            ami = am[i]
            idx1 = indices[i]
            for j in s.range(nshells):
                B = geom[atoms[j]]
                bb = exps[j]
                c2 = coeffs[j]
                amj = am[j]
                idx2 = indices[j]
                for k in s.range(nshells):
                    C = geom[atoms[k]]
                    cc = exps[k]
                    c3 = coeffs[k]
                    amk = am[k]
                    idx3 = indices[k]
                    for l in s.range(nshells):
                        D = geom[atoms[l]]
                        dd = exps[l]
                        c4 = coeffs[l]
                        aml = am[l]
                        idx4 = indices[l]

                        exp_combos = cartesian_product(aa,bb,cc,dd)
                        coeff_combos = np.prod(cartesian_product(c1,c2,c3,c4), axis=1)
                        am_vec = np.array([ami, amj, amk, aml]) 
                        # shape matches how far starting index should be extended
                        val = contraction(A,B,C,D, 
                                          exp_combos[:,0], 
                                          exp_combos[:,1], 
                                          exp_combos[:,2],
                                          exp_combos[:,3],
                                          coeff_combos, am_vec)

                        place = indx_array[i,j,k,l]
                        index = indices[place]
                        s.G = jax.ops.index_update(s.G, (index[:,0], index[:,1], index[:,2], index[:,3]), val)

        return s.G[:-1,:-1,:-1,:-1]
#
#G = compute(geom, coeffs, exps, atoms, am, indices)
#
#mints = psi4.core.MintsHelper(basis_set)
#psi_G = np.asarray(onp.asarray(mints.ao_eri()))
#
###print(G)
#G1 = G.flatten()
#G2 = psi_G.flatten()
#for i in range(10000):
#    b = np.allclose(G1[i], G2[i])
#    if b:
#        continue
#    else:
#        if G1[i] != 0.0:
#            print('ERROR', G1[i], G2[i])
#            print(onp.unravel_index([i], (10,10,10,10)))
#
##
#print(np.allclose(psi_G,G))

