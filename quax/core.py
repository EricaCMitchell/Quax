import jax 
from jax import jacfwd
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .integrals import libint_interface
from .methods.hartree_fock import restricted_hartree_fock
from .methods.mp2 import restricted_mp2
from .methods.mp2f12 import restricted_mp2_f12
from .methods.ccsd import rccsd
from .methods.ccsd_t import rccsd_t
from .utils import get_required_deriv_vecs, n_frozen_core

def check_options(options):
    """
    Checks user-supplied keyword options and assigns them

    Parameters
    ----------
    options : dict
        Dictionary of options controlling electronic structure code parameters

    Returns
    -------
    keyword_options : dict
        Dictionary of options controlling electronic structure code parameters
    """
    # Add all additional keywords to here
    keyword_options = {
                       'charge': 0,
                       'multiplicity': 1,
                       'maxit': 100,
                       'damping': False,
                       'damp_factor': 0.5,
                       'spectral_shift': True,
                       'integral_algo': 'libint_core',
                       'ints_tolerance': 1.0e-14,
                       'freeze_core': False,
                       'beta': 1.0,
                       'threads': 1
                      }

    for key in options.keys():
        if key in keyword_options.keys(): 
            if type(options[key]) == type(keyword_options[key]):
                # Override default and assign, else print warning
                keyword_options[key] = options[key]
            else:
                print("Value '{}' for keyword option '{}' not recognized. Ignoring.".format(options[key],key))
        else:
            print("{} keyword option not recognized.".format(key))
    return keyword_options

def compute(xyz_path, basis_name, method, options=None, deriv_order=0, partial=None):
    """
    General function for computing energies, derivatives, and partial derivatives.
    """
    # Set keyword options
    if options:
        options = check_options(options)
        if deriv_order == 0:
            options['integral_algo'] = 'libint_core'
    else:
        options = check_options({})
    print("Using integral method: {}".format(options['integral_algo']))
    libint_interface.set_num_threads(options['threads'])
    print("Number of OMP Threads: {}".format(libint_interface.get_num_threads()))

    # Load molecule data
    geom_list = libint_interface.geometry(xyz_path)
    geom = jnp.asarray(geom_list)
    dim = geom.shape[0]

    charge = options['charge']
    assert options['multiplicity'] == 1, "Multiplicity must be 1."
    nuclear_charges = jnp.asarray(libint_interface.nuclear_charges(xyz_path))
    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    nfrzn = n_frozen_core(geom, nuclear_charges, charge) if options['freeze_core'] else 0

    nbf = libint_interface.nbf(basis_name, xyz_path)
    print("Number of basis functions: ", nbf)
    basis_set = (basis_name, nbf)

    if 'f12' in method:
        nri = libint_interface.nbf(basis_name + '-cabs', xyz_path)
        cabs_set = (basis_name + '-cabs', nri)

    # Energy and full derivative tensor evaluations
    if not partial:
        # Create energy evaluation function
        if method == 'scf' or method == 'hf' or method == 'rhf':
            args = (geom, basis_set, nelectrons, nuclear_charges, xyz_path)
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return restricted_hartree_fock(*args, options=options, deriv_order=deriv_order)
        elif method =='mp2':
            args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return restricted_mp2(*args, options=options, deriv_order=deriv_order)
        elif method =='mp2-f12':
            args = (geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return restricted_mp2_f12(*args, options=options, deriv_order=deriv_order)
        elif method =='ccsd':
            args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return rccsd(*args, options=options, deriv_order=deriv_order)
        elif method =='ccsd(t)':
            args = (geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path)
            def electronic_energy(*args, options=options, deriv_order=deriv_order):
                return rccsd_t(*args, options=options, deriv_order=deriv_order)
        else:
            print("Desired electronic structure method not understood. Use 'scf' 'hf' 'mp2' 'ccsd' or 'ccsd(t)' ")

        # Evaluate energy or derivative 
        if deriv_order == 0:
            energy = electronic_energy(*args)
            return energy
        elif deriv_order == 1:
            grad = jacfwd(electronic_energy, 0)(*args)
            deriv = jnp.round(grad, 10)
        elif deriv_order == 2:
            hess = jacfwd(jacfwd(electronic_energy, 0))(*args)
            deriv = jnp.round(hess.reshape(dim,dim), 10)
        elif deriv_order == 3:
            cubic = jacfwd(jacfwd(jacfwd(electronic_energy, 0)))(*args)
            deriv = jnp.round(cubic.reshape(dim,dim,dim), 10)
        elif deriv_order == 4:
            quartic = jacfwd(jacfwd(jacfwd(jacfwd(electronic_energy, 0))))(*args)
            deriv = jnp.round(quartic.reshape(dim,dim,dim,dim), 10)
        else:
            print("Error: Order {} derivatives are not exposed to the API.".format(deriv_order))
            deriv = 0
        return np.asarray(deriv)

    # Partial derivatives
    else:
        if len(partial) != deriv_order:
            raise Exception("The length of the index coordinates given by 'partial' argument should be the same as the order of differentiation")

        # Estimate memory footprint of two electron integrals partial derivatives
        natoms = geom.shape[0] // 3
        nderivs = get_required_deriv_vecs(natoms, deriv_order, partial).shape[0]
        ngigabytes = nbf**4 * 64 * 8 * nderivs / 1e9
        print("Estimated memory footprint from two-electron integral partial derivatives: {} GB".format(ngigabytes))

        # For partial derivatives, need to unpack each geometric coordinate into separate arguments
        # to differentiate wrt specific coordinates using JAX AD utilities. 

        #TODO support internal coordinate wrapper function.
        # This will take in internal coordinates, transform them into cartesians, and then compute integrals, energy
        # JAX will then collect the internal coordinate partial derivative instead. 
        if method == 'scf' or method == 'hf' or method == 'rhf':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_scf = restricted_hartree_fock(geom, basis_set, nelectrons, nuclear_charges, xyz_path,\
                                                options=options, deriv_order=deriv_order, return_aux_data=False)
                return E_scf
        elif method =='mp2':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_mp2 = restricted_mp2(geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path,\
                                       options=options, deriv_order=deriv_order)
                return E_mp2
        elif method =='mp2-f12':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_mp2f12 = restricted_mp2_f12(geom, basis_set, cabs_set, nelectrons, nfrzn, nuclear_charges,\
                                               xyz_path, options=options, deriv_order=deriv_order)
                return E_mp2f12
        elif method =='ccsd':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_ccsd = rccsd(geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path,\
                               options=options, deriv_order=deriv_order)
                return E_ccsd
        elif method =='ccsd(t)':
            def partial_wrapper(*args):
                geom = jnp.asarray(args)
                E_ccsd_t = rccsd_t(geom, basis_set, nelectrons, nfrzn, nuclear_charges, xyz_path,\
                                   options=options, deriv_order=deriv_order)
                return E_ccsd_t
        else:
            raise Exception("Error: Method {} not supported.".format(method))

        if deriv_order == 1:
            i = partial[0]
            partial_deriv = jacfwd(partial_wrapper, i)(*geom_list)
        elif deriv_order == 2:
            i,j = partial[0], partial[1]
            partial_deriv = jacfwd(jacfwd(partial_wrapper, i), j)(*geom_list)
        elif deriv_order == 3:
            i,j,k = partial[0], partial[1], partial[2]
            partial_deriv = jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k)(*geom_list)
        elif deriv_order == 4:
            i,j,k,l = partial[0], partial[1], partial[2], partial[3]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l)(*geom_list)
        elif deriv_order == 5:
            i,j,k,l,m = partial[0], partial[1], partial[2], partial[3], partial[4]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m)(*geom_list)
        elif deriv_order == 6:
            i,j,k,l,m,n = partial[0], partial[1], partial[2], partial[3], partial[4], partial[5]
            partial_deriv = jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(jacfwd(partial_wrapper, i), j), k), l), m), n)(*geom_list)
        else:
            print("Error: Order {} partial derivatives are not exposed to the API.".format(deriv_order))
            partial_deriv = 0
        return jnp.round(partial_deriv, 10)

def energy(xyz_path, basis_name, method, options=None):
    """
    Call an energy method on a molecule and basis set.

    Parameters
    ----------
    xyz_path : str
        A string representing the location of an standard .xyz file with coordinates in Angstrom

    basis_name : str
        A string representing a Gaussian basis set available in Libint2's basis set library (also needs to be in Libint's basis set library if using Libint interface).

    method : str
        A string representing a quantum chemistry method supported in Quax
        method = 'scf', method = 'mp2', method = 'ccsd(t)'

    options : dict
        Dictionary of user-supplied keyword options.

    Returns
    -------
    The electronic energy in a.u. (Hartrees)
    """
    E = compute(xyz_path, basis_name, method, options)
    return E

def derivative(xyz_path, basis_name, method, deriv_order, options=None):
    """
    Compute the full Cartesian derivative tensor for a particular energy method, molecule, and basis set. 

    Parameters
    ----------
    xyz_path : str
        A string representing the location of an standard .xyz file with coordinates in Angstrom

    basis_name : str
        A string representing a Gaussian basis set available in Libint2's basis set library (also needs to be in Libint's basis set library if using Libint interface).

    method : str
        A string representing a quantum chemistry method supported in Quax
        method = 'scf', method = 'mp2', method = 'ccsd(t)'

    deriv_order : int
        The order of the derivative. order = 1 -> first derivative ; order = 2 --> second derivative ...

    options : dict
        Dictionary of user-supplied keyword options.

    Returns
    -------
    deriv : float
        The requested derivative tensor, elements have units of Hartree/bohr^(n)
    """
    deriv = compute(xyz_path, basis_name, method, options, deriv_order)
    return deriv

def partial_derivative(xyz_path, basis_name, method, deriv_order, partial, options=None):
    """
    Computes one particular nth-order partial derivative of the energy of an electronic structure method
    w.r.t. a set of cartesian coordinates. If you have N cartesian coordinates in your molecule, the nuclear derivative tensor
    is N x N x N ... however many orders of differentiation. This function computes one element of that tensor, depending
    on the address of the derivative you supply.
    If you have 9 cartesian coordinates x1,y1,z1,x2,y2,z2,x3,y3,z3 and you want the quartic derivative d^4E/dx1dy2(dz3)^2
    the partial derivative address in the quartic derivative tensor would be (0, 4, 8, 8).
    Note that this is the same derivative as, say, (4, 8, 0, 8), or any other permutation of that tuple.
    Also note this is dependent upon the order in which you supply the cartesian coordinates in the molecule object,
    because that will determine the indices of the coordinates.

    Parameters
    ----------
    xyz_path : str
        A string representing the location of an standard .xyz file with coordinates in Angstrom

    basis_name : str
        A string representing a Gaussian basis set available in Libint2's basis set library (also needs to be in Libint's basis set library if using Libint interface).

    method : str
        A string representing a quantum chemistry method supported in Quax e.g. 'scf', 'mp2' 'ccsd(t)'

    deriv_order : int
        The order of the derivative. order = 1 -> first derivative ; order = 2 --> second derivative ...

    partial : tuple of ints
       A tuple of indices at which the desired derivative appears in the derivative tensor. 
       Coordinates are indexed according to their location in the row-wise flattened Cartesian coordinate array:
       atom  x   y   z
        A    0   1   2
        B    3   4   5 
        C    6   7   8 
       E.g. The second derivative w.r.t the first atoms x-components would have partial=(0,0)
       The mixed partial derivative w.r.t. y-components on first and third atoms would be partial=(1,7)

    options : dict
        Dictionary of user-supplied keyword options.

    Returns
    -------
    partial_deriv : float
        The requested partial derivative of the energy in units of Hartree/bohr^(n)
    """
    partial_deriv = compute(xyz_path, basis_name, method, options, deriv_order, partial)
    return partial_deriv

