import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
import h5py
import os

# Check for Libint interface
from ..integrals import TEI
from ..integrals import OEI
from ..integrals import libint_interface


class Integrals(object):

    def __init__(self, geom, bs, xyz_path, deriv_order, options):
        # Load Integral Data
        self.algo = options['integral_algo']
        self.geom = geom
        self.xyz_path = xyz_path
        self.ints_tol = options['ints_tolerance']
        self.deriv_order = deriv_order

        # Load Basis Set Info
        self.bs = bs
        self.bs_name = bs[0]

    def compute_integrals(self):
        # Load integral algo, decides to compute integrals in memory or use disk
        libint_interface.initialize(self.xyz_path,self.bs_name, self.bs_name, 
                                    self.bs_name, self.bs_name, self.ints_tol)

        if self.algo == 'libint_disk':
            # Check disk for currently existing integral derivatives
            check_oei = check_oei_disk("all", self.bs, self.bs, self.deriv_order)
            check_tei = check_tei_disk("eri", self.bs, self.bs, self.bs, self.bs, self.deriv_order)

            oei_obj = OEI(self.bs, self.bs, self.xyz_path, self.deriv_order, 'disk')
            tei_obj = TEI(self.bs, self.bs, self.bs, self.bs, self.xyz_path, self.deriv_order, 'disk')
            # If disk integral derivs are right, nothing to do
            if check_oei:
                S = oei_obj.overlap(self.geom)
                T = oei_obj.kinetic(self.geom)
                V = oei_obj.potential(self.geom)
            else:
                libint_interface.oei_deriv_disk(self.deriv_order)
                S = oei_obj.overlap(self.geom)
                T = oei_obj.kinetic(self.geom)
                V = oei_obj.potential(self.geom)

            if check_tei:
                G = tei_obj.eri(self.geom)
            else:
                libint_interface.compute_2e_deriv_disk("eri", 0., self.deriv_order)
                G = tei_obj.eri(self.geom)

        else:
            # Precompute TEI derivatives
            oei_obj = OEI(self.bs, self.bs, self.xyz_path, self.deriv_order, 'core')
            tei_obj = TEI(self.bs, self.bs, self.bs, self.bs, self.xyz_path, self.deriv_order, 'core')
            # Compute integrals
            S = oei_obj.overlap(self.geom)
            T = oei_obj.kinetic(self.geom)
            V = oei_obj.potential(self.geom)
            G = tei_obj.eri(self.geom)

        libint_interface.finalize()
        return S, T, V, G

class F12_Integrals(object):

    def __init__(self, geom, xyz_path, deriv_order, options):
        # Load Integral Data
        self.algo = options['integral_algo']
        self.geom = geom
        self.xyz_path = xyz_path
        self.ints_tol = options['ints_tolerance']
        self.deriv_order = deriv_order

        # F12 Constant
        self.beta = options['beta']

    def compute_f12_oeints(self, bs1, bs2, is_cabs):
        # Initialize Libint2
        bs1_name = bs1[0]
        bs2_name = bs2[0]
        libint_interface.initialize(self.xyz_path, bs1_name, bs2_name, bs1_name, bs2_name, self.ints_tol)

        if is_cabs:
            if self.algo == 'libint_disk':
                # Check disk for currently existing integral derivatives
                check = check_oei_disk("overlap", self.bs1, self.bs2, self.deriv_order)
        
                oei_obj = OEI(self.bs1, self.bs2, self.xyz_path, self.deriv_order, 'disk')
                # If disk integral derivs are right, nothing to do
                if check:
                    S = oei_obj.overlap(self.geom)
                else:
                    libint_interface.compute_1e_deriv_disk("overlap", self.deriv_order)
                    S = oei_obj.overlap(self.geom)

            else:
                # Precompute OEI derivatives
                oei_obj = OEI(self.bs1, self.bs2, self.xyz_path, self.deriv_order, 'f12')
                # Compute integrals
                S = oei_obj.overlap(self.geom)
            
            libint_interface.finalize()
            return S

        else:
            if self.algo == 'libint_disk':
                # Check disk for currently existing integral derivatives
                check_T = check_oei_disk("kinetic", self.bs1, self.bs2, self.deriv_order)
                check_V = check_oei_disk("potential", self.bs1, self.bs2, self.deriv_order)

                oei_obj = OEI(self.bs1, self.bs2, self.xyz_path, self.deriv_order, 'disk')
                # If disk integral derivs are right, nothing to do
                if check_T:
                    T = oei_obj.kinetic(self.geom)
                else:
                    libint_interface.compute_1e_deriv_disk("kinetic", self.deriv_order)
                    T = oei_obj.kinetic(self.geom)

                if check_V:
                    V = oei_obj.potential(self.geom)
                else:
                    libint_interface.compute_1e_deriv_disk("potential", self.deriv_order)
                    V = oei_obj.potential(self.geom)

            else:
                # Precompute OEI derivatives
                oei_obj = OEI(self.bs1, self.bs2, self.xyz_path, self.deriv_order, 'f12')
                # Compute integrals
                T = oei_obj.kinetic(self.geom)
                V = oei_obj.potential(self.geom)
            
            libint_interface.finalize()
            return T, V

    def compute_f12_teints(self, bs1, bs2, bs3, bs4, int_type):
        # Initialize Libint2
        bs1_name = bs1[0]
        bs2_name = bs2[0]
        bs3_name = bs3[0]
        bs4_name = bs4[0]
        libint_interface.initialize(self.xyz_path, self.bs1_name, self.bs2_name, 
                                    self.bs3_name, self.bs4_name, self.ints_tol)

        if self.algo == 'libint_disk':
            # Check disk for currently existing integral derivatives
            check = check_tei_disk(int_type, self.bs1_name, self.bs2_name, 
                                   self.bs3_name, self.bs4_name, self.deriv_order)

            tei_obj = TEI(self.bs1_name, self.bs2_name, self.bs3_name, self.bs4_name, 
                          self.xyz_path, self.deriv_order, options, 'disk')
            # If disk integral derivs are right, nothing to do
            if check:
                match int_type:
                    case "f12":
                        F = tei_obj.f12(self.geom, self.beta)
                    case "f12_squared":
                        F = tei_obj.f12_squared(self.geom, self.beta)
                    case "f12g12":
                        F = tei_obj.f12g12(self.geom, self.beta)
                    case "f12_double_commutator":
                        F = tei_obj.f12_double_commutator(self.geom, self.beta)
                    case "eri":
                        F = tei_obj.eri(self.geom)
            else:
                match int_type:
                    case "f12":
                        libint_interface.compute_2e_deriv_disk(int_type, self.beta, self.deriv_order)
                        F = tei_obj.f12(self.geom, self.beta)
                    case "f12_squared":
                        libint_interface.compute_2e_deriv_disk(int_type, self.beta, self.deriv_order)
                        F = tei_obj.f12_squared(self.geom, self.beta)
                    case "f12g12":
                        libint_interface.compute_2e_deriv_disk(int_type, self.beta, self.deriv_order)
                        F = tei_obj.f12g12(self.geom, self.beta)
                    case "f12_double_commutator":
                        libint_interface.compute_2e_deriv_disk(int_type, self.beta, self.deriv_order)
                        F = tei_obj.f12_double_commutator(self.geom, self.beta)
                    case "eri":
                        libint_interface.compute_2e_deriv_disk(int_type, 0., self.deriv_order)
                        F = tei_obj.eri(self.geom)

        else:
            # Precompute TEI derivatives
            tei_obj = TEI(self.bs1_name, self.bs2_name, self.bs3_name, self.bs4_name, 
                          self.xyz_path, self.deriv_order, 'f12')
            # Compute integrals
            match int_type:
                case "f12":
                    F = tei_obj.f12(self.geom, self.beta)
                case "f12_squared":
                    F = tei_obj.f12_squared(self.geom, self.beta)
                case "f12g12":
                    F = tei_obj.f12g12(self.geom, self.beta)
                case "f12_double_commutator":
                    F = tei_obj.f12_double_commutator(self.geom, self.beta)
                case "eri":
                    F = tei_obj.eri(self.geom)

        libint_interface.finalize()
        return F

def check_oei_disk(int_type, basis1, basis2, deriv_order, address=None):
    # Check OEI's in compute_integrals
    correct_int_derivs = False
    correct_nbf1 = correct_nbf2 = correct_deriv_order = False

    if ((os.path.exists("oei_derivs.h5"))):
        print("Found currently existing one-electron integral derivatives in your working directory. Trying to use them.")
        oeifile = h5py.File('oei_derivs.h5', 'r')
        nbf1 = basis1[1]
        nbf2 = basis2[1]

        if int_type == "all":
            oei_name = ["overlap_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order),\
                        "kinetic_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order),\
                        "potential_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order)]
        else:
            oei_name = int_type + "_" + str(nbf1) + "_" + str(nbf2) + "_deriv" + str(deriv_order)

        for name in list(oeifile.keys()):
            if name in oei_name:
                correct_nbf1 = oeifile[name].shape[0] == nbf1
                correct_nbf2 = oeifile[name].shape[1] == nbf2
                correct_deriv_order = True
        oeifile.close()
        correct_int_derivs = correct_deriv_order and correct_nbf1 and correct_nbf2

    if correct_int_derivs:
        print("Integral derivatives appear to be correct. Avoiding recomputation.")
    return correct_int_derivs

def check_tei_disk(int_type, basis1, basis2, basis3, basis4, deriv_order, address=None):
    # Check TEI's in compute_integrals
    correct_int_derivs = False
    correct_nbf1 = correct_nbf2 = correct_nbf3 = correct_nbf4 = correct_deriv_order = False

    if ((os.path.exists(int_type + "_derivs.h5"))):
        print("Found currently existing " + int_type + " integral derivatives in your working directory. Trying to use them.")
        erifile = h5py.File(int_type + '_derivs.h5', 'r')
        nbf1 = basis1[1]
        nbf2 = basis2[1]
        nbf3 = basis3[1]
        nbf4 = basis4[1]
        tei_name = int_type + "_" + str(nbf1) + "_" + str(nbf2)\
                            + "_" + str(nbf3) + "_" + str(nbf4) + "_deriv" + str(deriv_order)
        
        # Check nbf dimension of integral arrays
        for name in list(erifile.keys()):
            if name in tei_name:
                correct_nbf1 = erifile[name].shape[0] == nbf1
                correct_nbf2 = erifile[name].shape[1] == nbf2
                correct_nbf3 = erifile[name].shape[2] == nbf3
                correct_nbf4 = erifile[name].shape[3] == nbf4
                correct_deriv_order = True
        erifile.close()
        correct_int_derivs = correct_deriv_order and correct_nbf1 and correct_nbf2 and correct_nbf3 and correct_nbf4
    
    if correct_int_derivs:
        print("Integral derivatives appear to be correct. Avoiding recomputation.")
    return correct_int_derivs
