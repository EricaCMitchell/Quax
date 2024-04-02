import jax.numpy as jnp

from .integrals import libint_interface
from .utils import n_frozen_core

class Molecule(object):

    def __init__(self, xyz_path, basis_name, options):
        # Load Molecular Data
        self.basis_name = basis_name
        self.xyz_path = xyz_path

        geom_list = libint_interface.geometry(self.xyz_path)
        self.geom = jnp.asarray(geom_list)
        self.natom = len(geom_list) // 3

        self.charge = options['charge']
        try:
            self.multiplicity = options['multiplicity']
        except:
            raise Exception("Multiplicity must be equal to 1")

        # Nuclear and electronic data
        self.nuclear_charges = jnp.asarray(libint_interface.nuclear_charges(xyz_path))
        self.nelectrons = int(jnp.sum(self.nuclear_charges)) - self.charge
        self.nfrzn = n_frozen_core(self.geom, self.nuclear_charges, self.charge) if options['freeze_core'] else 0

        # Basis set data
        self.nbf = libint_interface.nbf(self.basis_name, self.xyz_path)
        self.basis_set = (self.basis_name, self.nbf)
    
    def cabs(self):
        self.nri = libint_interface.nbf(self.basis_name + '-cabs', self.xyz_path)
        self.cabs_set = (self.basis_name + '-cabs', self.nri)
        return self.cabs_set
    
    def print_out(self):
        print("Basis Name: ", self.basis_name.upper())
        print("Number of Basis Functions: ", self.nbf)
        return self
    
    def __len__(self): #Allows the Molecule to be used by len()
        return self.natom