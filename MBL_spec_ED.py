from __future__ import print_function, division
import sys,os

from quspin.basis import spin_basis_1d,boson_basis_1d, photon_basis, tensor_basis # Hilbert space bases
from quspin.operators import hamiltonian, quantum_operator # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
from quspin.tools.Floquet import Floquet,Floquet_t_vec # Floquet Hamiltonian
from quspin.basis.photon import coherent_state # HO coherent state
import numpy as np # generic math functions
import scipy as sp
import pandas as pd

from quspin.operators import exp_op # operators
from quspin.basis import spin_basis_general # spin basis constructor
from quspin.tools.measurements import ent_entropy # Entanglement Entropy

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from qutip import *
from qutip.piqs import *

from scipy.sparse import load_npz, save_npz


def Hamiltonian_level_statistics(N, J_zz=0.0, J_xx=0.0, J_yy=0, J_xy=0.0, J_z=0.0, J_x=0.0, amplitudez=0.0, amplitudex=0.0, periodic=True, conserve=False, seed=100):

    np.random.seed(seed)
    L = N;
    if periodic==True:
        boundary = L;
    else:
        boundary = L-1;


    random_disorder_z = 4 * amplitudez * (np.random.rand(L) - 0.5);
    random_disorder_x = 4 * amplitudex * (np.random.rand(L) - 0.5);

    H_zz = [[J_zz,i,(i+1)%L] for i in range(boundary)] # PBC
    H_xx = [[J_xx,i,(i+1)%L] for i in range(boundary)] # PBC
    H_xy = [[J_xy,i,(i+1)%L] for i in range(boundary)] # PBC
    H_yy = [[J_yy,i,(i+1)%L] for i in range(boundary)] # PBC
    H_z = [[J_z + random_disorder_z[i],i] for i in range(L)] # PBC
    H_x = [[J_x + random_disorder_x[i],i] for i in range(L)] # PBC

    static=[["+-",H_xy],["-+",H_xy],["zz",H_zz],["xx",H_xx], ["yy",H_yy],["z",H_z],["x",H_x]];
    dynamic=[];

    if conserve==True:
        basis=spin_basis_1d(L=N, Nup=N//2);
    else:
        basis=spin_basis_1d(L=N);
    H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);

    eig_vals = H.eigvalsh();

    length = len(eig_vals) // 2;
    start = length - 25;
    end = length + 25;
    eig_vals = eig_vals[start:end]
    eig_vals_spacing = np.array([eig_vals[i+1] - eig_vals[i] for i in range(len(eig_vals) - 1)]);

    r_val = 0.0;

    size = len(eig_vals_spacing)-1;
    r_val_dist = np.zeros([size]);
    print(size);
    j=0;
    for i in range(size):
        val = np.min([eig_vals_spacing[i], eig_vals_spacing[i+1]]) / np.max([eig_vals_spacing[i], eig_vals_spacing[i+1]]);
        r_val_dist[i] = size;
        #print(val);
        if val > -0.1:
            r_val += val;
            j+=1;

    print(r_val/j);
    return r_val/j, r_val_dist;

def timeEvolve_Hamiltonian(initial_state, Hamiltonian):
