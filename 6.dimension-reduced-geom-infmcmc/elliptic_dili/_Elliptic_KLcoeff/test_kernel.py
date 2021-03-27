"""
Test assembling kernel function and solving the associated eigen-problem for the Gaussian prior
Shiwei Lan @ U of Warwick, 2016
"""


from dolfin import *
import numpy as np
import time
import scipy.sparse as sps
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib as mp

from Elliptic import Elliptic

# parameters["num_threads"] = 2

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_petsc4py():
    print("DOLFIN has not been configured with petsc4py. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

np.random.seed(2016)

# define elliptic model
elliptic=Elliptic(nx=40,ny=40)

# create and assemble the kernel, solve the associated eigen-problem
ker=elliptic.kernel()
# Compute all eigenvalues of A x = \lambda x
n=100
eigen=ker.get_eigen(n)

# Plot eigenfunctions
eigs_plot=np.array([1,2,10,n])
u = Function(elliptic.V)
parameters["plotting_backend"]="matplotlib"
fig,axes = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=(10,6))

for j,ax in enumerate(axes.flat):
    # Extract largest (first) eigenpair
    r, c, rx, cx = eigen.get_eigenpair(eigs_plot[j]-1)
    # Initialize function and assign eigenvector
    u.vector()[:] = rx
    # plot
    plt.axes(ax)
    sub_fig=plot(u)
    plt.axis('tight')
    ax.set_title('The %d-th eigenvalue: %.2e' % (eigs_plot[j],r),fontsize=10)
    ax.set_aspect(aspect='equal', adjustable='box-forced')

cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)
# plt.savefig('../result/eigenfunctions.png',bbox_inches='tight')
plt.show()