#!/usr/bin/env python
"""
Some handy functions to facilitate usage of dolfin
Shiwei Lan @ U of Warwick, 2016
"""

import dolfin as df
import numpy as np

def create_PETScMatrix(shape,mpi_comm=None,rows=None,cols=None,values=None):
    """
    Create and set up PETScMatrix of arbitrary size using petsc4py.
    """
    if df.has_petsc4py():
        from petsc4py import PETSc
    else:
        print('Dolfin is not compiled with petsc4py! Cannot create PETScMatrix of arbitrary size.')
        exit()
    if mpi_comm is None:
        mpi_comm = df.mpi_comm_world()
    mat = PETSc.Mat()
    mat.create(mpi_comm)
    mat.setSizes(shape)
    mat.setType('aij')
    mat.setUp()
    mat.setValues(rows,cols,values)
    mat.assemble()
    return mat

def get_dof_coords(V):
    """
    Get the coordinates of dofs.
    """
    try:
        dof_coords = V.tabulate_dof_coordinates() # post v1.6.0
    except AttributeError:
        print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
        dof_coords = V.dofmap().tabulate_all_coordinates(V.mesh())
    dof_coords.resize((V.dim(), V.mesh().geometry().dim()))
    return dof_coords

def check_in_dof(points,V,tol=2*df.DOLFIN_EPS):
    """
    Check whether points are nodes where dofs are defined and output those dofs
    """
    # V should NOT be mixed function space! Unless you know what you are doing...
    if V.num_sub_spaces()>1:
        print('Warning: Multiple dofs associated with each point, unreliable outputs!')
    # obtain coordinates of dofs
    dof_coords=get_dof_coords(V)
    # check whether those points are close to nodes where dofs are defined
    pdist_pts2dofs = np.einsum('ijk->ij',(points[:,None,:]-dof_coords[None,:,:])**2)
    idx_in_dof = np.argmin(pdist_pts2dofs,axis=1)
    rel_idx_in = np.where(np.einsum('ii->i',pdist_pts2dofs[:,idx_in_dof])<tol**2)[0] # index relative to points
    idx_in_dof = idx_in_dof[rel_idx_in]
    loc_in_dof = points[rel_idx_in,]
    return idx_in_dof,loc_in_dof,rel_idx_in

def vec2fun(vec,V):
    """
    Convert a vector to a dolfin function such that the function has the vector as coefficients.
    """
    f = df.Function(V)
    f.vector()[:] = np.array(vec)
#     f.vector().set_local(vec[df.vertex_to_dof_map(V)])
    return f
 
def mat2fun(mat,V):
    """
    Convert a matrix to a dolfin mixed function such that each column corresponds to a component function.
    """
    k = mat.shape[1]
    # mixed functions to store functions
    M=df.MixedFunctionSpace([V]*k)
    # Extract subfunction dofs
    dofs = [M.sub(i).dofmap().dofs() for i in range(k)]
    f=df.Function(M)
    for i,dof_i in enumerate(dofs):
        f.vector()[dof_i]=mat[:,i]
    return f,dofs
    