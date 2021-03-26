import dolfin as dl
import numpy as np
import scipy
from scipy.sparse.linalg import spsolve

# taken from http://fenicsproject.org/qa/7510/how-can-i-get-the-local-mesh-size-in-velocity-direction
def _sym2asym(HH):
    if HH.shape[0] == 3:
        return np.array([HH[0,:],HH[1,:],\
                             HH[1,:],HH[2,:]])
    else:
        return np.array([HH[0,:],HH[1,:],HH[3,:],\
                         HH[1,:],HH[2,:],HH[4,:],\
                         HH[3,:],HH[4,:],HH[5,:]])


def _c_cell_dofs(mesh,V):
    if V.ufl_element().is_cellwise_constant():
        return np.arange(mesh.num_cells()*mesh.coordinates().shape[1]**2)
    else:
        return np.arange(mesh.num_vertices()*mesh.coordinates().shape[1]**2)

def mesh_metric(mesh):
    """
    Compute the square inverse of a mesh metric
    """
    # this function calculates a mesh metric (or perhaps a square inverse of that, see mesh_metric2...)
    cell2dof = _c_cell_dofs(mesh,dl.TensorFunctionSpace(mesh, "DG", 0))
    cells = mesh.cells()
    coords = mesh.coordinates()
    p1 = coords[cells[:,0],:]
    p2 = coords[cells[:,1],:]
    p3 = coords[cells[:,2],:]
    r1 = p1-p2; r2 = p1-p3; r3 = p2-p3
    Nedg = 3
    if mesh.geometry().dim() == 3:
        Nedg = 6
        p4 = coords[cells[:,3],:]
        r4 = p1-p4; r5 = p2-p4; r6 = p3-p4

    rall = np.zeros([p1.shape[0],p1.shape[1],Nedg])
    rall[:,:,0] = r1; rall[:,:,1] = r2; rall[:,:,2] = r3
    if mesh.geometry().dim() == 3:
        rall[:,:,3] = r4; rall[:,:,4] = r5; rall[:,:,5] = r6

    All = np.zeros([p1.shape[0],Nedg**2])
    inds = np.arange(Nedg**2).reshape([Nedg,Nedg])
    for i in range(Nedg):
        All[:,inds[i,0]] = rall[:,0,i]**2; All[:,inds[i,1]] = 2.*rall[:,0,i]*rall[:,1,i]; All[:,inds[i,2]] = rall[:,1,i]**2
        if mesh.geometry().dim() == 3:
            All[:,inds[i,3]] = 2.*rall[:,0,i]*rall[:,2,i]; All[:,inds[i,4]] = 2.*rall[:,1,i]*rall[:,2,i]; All[:,inds[i,5]] = rall[:,2,i]**2

    Ain = np.zeros([Nedg*2-1,Nedg*p1.shape[0]])
    ndia = np.zeros(Nedg*2-1)
    for i in range(Nedg):
        for j in range(i,Nedg):
            iks1 = np.arange(j,Ain.shape[1],Nedg)
            if i==0:
                Ain[i,iks1] = All[:,inds[j,j]]
            else:
                iks2 = np.arange(j-i,Ain.shape[1],Nedg)
                Ain[2*i-1,iks1] = All[:,inds[j-i,j]]
                Ain[2*i,iks2]   = All[:,inds[j,j-i]]
                ndia[2*i-1] = i
                ndia[2*i]   = -i

    A = scipy.sparse.spdiags(Ain, ndia, Ain.shape[1], Ain.shape[1]).tocsr()
    b = np.ones(Ain.shape[1])
    X = spsolve(A,b)
    #set solution
    XX = _sym2asym(X.reshape([mesh.num_cells(),Nedg]).transpose())
    M = dl.Function(dl.TensorFunctionSpace(mesh,"DG", 0))
    M.vector().set_local(XX.transpose().flatten()[cell2dof])
    return M

def hinv_u(M, u, reg_u = dl.Constant(dl.DOLFIN_SQRT_EPS)):
    """
    Compute the inverse of the mesh size in the direction u.
    """
    return dl.sqrt(dl.dot(dl.dot(u,M),u)/(dl.dot(u,u)+ reg_u) )

def h_u(M, u, reg_u = dl.Constant(dl.DOLFIN_SQRT_EPS)):
    """
    Compute the mesh size in the direction u.
    """
    return dl.sqrt( dl.dot(u,u)/(dl.dot(dl.dot(u,M),u) + reg_u) )

def h_u2(M, u, reg_u = dl.Constant(dl.DOLFIN_SQRT_EPS)):
    """
    Compute the square mesh size in the direction u.
    """
    return dl.dot(u,u)/(dl.dot(dl.dot(u,M),u) + reg_u)

def h_over_u(M, u, reg_u = dl.Constant(dl.DOLFIN_SQRT_EPS)):
    """
    Compute h/norm_u.
    """
    return dl.Constant(1.)/dl.sqrt( dl.dot(dl.dot(u,M),u) + reg_u )

def h_dot_u(M, u, reg_u = dl.Constant(dl.DOLFIN_SQRT_EPS)):
    """
    Compute h times norm_u
    """
    return dl.dot(u,u) / dl.sqrt( (dl.dot(dl.dot(u,M),u) + reg_u) )
