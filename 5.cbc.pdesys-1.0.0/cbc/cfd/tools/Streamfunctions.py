__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-11-03"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

import ufl
from dolfin import *

__all__ = ['StreamFunction', 'StreamFunction3D']

def StreamFunction(u, bcs, use_strong_bc = False):
    """Stream function for a given general 2D velocity field.
    The boundary conditions are weakly imposed through the term
    
        inner(q, grad(psi)*n)*ds, 
    
    where grad(psi) = [-v, u] is set on all boundaries. 
    This should work for any collection of boundaries: 
    walls, inlets, outlets etc.    
    """
    # Check dimension
    if isinstance(u, ufl.tensors.ListTensor):
        mesh = u[0].function_space().mesh()
    else:
        mesh = u.function_space().mesh()
    if not mesh.topology().dim() == 2:
        error("Stream-function can only be computed in 2D.")

    #V   = u.function_space().sub(0)
    degree = u[0].ufl_element().degree() if isinstance(u, ufl.tensors.ListTensor) else \
             u.ufl_element().degree()
    V   = FunctionSpace(mesh, 'CG', degree)
    q   = TestFunction(V)
    psi = TrialFunction(V)
    a   = dot(grad(q), grad(psi))*dx 
    L   = dot(q, (u[1].dx(0) - u[0].dx(1)))*dx
    n   = FacetNormal(mesh)
    
    if(use_strong_bc): 
        # Strongly set psi = 0 on entire domain. Used for drivencavity.
        bcu = [DirichletBC(V, Constant(0), DomainBoundary())]
    else:
        bcu=[]        
        L = L + q*(n[1]*u[0] - n[0]*u[1])*ds

    # Compute solution
    psi = Function(V)
    A = assemble(a)
    b = assemble(L)
    if not use_strong_bc: 
        normalize(b)  # Because we only have Neumann conditions
    [bc.apply(A, b) for bc in bcu]
    solve(A, psi.vector(), b, 'gmres', 'amg')
    if not use_strong_bc: 
        normalize(psi.vector())

    return psi
    
def StreamFunction3D(u, bcs):
    """Stream function for a given 3D velocity field.
    The boundary conditions are weakly imposed through the term
    
        inner(q, grad(psi)*n)*ds, 
    
    where u = curl(psi) is used to fill in for grad(psi) and set 
    boundary conditions. This should work for walls, inlets, outlets, etc.
    """
    # Check dimension
    mesh = u.function_space().mesh()
    if not mesh.topology().dim() == 3:
        error("Function used only for 3D.")

    V   = u.function_space()
    q   = TestFunction(V)
    psi = TrialFunction(V)
    a   = inner(grad(q), grad(psi))*dx
    L   = inner(q, curl(u))*dx 
    n   = FacetNormal(mesh)

    bcu = []
    efd = None 
    bid = 0
    #if any([bc.type() is 'Periodic' for bc in bcs]):
    if False:
        for bc in bcs:
            if(bc.type()=='Periodic'):
                b1 = PeriodicBC(V, bc)
                b1.type = bc.type
                bcu.append(b1)
            else:
                if(hasattr(bc,'mf')):
                    efd = bc.mf
                    bid = bc.bid
                else: 
                    # mark boundary with a meshfunction first and then add to L
                    if(not efd):
                        efd = MeshFunction('uint', mesh, 1)
                        efd.set_all(0)
                    bc.mark(efd, bid + 1)
                    bid = bid + 1
                L = L + (q[0]*n[2]*u[1] - q[0]*n[1]*u[2] + 
                         q[1]*n[0]*u[2] - q[1]*n[2]*u[0] + 
                         q[2]*n[1]*u[0] - q[2]*n[0]*u[1])*ds(bid)
                         
                a = a + dot(q[0]*n[0], (psi[0].dx(0) + psi[0].dx(1) + 
                                        psi[0].dx(2)))*ds(bid)
                a = a + dot(q[1]*n[1], (psi[1].dx(0) + psi[1].dx(1) + 
                                        psi[1].dx(2)))*ds(bid)
                a = a + dot(q[2]*n[2], (psi[2].dx(0) + psi[2].dx(1) + 
                                        psi[2].dx(2)))*ds(bid)
                
    else: # All other boundaries can be set directly
        L = L + (q[0]*n[2]*u[1] - q[0]*n[1]*u[2] + 
                 q[1]*n[0]*u[2] - q[1]*n[2]*u[0] + 
                 q[2]*n[1]*u[0] - q[2]*n[0]*u[1])*ds
              
        a = a + (dot(q[0]*n[0], (psi[0].dx(0) + psi[0].dx(1) + psi[0].dx(2))) +
                 dot(q[1]*n[1], (psi[1].dx(0) + psi[1].dx(1) + psi[1].dx(2))) +
                 dot(q[2]*n[2], (psi[2].dx(0) + psi[2].dx(1) + psi[2].dx(2))) 
                 )*ds
        
    # Compute solution
    psi = Function(V)
    A = assemble(a, exterior_facet_domains=efd)
    b = assemble(L, exterior_facet_domains=efd)
    if not use_strong_bc: 
        # Because we only have Neumann conditions
        normalize(b) 
    [bc.apply(A, b) for bc in bcu]
    solve(A, psi.vector(), b)
    if not use_strong_bc: 
        normalize(psi.vector())

    return psi
    