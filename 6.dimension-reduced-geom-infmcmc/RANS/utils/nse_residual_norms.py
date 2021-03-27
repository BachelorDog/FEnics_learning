'''
Created on Feb 15, 2016

@author: uvilla
'''

import dolfin as dl
import numpy as np

class ResNorml2:
    """
    Standard l2 norm of a vector
    """
    def mult(self, x, y):
        y.zero()
        y.axpy(1.,x)
        
    def norm(self,x):
        return x.norm("l2")
    
    def inner(self,x,y):
        return x.inner(y)

class ResNormAinv:
    """
    This class is used in NewtonBacktrack (and possibly other non-linear solvers),
    to compute weighted norms of the residual.
    More specifically:
    - if A is the finite element mass matrix, then this class will compute
    the L^2(\Omega) norm of the residual)
    - if A is the finite element stiffness matrix, then this class will
    compute the H^{-1}(\Omega) norm of the residual)
    """
    def __init__(self,A):
        """
        Contructor
        
        INPUT:
        - A: a s.p.d. finite element matrix
        """
        self.Ainv = dl.PETScLUSolver()
        self.Ainv.set_operator(A)
        self.tmp = dl.Vector()
        A.init_vector(self.tmp,0)
        
    def mult(self, x, y):
        """
        Apply A inverse to x, returns the result in y
        """
        self.Ainv.solve(y,x)
        
    def inner(self,x,y):
        """
        Compute the A inverse weighted inner product of x and y: (A^-1 x, y)
        """
        self.Ainv.solve(self.tmp,x)
        return self.tmp.inner(y)
    
    def norm(self,x):
        """
        Compute the A inverse weighted norm of x: sqrt{( A^-1 x, x)}
        """
        return np.sqrt( self.inner(x,x) )