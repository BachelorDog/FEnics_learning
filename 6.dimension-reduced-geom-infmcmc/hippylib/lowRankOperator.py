# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

from dolfin import Vector, MPI
from linalg import MultiVector, MatMvMult
import numpy as np

class LowRankOperator:
    """
    This class model the action of a low rank operator A = U D U^T.
    Here D is a diagonal matrix, and the columns of are orthonormal
    in some weighted inner-product.
    
    This class works only in serial!
    """
    def __init__(self,d,U, my_init_vector = None):
        """
        Construct the low rank operator given d and U.
        """
        self.d = d
        self.U = U
        self.my_init_vector = my_init_vector
        
    def init_vector(self, x, dim):
        """
        Initialize x to be compatible with the range/domain of A.
        """
        assert self.my_init_vector is not None
        self.my_init_vector(x,dim)
        
    def mult(self,x,y):
        """
        Compute y = Ax = U D U^T x
        """
        Utx = self.U.dot_v(x)
        dUtx = self.d*Utx   #elementwise mult
        y.zero()
        self.U.reduce(y, dUtx)
        
    def inner(self, x, y):
        Utx = self.U.dot_v(x)
        Uty = self.U.dot_v(y)
        return np.sum(self.d*Utx*Uty)        
        
    def solve(self, sol, rhs):
        """
        Compute sol = U D^-1 U^T x
        """
        Utr = self.U.dot_v(rhs)
        dinvUtr = Utr / self.d
        sol.zero()
        self.U.reduce(sol, dinvUtr)

        
    def get_diagonal(self, diag):
        """
        Compute the diagonal of A.
        """
        diag.zero()
        tmp = self.U[0].copy()
        for i in range(self.U.nvec()):
            tmp.zero()
            tmp.axpy(1., self.U[i] )
            tmp*= self.U[i]
            diag.axpy(self.d[i], tmp)

        
    def trace(self,W=None):
        """
        Compute the trace of A.
        If the weight W is given compute the trace of W^1/2AW^1/2.
        This is equivalent to
        tr_W(A) = \sum_i lambda_i,
        where lambda_i are the generalized eigenvalues of
        A x = lambda W^-1 x.
        
        Note if U is a W-orthogonal matrix then
        tr_W(A) = \sum_i D(i,i). 
        """
        if W is None:
            tmp = self.U[0].copy()
            tmp.zero()
            self.U.reduce(tmp, np.sqrt(self.d))
            tr = tmp.inner(tmp)
        else:
            WU = MultiVector(self.U[0], self.U.nvec())
            MatMvMult(W,self.U,WU)
            diagWUtU = np.zeros_like(self.d)
            for i in range(self.d.shape[0]):
                diagWUtU[i] = WU[i].inner(self.U[i])
            tr = np.sum(self.d*diagWUtU)
            
        return tr
    
    def trace2(self,W=None):
        """
        Compute the trace of A*A (Note this is the square of Frob norm, since A is symmetic).
        If the weight W is provided, it will compute the trace of (AW)^2.
        
        This is equivalent to 
        tr_W(A) = \sum_i lambda_i^2,
        where lambda_i are the generalized eigenvalues of
        A x = lambda W^-1 x.
        
        Note if U is a W-orthogonal matrix then
        tr_W(A) = \sum_i D(i,i)^2. 
        """
        if W is None:
            UtU = self.U.dot_mv(self.U)
            dUtU = self.d[:,None] * UtU #diag(d)*UtU.
            tr2 = np.sum(dUtU*dUtU)
        else:
            WU = MultiVector(self.U[0], self.U.nvec())
            MatMvMult(W,self.U,WU)
            WU = np.zeros(self.U.shape, dtype=self.U.dtype)
            UtWU = self.U.dot_mv(WU)
            dUtWU = self.d[:,None] * UtWU #diag(d)*UtU.
            tr2 = np.power(np.linalg.norm(dUtWU),2)
            
        return tr2