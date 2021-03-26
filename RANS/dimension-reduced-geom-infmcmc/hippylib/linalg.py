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

from dolfin import compile_extension_module, Vector, PETScKrylovSolver, Function, MPI, DoubleArray, File, la_index_dtype
from random import Random
import os
import numpy as np

def amg_method():
    """
    Determine which AMG preconditioner to use.
    If avaialable use ML, which is faster than the PETSc one.
    """
    S = PETScKrylovSolver()
    for pp in S.preconditioners():
        if pp == 'ml_amg':
            return 'ml_amg'
        
    return 'petsc_amg'

abspath = os.path.dirname( os.path.abspath(__file__) )
sdir = os.path.join(abspath,"cpp_linalg")
header_file = open(os.path.join(sdir,"linalg.h"), "r")
code = header_file.read()
header_file.close()
cpp_sources = ["linalg.cpp"]  
cpp_module = compile_extension_module(
code=code, source_directory=sdir, sources=cpp_sources,
include_dirs=[".",  sdir])

class MultiVector(cpp_module.MultiVector):
    def dot_v(self, v):
        m = DoubleArray(self.nvec())
        self.dot(v, m)
        return np.zeros(self.nvec()) + m.array()
    
    def dot_mv(self,mv):
        shape = (self.nvec(),mv.nvec())
        m = DoubleArray(shape[0]*shape[1])
        self.dot(mv, m)
        return np.zeros(shape) + m.array().reshape(shape, order='C')
    
    def norm(self, norm_type):
        shape = self.nvec()
        m = DoubleArray(shape)
        self.norm_all(norm_type, m)
        return np.zeros(shape) + m.array()
    
    def Borthogonalize(self,B):
        """ 
        Returns QR decomposition of self.
        Q and R satisfy the following relations in exact arithmetic
        1. QR        = Z
        2. Q^*BQ     = I
        3. Q^*BZ    = R 
        4. ZR^{-1}    = Q
        
        Returns
        Bq : MultiVector: The B^{-1}-orthogonal vectors
        r : ndarray: The r of the QR decomposition
        Note: self is overwritten by Q    
        """
        return self._mgs_stable(B)
    
    def orthogonalize(self):
        """ 
        Returns QR decomposition of self.
        Q and R satisfy the following relations in exact arithmetic
        1. QR        = Z
        2. Q^*Q     = I
        3. Q^*Z    = R 
        4. ZR^{-1}  = Q
        
        Returns
        r : ndarray: The r of the QR decomposition
        Note: self is overwritten by Q    
        """
        return self._mgs_reortho()
    
    def _mgs_stable(self, B):
        """ 
        Returns QR decomposition of self, which satisfies conditions 1--4

        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the B-orthogonal QR factorization
        
        References
        ----------
        .. [1] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
               Hermitian Eigenvalue Problems with application to computing 
               Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885
               
        .. [2] W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980
        
        https://github.com/arvindks/kle
        
        """
        n = self.nvec()
        Bq = MultiVector(self[0], n)
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            B.mult(self[k], Bq[k])
            t = np.sqrt( Bq[k].inner(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = Bq[i].inner(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                B.mult(self[k], Bq[k])
                tt = np.sqrt(Bq[k].inner(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            Bq.scale(k, tt)
            
        return Bq, r 
    
    def _mgs_reortho(self):
        n = self.nvec()
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            t = np.sqrt( self[k].inner(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = self[i].inner(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                tt = np.sqrt(self[k].inner(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            
        return r
    
    def export(self, Vh, filename, varname = "mv", normalize=False):
        """
        Export in paraview this multivector
        Inputs:
        - Vh:        the parameter finite element space
        - filename:  the name of the paraview output file
        - varname:   the name of the paraview variable
        - normalize: if True the vector are rescaled such that || u ||_inf = 1 
        """
        fid = File(filename)
        if not normalize:
            for i in range(self.nvec()):
                fun = vector2Function(self[i], Vh, name = varname)
                fid << fun
        else:
            tmp = self[0].copy()
            for i in range(self.nvec()):
                s = self[i].norm("linf")
                tmp.zero()
                tmp.axpy(1./s, self[i])
                fun = vector2Function(tmp, Vh, name = varname)
                fid << fun
            
    
def MatMvMult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    for i in range(x.nvec()):
        A.mult(x[i], y[i])
        
def MvDSmatMult(X, A, Y):
    assert X.nvec() == A.shape[0], "X Number of vecs incompatible with number of rows in A"
    assert Y.nvec() == A.shape[1], "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec()):
        Y[j].zero()
        X.reduce(Y[j], A[:,j].flatten())
        

def MatMatMult(A,B):
    """
    Compute the matrix-matrix product A*B.
    """
    s = cpp_module.cpp_linalg()
    return s.MatMatMult(A,B)

def MatPtAP(A,P):
    """
    Compute the triple matrix product P^T*A*P.
    """
    s = cpp_module.cpp_linalg()
    return s.MatPtAP(A,P)

def MatAtB(A,B):
    """
    Compute the matrix-matrix product A^T*B.
    """
    s = cpp_module.cpp_linalg()
    return s.MatAtB(A,B)

def Transpose(A):
    """
    Compute the matrix transpose
    """
    s = cpp_module.cpp_linalg()
    return s.Transpose(A)

def SetToOwnedGid(v, gid, val):
    s = cpp_module.cpp_linalg()
    s.SetToOwnedGid(v, gid, val)
    
def GetFromOwnedGid(v, gid):
    s = cpp_module.cpp_linalg()
    return s.GetFromOwnedGid(v, gid)
    

def to_dense(A):
    """
    Convert a sparse matrix A to dense.
    For debugging only.
    """
    v = Vector()
    A.init_vector(v)
    mpi_comm = v.mpi_comm()
    nprocs = MPI.size(mpi_comm)
    
    if nprocs > 1:
        raise Exception("to_dense is only serial")
    
    if hasattr(A, "getrow"):
        n  = A.size(0)
        m  = A.size(1)
        B = np.zeros( (n,m), dtype=np.float64)
        for i in range(0,n):
            [j, val] = A.getrow(i)
            B[i,j] = val
        
        return B
    else:
        x = Vector()
        Ax = Vector()
        A.init_vector(x,1)
        A.init_vector(Ax,0)
        
        n = Ax.array().shape[0]
        m = x.array().shape[0]
        B = np.zeros( (n,m), dtype=np.float64) 
        for i in range(0,m):
            i_ind = np.array([i], dtype=np.intc)
            x.set_local(np.ones(i_ind.shape), i_ind)
            x.apply("sum_values")
            A.mult(x,Ax)
            B[:,i] = Ax.array()
            x.set_local(np.zeros(i_ind.shape), i_ind)
            x.apply("sum_values")
            
        return B


def trace(A):
    """
    Compute the trace of a sparse matrix A.
    """
    v = Vector()
    A.init_vector(v)
    mpi_comm = v.mpi_comm()
    nprocs = MPI.size(mpi_comm)
    
    if nprocs > 1:
        raise Exception("trace is only serial")
    
    n  = A.size(0)
    tr = 0.
    for i in range(0,n):
        [j, val] = A.getrow(i)
        tr += val[j == i]
    return tr

def get_diagonal(A, d):
    """
    Compute the diagonal of the square operator A.
    Use Solver2Operator if A^-1 is needed.
    """
    ej, xj = Vector(), Vector()
    A.init_vector(ej,1)
    A.init_vector(xj,0)
                    
    g_size = ej.size()    
    d.zero()
    for gid in xrange(g_size):
        owns_gid = ej.owns_index(gid)
        if owns_gid:
            SetToOwnedGid(ej, gid, 1.)
        ej.apply("insert")
        A.mult(ej,xj)
        if owns_gid:
            val = GetFromOwnedGid(xj, gid)
            SetToOwnedGid(d, gid, val)
            SetToOwnedGid(ej, gid, 0.)
        ej.apply("insert")
        
    d.apply("insert")

    

def estimate_diagonal_inv2(Asolver, k, d):
    """
    An unbiased stochastic estimator for the diagonal of A^-1.
    d = [ \sum_{j=1}^k vj .* A^{-1} vj ] ./ [ \sum_{j=1}^k vj .* vj ]
    where
    - vj are i.i.d. ~ N(0, I)
    - .* and ./ represent the element-wise multiplication and division
      of vectors, respectively.
      
    REFERENCE:
    Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad,
    An estimator for the diagonal of a matrix,
    Applied Numerical Mathematics, 57 (2007), pp. 1214-1229.
    """
    x, b = Vector(), Vector()
    
    if hasattr(Asolver, "init_vector"):
        Asolver.init_vector(x,1)
        Asolver.init_vector(b,0)
    else:       
        Asolver.get_operator().init_vector(x,1)
        Asolver.get_operator().init_vector(b,0)
        
    d.zero()
    for i in range(k):
        x.zero()
        Random.normal(b, 1., True)
        Asolver.solve(x,b)
        x *= b
        d.axpy(1./float(k), x)
        
def randn_perturb(x, std_dev):
    """
    Add a Gaussian random perturbation to x:
    x = x + eta, eta ~ N(0, std_dev^2 I)
    """
    Random.normal(x, std_dev, False)
#    n = x.array().shape[0]
#    noise = np.random.normal(0, 1, n)
#    x.set_local(x.array() + std_dev*noise)
#    x.apply("add_values")
    
class Solver2Operator:
    def __init__(self,S):
        self.S = S
        self.tmp = Vector()
        
    def init_vector(self, x, dim):
        if hasattr(self.S, "init_vector"):
            self.S.init_vector(x,dim)
        elif hasattr(self.S, "operator"):
            self.S.operator().init_vector(x,dim)
        elif hasattr(self.S, "get_operator"):
            self.S.get_operator().init_vector(x,dim)
        else:
            raise
        
    def mult(self,x,y):
        self.S.solve(y,x)
        
    def inner(self, x, y):
        self.S.solve(self.tmp,y)
        return self.tmp.inner(x)
    
def vector2Function(x,Vh, **kwargs):
    """
    Wrap a finite element vector x into a finite element function in the space Vh.
    kwargs is optional keywords arguments to be passed to the construction of a dolfin Function
    """
    fun = Function(Vh,**kwargs)
    fun.vector().zero()
    fun.vector().axpy(1., x)
    
    return fun
