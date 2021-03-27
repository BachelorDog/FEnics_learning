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

from dolfin import *
import matplotlib.pyplot as plt
from scipy import linalg as sla
import math
import sys
sys.path.append( "../../" )
from hippylib import *

class Cov:
    def __init__(self,n, corrlen):
        self.n = n
        self.C = np.zeros((n,n))
        self.h = float(1.)/float(n)
            
        self.mesh = UnitIntervalMesh(n)
        self.Vh = FunctionSpace(self.mesh, 'DG', 0)
        self.Mass = assemble(TrialFunction(self.Vh)*TestFunction(self.Vh)*dx)
             
        for i in range(0,n):
            self.C[i,i] = 1.
            for j in range(0,i):
                val = math.exp(-abs( float(i-j) )*self.h/corrlen)
                self.C[i,j] = val
                self.C[j,i] = val
                    
        self.C *= self.h
                    
    def init_vector(self,x,dim=0):
        self.Mass.init_vector(x, dim)
            
    def mult(self,x,y):
        xx = x.array()
        yy = np.dot(self.C, xx)
        y.set_local(yy)

if __name__ == "__main__":
                        
    n = 1000
    nvect = 70
    k = 50
    corrlen = .1
    C = Cov(n, corrlen)
    Omega = np.random.randn(n, nvect)
    
    dtrue, Vtrue = np.linalg.eigh(C.C)
    
    sort_perm = dtrue.argsort()
    dtrue = dtrue[ sort_perm[::-1] ]
    dtrue = dtrue[0:k]
    
    Vtrue = Vtrue[:, sort_perm[::-1] ]
    Vtrue = Vtrue[:,0:k]
    
    print dtrue
    
    d, U = singlePass(C,Omega,k)
    print d
    
    print( np.linalg.norm(d - dtrue) )
    
    d2, U2 = doublePass(C,Omega,k)
    print d2
    
    print( np.linalg.norm(d2 - dtrue) )
    
    i = range(0,k)
    plt.plot(i, dtrue, 'b-', i, d, 'r--', i, d2, 'g:', linewidth=3)
    plt.show()
    
    
    B = C.Mass
    Binv = LUSolver(B)
    
    C.C *= C.h    
    dG, UG = singlePassG(C,B, Binv, Omega,k, check_Bortho=True, check_Aortho=True, check_residual=True)
    dG2, UG2 = doublePassG(C,B, Binv, Omega,k,check_Bortho=True, check_Aortho=True, check_residual=True)
    plt.plot(i, dtrue, 'b-', i, dG, 'r--', i, d2, 'g:', linewidth=3)
    plt.show()
