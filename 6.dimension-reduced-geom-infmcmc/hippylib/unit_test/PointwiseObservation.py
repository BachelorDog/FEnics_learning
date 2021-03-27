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
import sys
sys.path.append( "../../" )
from hippylib import *
import numpy as np

if __name__ == "__main__":
    set_log_active(False)
    ndim = 2
    nx = 64
    ny = 64
    mesh = UnitSquareMesh(nx, ny)
    
    ntargets = 3
    targets = np.array([[.3,.3], [.5,.5], [.7,.9]])
    
    Vh = FunctionSpace(mesh, 'Lagrange', 1)
    B = assemblePointwiseObservation(Vh,targets)
    u = Vector()
    B.init_vector(u,1)
    
    o = Vector()
    B.init_vector(o,0)
    
    uh = interpolate(Expression("x[0]"), Vh)
    u.axpy(1., uh.vector())
        
    B.mult(u,o)
    
    o_serial = o.gather_on_zero()
    
    print targets
    print o_serial
    for i in range(o_serial.shape[0]):
        assert np.abs( o_serial[i] - targets[i,0] ) < 1e-10
            
    Vh2 = VectorFunctionSpace(mesh, 'Lagrange', 1)
    B2 = assemblePointwiseObservation(Vh2,targets)
    u2 = Vector()
    B2.init_vector(u2,1)
    
    o2 = Vector()
    B2.init_vector(o2,0)
    
    u2h = interpolate(Expression(("x[0]", "x[1]") ), Vh2)
    u2.axpy(1., u2h.vector())
        
    B2.mult(u2,o2)
    
    o_serial = o2.gather_on_zero()
    for i in range(o_serial.shape[0]/2):
        assert np.abs( o_serial[2*i] - targets[i,0] ) < 1e-10
        assert np.abs( o_serial[2*i+1] - targets[i,1] ) < 1e-10
        
    Xh = Vh2*Vh
    B3 = assemblePointwiseObservation(Xh,targets)
    up = Vector()
    B3.init_vector(up,1)
    
    o_up = Vector()
    B3.init_vector(o_up,0)
    
    uph = interpolate(Expression(("x[0]", "x[1]", "2.*x[0]+3.*x[1]+10." )), Xh)
    up.axpy(1., uph.vector())
        
    B3.mult(up,o_up)
    
    o_serial = o_up.gather_on_zero()
    for i in range(o.array().shape[0]/3):
        assert np.abs( o_serial[3*i] - targets[i,0] ) < 1e-10
        assert np.abs( o_serial[3*i+1] - targets[i,1] ) < 1e-10
        assert np.abs( o_serial[3*i+2] - (2.*targets[i,0] + 3.*targets[i,1] + 10.) ) < 1e-10
        
    Vh_RT = FunctionSpace(mesh, 'RT', 1)
    B_RT = assemblePointwiseObservation(Vh_RT,targets)
    u_RT = Vector()
    B_RT.init_vector(u_RT,1)
    
    o_RT = Vector()
    B_RT.init_vector(o_RT,0)
    
    uh_RT = interpolate(Expression(("x[0]", "x[1]") ), Vh_RT)
    u_RT.axpy(1., uh_RT.vector())
        
    B_RT.mult(u_RT,o_RT)
    
    o_serial = o_RT.gather_on_zero()
    for i in range(o_serial.shape[0]/2):
        assert np.abs( o_serial[2*i] - targets[i,0] ) < 1e-10
        assert np.abs( o_serial[2*i+1] - targets[i,1] ) < 1e-10

    