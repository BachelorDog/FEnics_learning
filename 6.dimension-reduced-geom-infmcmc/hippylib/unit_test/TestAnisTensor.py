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

import dolfin as dl
import sys
sys.path.append( "../../" )
from hippylib import *
import numpy as np

if __name__ == "__main__":
    dl.set_log_active(False)
    ndim = 2
    nx = 64
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh = dl.FunctionSpace(mesh, "CG", 1)
    
    T = dl.Expression(code_AnisTensor2D)
    T.theta0 = 2.
    T.theta1 = .5
    T.alpha = math.pi/4
    
    u = dl.TrialFunction(Vh)
    v = dl.TestFunction(Vh)
    a = dl.inner(T*dl.grad(u), dl.grad(v))*dl.dx
    
    A = dl.assemble(a)
    
    e = dl.Expression(code_Mollifier)
    e.o = 2
    e.l = 0.2
    e.addLocation(.1,.1)
    e.addLocation(.1,.9)
    e.addLocation(.5,.5)
    e.addLocation(.9,.1)
    e.addLocation(.9,.9)
    e.theta0 = 1./T.theta0
    e.theta1 = 1./T.theta1
    e.alpha  = T.alpha
    
    m = dl.interpolate(e, Vh)
    
    dl.plot(m)
    dl.interactive()