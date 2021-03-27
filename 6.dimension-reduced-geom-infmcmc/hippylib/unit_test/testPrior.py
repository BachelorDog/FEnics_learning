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
import numpy as np
import sys
sys.path.append( "../../" )
from hippylib import *

def testSqrtIdea(mesh, order, prior_type):
    
    print prior_type, "order: ", order, "ndim: ", mesh.geometry().dim()
    
    Vh = dl.FunctionSpace(mesh, 'Lagrange', order)
    if prior_type == "BiLaplacian":
        prior = BiLaplacianPrior(Vh, 1.,100., max_iter=500, rel_tol=1e-9)
    elif prior_type == "Laplacian":
        prior = LaplacianPrior(Vh, 1.,100., max_iter=500, rel_tol=1e-9)
    else:
        raise prior_type

    y = dl.Vector()
    prior.init_vector(y,0)
    tmp = dl.Vector()
    prior.init_vector(tmp, "noise")
    y2 = dl.Vector()
    prior.init_vector(y2,0)

    randx = dl.Vector()
    prior.init_vector(randx,0)
    randx.set_local(np.random.rand(randx.size() ) )
    for x in [dl.interpolate(dl.Constant(1), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[1]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[0]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[0]*x[1]"), Vh).vector(),
          randx]:
        
        if prior_type == "BiLaplacian":
            prior.M.mult(x,y)
            prior.sqrtM.transpmult(x,tmp)
            prior.sqrtM.mult(tmp, y2)
        else:
            prior.R.mult(x,y)
            prior.sqrtR.transpmult(x,tmp)
            prior.sqrtR.mult(tmp, y2)
        
        assert (y - y2).norm("linf")/y.norm("linf") < 1e-10
        
    print "SUCCESS"
        
if __name__ == "__main__":
    dl.set_log_active(False)
    nx = 8
    ny = 8
    nz = 8
    mesh2D = dl.UnitSquareMesh(nx, ny)
    mesh3D = dl.UnitCubeMesh(nx, ny, nz)
    orders = [1,2,3]
    
    for order in orders:
        testSqrtIdea(mesh2D, order, "BiLaplacian")
        testSqrtIdea(mesh2D, order, "Laplacian")
        testSqrtIdea(mesh3D, order, "BiLaplacian")
        testSqrtIdea(mesh3D, order, "Laplacian")
        
    