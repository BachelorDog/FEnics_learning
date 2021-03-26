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

nx = 32
ny = 32
mesh = dl.UnitSquareMesh(nx, ny)
Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)
Prior = LaplacianPrior(Vh, 1.,100.)

nsamples = 1000

s = dl.Function(Vh, name = "sample")
noise = dl.Vector()
Prior.init_vector(noise,"noise")
size = len(noise.array())

fid = dl.File("results_cg/samples.pvd")

for i in range(0, nsamples ):
    noise.set_local( np.random.randn( size ) )
    Prior.sample(noise, s.vector())
    fid << s


