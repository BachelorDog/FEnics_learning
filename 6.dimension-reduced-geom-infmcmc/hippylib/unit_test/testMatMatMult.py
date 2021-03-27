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

dl.set_log_active(False)
nx = 8
ny = 8
mesh = dl.UnitSquareMesh(nx, ny)

Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)

uh = dl.TrialFunction(Vh)
vh = dl.TestFunction(Vh)

A = dl.assemble(uh*vh*dl.dx)
B = dl.assemble(dl.inner(dl.nabla_grad(uh), dl.nabla_grad(vh))*dl.dx)

C = MatMatMult(A,B)
D = MatPtAP(A,B)
E = MatAtB(A,B)
F = Transpose(A)