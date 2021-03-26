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

from time import time
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt="Elapsed: %s s"):
    print fmt % (time() - _tstart_stack.pop())

dl.set_log_active(False)
nx = 8
ny = 8
mesh = dl.UnitSquareMesh(nx, ny)

for i in range(0):
    cell_markers = dl.CellFunction("bool", mesh)
    cell_markers.set_all(False)
    for cell in dl.cells(mesh):
        if cell.midpoint()[1] < .75 and cell.midpoint()[1] > .25 and cell.midpoint()[0] > .2 and cell.midpoint()[0] < .5:
            cell_markers[cell] = True
            
    cell_markers.set_all(True)
    
    mesh = dl.refine(mesh, cell_markers)

Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)

#fname = "results_biLapl/samples.pvd"
fname = "results_Lapl/samples.pvd"

if fname == "results_biLapl/samples.pvd":
    Prior = BiLaplacianPrior(Vh, 1.,100., max_iter=500, rel_tol=1e-9)
    randx = dl.Vector()
    Prior.init_vector(randx,0)
    randx.set_local(np.random.rand(randx.size() ) )
    for x in [dl.interpolate(dl.Constant(1), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[1]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[0]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[0]*x[1]"), Vh).vector(),
          randx]:
        y = dl.Vector()
        Prior.init_vector(y,0)
        Prior.M.mult(x,y)
        tmp = dl.Vector()
        Prior.init_vector(tmp, "noise")
        Prior.sqrtM.transpmult(x,tmp)
        y2 = dl.Vector()
        Prior.init_vector(y2,0)
        Prior.sqrtM.mult(tmp, y2)
        print (y - y2).norm("linf")/y.norm("linf")
elif fname == "results_Lapl/samples.pvd":
    Prior = LaplacianPrior(Vh, 1.,100., max_iter=500, rel_tol=1e-9)
    randx = dl.Vector()
    Prior.init_vector(randx,0)
    randx.set_local(np.random.rand(randx.size() ) )
    for x in [dl.interpolate(dl.Constant(1), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[1]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[0]"), Vh).vector(),
          dl.interpolate(dl.Expression("x[0]*x[0]*x[1]"), Vh).vector(),
          randx]:
        y = dl.Vector()
        Prior.init_vector(y,0)
        Prior.R.mult(x,y)
        tmp = dl.Vector()
        Prior.init_vector(tmp, "noise")
        Prior.sqrtR.transpmult(x,tmp)
        y2 = dl.Vector()
        Prior.init_vector(y2,0)
        Prior.sqrtR.mult(tmp, y2)
        print (y - y2).norm("linf")/y.norm("linf")
else:
    assert(1==0)
    
exit()
nsamples = 500

s = dl.Function(Vh, name = "sample")
noise = dl.Vector()
Prior.init_vector(noise,0)

# Test if R and Rsolver work ok.
x = dl.interpolate(dl.Expression("x[0]+x[1]*x[1]"), Vh)
Prior.R.mult(x.vector(), noise)
nit = Prior.Rsolver.solve(s.vector(), noise)
print "N iter: ", nit, " error:", (s.vector()-x.vector()).norm("linf")

pointwise_var = dl.Vector()
Prior.init_vector(pointwise_var,0)
tic()
if type(Prior.R) == dl.Matrix:
    Rsparsity = Prior.R
else:
    Rsparsity = MatPtAP(Prior.M, Prior.A)
    
coloring = getColoring(Rsparsity,8)
estimate_diagonal_inv_coloring(Prior.Rsolver, coloring, pointwise_var)
toc("Estimate Elapsed: %s s")

pointwise_var_2 = dl.Vector()
Prior.init_vector(pointwise_var_2,0)
tic()
estimate_diagonal_inv2(Prior.Rsolver, 667, pointwise_var_2)
toc("Estimate Elapsed: %s s")

pointwise_var_exact = dl.Vector()
Prior.init_vector(pointwise_var_exact,0)
tic()
get_diagonal(Prior.Rsolver, pointwise_var_exact, solve_mode=True)
toc("Exact Elapsed: %s s")

print ( pointwise_var - pointwise_var_exact).norm("linf") / ( pointwise_var_exact).norm("linf")
print ( pointwise_var_2 - pointwise_var_exact).norm("linf") / ( pointwise_var_exact).norm("linf")

dl.plot(vector2Function(pointwise_var_exact - pointwise_var, Vh) )
dl.plot(vector2Function(pointwise_var_exact - pointwise_var_2, Vh) )
dl.interactive()

dl.File("marginal_variance.pvd") << vector2Function(pointwise_var, Vh, name="pointwise_variance") << \
vector2Function(pointwise_var_2, Vh, name="pointwise_variance") << vector2Function(pointwise_var_exact, Vh, name="pointwise_variance")

exit()

class RinvM:
    def mult(self,x,y):
        Prior.Rsolver.solve(y, Prior.M*x)
        
    def init_vector(self,x, dim):
        Prior.init_vector(x,dim)
        
class MRinv:
    def mult(self,x,y):
        help = dl.Vector()
        Prior.M.init_vector(help,0)
        Prior.Rsolver.solve(help, x)
        Prior.M.mult(help,y)
    def init_vector(self,x, dim):
        Prior.init_vector(x,dim)

tr_estimator = TraceEstimator(RinvM(), False, 1e-1, Prior.init_vector)
print "Trace: ", tr_estimator(min_iter=20, max_iter=1000), tr_estimator.converged, tr_estimator.iter

marginal_variance_RinvM = dl.Vector()
Prior.init_vector(marginal_variance_RinvM,0)
get_diagonal(RinvM(), marginal_variance_RinvM, solve_mode=False)
print "trace using true diagonal R^-1M: ", np.sum( marginal_variance_RinvM.array() )

dl.File("marginal_variance_RinvM.pvd") << vector2Function(marginal_variance_RinvM, Vh, name="marginal_variance_RinvM")

marginal_variance_MRinv = dl.Vector()
Prior.init_vector(marginal_variance_MRinv,0)
get_diagonal(MRinv(), marginal_variance_MRinv, solve_mode=False)
print "trace using true diagonal MR^-1: ", np.sum( marginal_variance_MRinv.array() )

fid = dl.File(fname)

sum_tr = 0
sum_tr2 = 0
# Compute samples. Use Paraview Temporal Statical Filter to visualize average and std_dev
my_size = len(noise.array())
marginal_sample_variance = np.zeros(my_size)
for i in range(0, nsamples ):
    noise.set_local( np.random.randn( my_size ) )
    Prior.sample(noise, s.vector())
    sv = s.vector()
    marginal_sample_variance += sv.array()*sv.array()
    tr = sv.inner(Prior.M*sv)
    sum_tr2 += tr*tr
    sum_tr += tr
    fid << s
    
marginal_sample_variance /= nsamples


tmp = dl.Function(Vh)
tmp.vector().set_local(marginal_sample_variance)

dl.plot(vector2Function(marginal_variance, Vh) )
dl.plot( tmp )
dl.plot( tmp - vector2Function(marginal_variance, Vh) )
  

exp_tr = sum_tr/nsamples
exp_tr2 = sum_tr2/nsamples    
print "Trace computed from L2 norm of samples: mean={0}, variance={1}".format(exp_tr, exp_tr2 - exp_tr*exp_tr)
    
dl.interactive()




