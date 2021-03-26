"""
Test solutions of Elliptic PDE in the DILI paper by Cui et~al (2016)
Shiwei Lan @ U of Warwick, 2016
"""


from dolfin import *
import numpy as np
import time
# import matplotlib.pyplot as plt

from Elliptic import Elliptic

# parameters["num_threads"] = 2

np.random.seed(2016)
# settings
dim=25
# choice of coefficient definition
# kl_opt='fb'
kl_opt='kf'

# generate observations
# theta=.1*np.ones(dim)
theta=.1*np.random.randn(dim)
elliptic=Elliptic(nx=30,ny=30)
# K-L expansion with specific choice
coeff=elliptic.coefficient(theta=theta,kl_opt=kl_opt,degree=2)

# solve forward equation
# u_fwd,p_fwd,l_fwd=elliptic.soln_fwd(theta)

# obtain observations
obs,idx,loc,sd_noise=elliptic.get_obs(coeff)

# define data misfit class
print('\nDefining data-misfit...')
misfit=elliptic.data_misfit(obs,1./sd_noise**2,idx,loc)

# obtain the geometric quantities
print('\n\nObtaining full geometric quantities with Adjoint method...')
start = time.time()
nll,dnll,Fv,FI = elliptic.get_geom(coeff,misfit,[0,1,1.5,2])
if dnll is not None:
    print('gradient:')
    print(dnll)
v = np.random.randn(coeff.l)
if Fv is not None:
    Ma = Fv(v)
    print('metric action on a random vector:')
    print(Ma)
if FI is not None:
    print('metric:')
    print(FI)
end = time.time()
print('Time used is %.4f' % (end-start))

# obtain the truncated geometric quantities
opt=1
lx=np.int(np.ceil(np.sqrt(dim)))
ly=np.int(np.ceil(dim/lx))
# idx=range(3)
idx=np.random.choice(min([lx,ly]),size=3,replace=False)
ind_x=np.zeros(lx); ind_x[idx]=1
ind_y=np.zeros(ly); ind_y[idx]=1
ind = ind_x[:,None].dot(ind_y[None,]).flatten()
idx2=np.where(ind)[0]
print('\nNow truncating on '+{0:'value, gradient, and metric (no sense)',
                            1:'gradient and metric',
                            2:'metric',
                            3:'none'}[opt]+'...')

# delete dlogcoeff_mat and dlogcoeff_sps for reinitialization
del elliptic.states_fwd,elliptic.dlogcoeff_mat,elliptic.dlogcoeff_sps
print('\n\nObtaining truncated geometric quantities with Adjoint method...')
start = time.time()
nll_t,dnll_t,Fv_t,FI_t = elliptic.get_geom(coeff,misfit,[0,1,1.5,2],opt,idx2)
if dnll_t is not None:
    print('gradient:')
    print(dnll_t)
if opt>2:
    v_t = v
else:
    v_t = v[idx2]
if Fv_t is not None:
    Ma_t = Fv_t(v_t)
    print('metric action on a random vector:')
    print(Ma_t)
if FI_t is not None:
    print('metric:')
    print(FI_t)
end = time.time()
print('Time used is %.4f' % (end-start))


# test with truncation
if opt<=2:
    print('The difference in metric is: %.10f' % np.linalg.norm(FI[np.ix_(idx2,idx2)]-FI_t)) # FI[idx2,idx2] is not the correct way to extract submatrix in numpy!
    print('The difference in metric action is: %.10f... expected... think about it:)' % np.linalg.norm(Ma[idx2]-Ma_t))
if opt<=1:
    print('The difference in gradient is: %.10f' % np.linalg.norm(dnll[idx2]-dnll_t))
if opt==0:
    print('The difference in misfit value is: %.10f' % np.linalg.norm(nll-nll_t))
