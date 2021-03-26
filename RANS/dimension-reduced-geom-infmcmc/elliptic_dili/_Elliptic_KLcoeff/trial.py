"""
Test solutions of Elliptic PDE model in the DILI paper by Cui et~al (2016)
Shiwei Lan @ U of Warwick, 2016
"""


from dolfin import *
import numpy as np
import time
import matplotlib.pyplot as plt

from Elliptic import Elliptic

# parameters["num_threads"] = 2

np.random.seed(2016)
# settings
dim=100
# choice of coefficient definition
# kl_opt='fb'
kl_opt='kf'

# generate observations
# theta=.1*np.ones(10)#np.random.randn(dim)
theta=np.random.randn(dim)
elliptic=Elliptic(nx=40,ny=40)
coeff=elliptic.coefficient(theta=theta,kl_opt=kl_opt,degree=2)

# solve forward equation
# u_fwd,p_fwd,l_fwd=elliptic.soln_fwd(theta)

# obtain observations
print('Obtaining observations...')
obs,idx,loc,sd_noise=elliptic.get_obs()
# print(obs)
# print(loc)
num_obs=len(idx)
print('%d observations have been obtained!' % num_obs)
# plot
# plot(elliptic.states_fwd.split()[0])
# interactive()

# define data misfit class
print('\nDefining data-misfit...')
misfit=elliptic.data_misfit(obs,1./sd_noise**2,idx,loc)

# obtain the geometric quantities
print('\n\nObtaining geometric quantities with Adjoint method...')
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
# plot
# elliptic.plot()
end = time.time()
print('Time used is %.4f' % (end-start))

# save solutions to file
# elliptic.save()
# plot solutions
elliptic.plot(backend='matplotlib',SAVE=True)

# plot solutions
# parameters["plotting_backend"]="matplotlib"
# u_fwd,p_fwd,_=elliptic.states_fwd.split(True)
# plt.figure(0)
# fig=plot(p_fwd)
# plt.colorbar(fig)
# plot(u_fwd)
# plt.show()
