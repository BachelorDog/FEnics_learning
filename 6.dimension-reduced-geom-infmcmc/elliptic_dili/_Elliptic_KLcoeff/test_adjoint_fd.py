"""
Test solutions of Elliptic PDE in the DILI paper by Cui et~al (2016)
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
dim=9
# choice of coefficient definition
kl_opt='fb'
# kl_opt='kf'

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

# parameters["plotting_backend"]="matplotlib"
# plt.figure(0)
# fig=plot(elliptic.states_fwd.split(True)[0])
# plt.colorbar(fig)

# define data misfit class
print('\nDefining data-misfit...')
misfit=elliptic.data_misfit(obs,1./sd_noise**2,idx,loc)

# ------------ early test ---------------------#
# elliptic.set_forms(coeff)
# u0,_=elliptic.soln_fwd()
# u1,_=elliptic.states_fwd.split(True)
# u2=u1.vector()
#
# print('Data-misfit: % .10f' % misfit.eval(u0))
# print('Data-misfit: % .10f' % misfit.eval(u1))
# print('Data-misfit: % .10f' % misfit.eval(u2))
#
# # test misfit as functional for adjoint
# J_form = misfit.form(u0)
# J_assemb = assemble(J_form)
# print('Assembled data-misfit form: % .10f' % J_assemb)
# J_func = misfit.func(u0)
# J_value = sum([J_func(list(p)) for p in loc])
# print('Evaluated data-misfit function: % .10f' % J_value)
#
# # solve adjoint equation
# elliptic.set_forms(coeff,ord=[0,1])
# u_adj,l_adj=elliptic.soln_adj(misfit)
# elliptic.plot(backend='vtk')
#
# # obtain gradient of data-misfit
# g = elliptic.get_grad(misfit)
# print(g)
#
# # solve 2nd forward equation
# u_actedon = np.random.randn(len(theta))
# # u_fwd2,p_fwd2,l_fwd2=elliptic.soln_fwd2(u_actedon)
#
# # solve 2nd adjoint equation
# # u_adj,p_adj2,l_adj2=elliptic.soln_adj2()
#
# # obtain metric action
# Ma = elliptic.get_metact(u_actedon)
# print (Ma)


# ------------ adjoint method ---------------------#

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
# elliptic.plot()

# ------------ finite difference ---------------------#

# check with finite difference
print('\n\nTesting against Finite Difference method...')
start = time.time()
h = 1e-6
theta1 = theta.copy(True);

## gradient
print('\nFirst gradient:')
dnll_fd = np.zeros_like(dnll)
for i in range(len(theta)):
    theta1[i]+=h; coeff.theta=theta1
    nll_p,_,_,_ = elliptic.get_geom(coeff,misfit)
    theta1[i]-=2*h; coeff.theta=theta1
    nll_m,_,_,_ = elliptic.get_geom(coeff,misfit)
    dnll_fd[i] = (nll_p-nll_m)/(2*h)
    theta1[i]+=h;
print('gradient:')
print(dnll_fd)
diff_grad = dnll_fd-dnll
print('Difference in gradient between adjoint and finite difference: %.10f (inf-norm) and %.10f (2-norm)' % (np.linalg.norm(diff_grad,np.inf),np.linalg.norm(diff_grad)))

## metric-action
print('\nThen Metric-action:')
Ma_fd = np.zeros_like(Ma)
# obtain sensitivities
for n in range(len(idx)):
    misfit_n=elliptic.data_misfit(obs[n],1./sd_noise**2,idx[n],loc[None,n,])
    dudtheta=np.zeros_like(theta)
    for i in range(len(theta)):
        theta1[i]+=h; coeff.theta=theta1
        elliptic.set_forms(coeff)
        u_p,_ = elliptic.soln_fwd()
        u_p_vec = misfit_n.extr_sol_vec(u_p)
        theta1[i]-=2*h; coeff.theta=theta1
        elliptic.set_forms(coeff)
        u_m,_ = elliptic.soln_fwd()
        u_m_vec = misfit_n.extr_sol_vec(u_m)
        dudtheta[i]=(u_p_vec-u_m_vec)/(2*h)
        theta1[i]+=h;
    Ma_fd += dudtheta*(dudtheta.dot(v))
Ma_fd *= misfit.prec
print('metric action on a random vector:')
print(Ma_fd)
diff_Ma = Ma_fd-Ma
print('Difference in metric-action between adjoint and finite difference: %.10f (inf-norm) and %.10f (2-norm)' % (np.linalg.norm(diff_Ma,np.inf),np.linalg.norm(diff_Ma)))
end = time.time()
print('Time used is %.4f' % (end-start))
