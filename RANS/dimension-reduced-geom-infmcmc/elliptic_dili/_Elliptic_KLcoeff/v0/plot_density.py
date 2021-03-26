"""
Plot posterior of KL coefficients in inverse problem of Elliptic PDE model in the DILI paper by Cui et~al (2016)
Shiwei Lan @ U of Warwick, 2016
"""

from dolfin import *
import numpy as np
import time
import matplotlib.pyplot as plt

from Elliptic import Elliptic

# from joblib import Parallel, delayed
# import multiprocessing

# parameters["num_threads"] = 2

np.random.seed(2016)
# settings
theta_dim=100
# choice of coefficient definition
# kl_opt='fb'
kl_opt='kf'

# generate date
# theta=.1*np.ones(theta_dim)#np.random.randn(theta_dim)
# theta=np.random.randn(theta_dim)
theta=np.zeros(theta_dim)
elliptic=Elliptic(nx=30,ny=30)
coeff=elliptic.coefficient(theta=theta,kl_opt=kl_opt,degree=2)

# obtain observations
obs,idx,loc,sd_noise=elliptic.get_obs()

# define data misfit class
print('\nDefining data-misfit...')
misfit=elliptic.data_misfit(obs,1./sd_noise**2,idx,loc)

# preparing density plot
dim=[1,2]
print('\nPreparing posterior density plot in dimensions (%d, %d)...' % tuple(dim))
# log density function
# def logdensity(x,dim=dim):
#     theta_i = theta
#     N = x.size/len(dim)
#     ll = np.full(N,np.nan)
#     for i,p in zip(range(N),x):
#         theta_i[np.array(dim)-1] = p; coeff.theta = theta_i
#         try:
#             nll,_,_,_=elliptic.get_geom(coeff,misfit)
#             ll[i] = -nll
#         except RuntimeError:
#             print('Bad point (%.4f,%.4f) encountered: divergent solution!' % tuple(p))
#             pass
#     return ll

def logpdf(x,dim=dim,theta=theta,coeff=coeff,PDE=elliptic,obj=misfit):
    theta[np.array(dim)-1] = x; coeff.theta = theta
    try:
        nll,_,_,_=PDE.get_geom(coeff,obj)
        logpost = -nll #+ theta.dot(theta)/2
#         print('successful solving!')
    except RuntimeError:
        print('Bad point (%.4f,%.4f) encountered: divergent solution!' % tuple(x))
        logpost = np.nan
        pass
    return logpost



# from multiprocessing import Pool
# h =.2
# x = np.arange(-4.0, 4.0+h, h);
n = 10
x = np.linspace(-4.0, 4.0, num=n);
y = x.copy()
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(),Y.flatten()]).T
print(XY.shape)

start = time.time()
Z = map(logpdf,XY)
# num_cores = multiprocessing.cpu_count()
# Z = Parallel(n_jobs=2,backend="threading")(delayed(logpdf)(r) for r in XY)
# Z = Parallel(n_jobs=4,backend="threading")(map(delayed(logpdf), XY))
end = time.time()
print('Time used is %.4f' % (end-start))

Z=np.reshape(Z,X.shape)

# save
import pickle
f=open('../result/posterior.pckl','wb')
pickle.dump([X,Y,Z],f)
f.close()

# plot
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel(r'$\theta_{%d}$' % dim[0]); plt.ylabel(r'$\theta_{%d}$' % dim[1])
plt.title('log-Posterior Contour')
plt.savefig('../result/post_contour.png',bbox_inches='tight')
plt.show()
