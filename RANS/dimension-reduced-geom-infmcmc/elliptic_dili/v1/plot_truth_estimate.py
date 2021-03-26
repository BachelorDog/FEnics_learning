"""
Plot the true and estimated transmissivity field of Elliptic inverse problem (DILI; Cui et~al, 2016)
Shiwei Lan @ U of Warwick, 2016
"""

import os,pickle
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

from Elliptic import Elliptic

np.random.seed(2016)
# define the PDE problem
elliptic=Elliptic(nx=40,ny=40)
# true transmissivity field
true_coeff=elliptic.true_coeff()
# coefficient
dim=100
theta=np.zeros(dim)
sigma=1.25;s=0.0625;kl_opt='kf'
coeff=elliptic.coefficient(theta=theta,kl_opt=kl_opt,sigma=sigma,s=s,degree=2)

# algorithms
algs=('pCN','infMALA','infHMC','infmMALA','infmHMC','splitinfmMALA','splitinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','$\infty$-mMALA','$\infty$-mHMC','split$\infty$-mMALA','split$\infty$-mHMC')
num_algs=len(algs)
# preparation for estimates
folder = './analysis'
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]

# plot
num_rows=2
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(16,6))
parameters["plotting_backend"]="matplotlib"

for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot truth
        u_coeff=interpolate(true_coeff,elliptic.V)
        plot(u_coeff)
        ax.set_title('True transmissivity')
    elif 1<=i<=num_algs:
        # plot posterior mean
        for f_i in fnames:
            if '_'+algs[i-1]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    _,_,_,samp,_,_,_,_=pickle.load(f)
                    f.close()
                    found=True
                except:
                    found=False
                    pass
        if found:
            theta_mean = np.mean(samp,axis=0)
            coeff.theta=theta_mean
            u_coeff,_,_=coeff.get_coeff()
            sub_fig=plot(u_coeff)
            ax.set_title('Estimate by '+alg_names[i-1])
    plt.axis('tight')

# set color bar
cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)

# save plot
# fig.tight_layout()
plt.savefig('./analysis/truth_est.png',bbox_inches='tight')

plt.show()
