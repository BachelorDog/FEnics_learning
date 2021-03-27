"""
Plot data-misfits
Shiwei Lan @ U of Warwick, 2016
"""

import os,pickle
# from df import *
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from itertools import cycle

def autocorr(x):
    """This one is closest to what plt.acorr does.
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array(
        [(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result

algs=('pCN','infMALA','infHMC','infmMALA','infmHMC','splitinfmMALA','splitinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','$\infty$-mMALA','$\infty$-mHMC','split$\infty$-mMALA','split$\infty$-mHMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)

folder = './analysis'
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]

max_iter=500
max_time=3000

# plot data-misfit
fig,axes = plt.subplots(num=0,nrows=2,figsize=(12,6))
lines = ["-","-.","--","-","--",":","--"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

for a in range(num_algs):
    for f_i in fnames:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                _,_,_,_,loglik,_,time,_=pickle.load(f)
#                 _,_,_,_,_,_,_,_,args=pickle.load(f)
                f.close()
                found[a]=True
            except:
                pass
    if found[a]:
#         spiter=time/(args.num_samp)
        spiter=time/10000
        axes[0].semilogy(range(max_iter),-loglik[:max_iter],next(linecycler0),linewidth=1.25)
        nsamp_in=np.floor(max_time/spiter)
        axes[1].semilogy(np.linspace(0,max_time,num=nsamp_in),-loglik[:nsamp_in],next(linecycler1),linewidth=1.25)

plt.axes(axes[0])
plt.axis('tight')
plt.xlabel('iteration',fontsize=14); plt.ylabel('data-misfit',fontsize=14)
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
plt.axis('tight')
plt.xlabel('time (seconds)',fontsize=14); plt.ylabel('data-misfit',fontsize=14)
# plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# add legend
h2_pos=axes[1].get_position()
plt.legend(np.array(alg_names)[found],fontsize=11,loc=2,bbox_to_anchor=(1.02,2.32),labelspacing=4.1)
fig.tight_layout(rect=[0,0,.85,1])
plt.savefig('./analysis/misfit.png')

plt.show()
