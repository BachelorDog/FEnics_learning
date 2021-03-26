"""
Plot data-misfits as function of time
Shiwei Lan @ CalTech, 2017
"""

import os,pickle
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

# algorithms
algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','DR-$\infty$-mMALA','DR-$\infty$-mHMC','DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)
# preparation for estimates
SNR=100
folder = './analysis_f_SNR'+str(SNR)
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]

max_iter=100
max_time=100

# plot data-misfit
fig,axes = plt.subplots(num=0,nrows=2,figsize=(12,6))
lines = ["-","-.","--","-","--",":","--"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

for a in range(num_algs):
    for f_i in fnames:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f,encoding='bytes')
                adjuster=-(algs[a]=='DILI')+('aDRinf' in algs[a])
                loglik=f_read[adjuster+3]
                time=f_read[adjuster+5]
#                 times=f_read[adjuster+6]
                f.close()
                found[a]=True
            except:
                pass
    if found[a]:
        axes[0].semilogy(range(max_iter),-loglik[:max_iter],next(linecycler0),linewidth=1.25)
        spiter=time/2000
        nsamp_in=np.int(np.floor(max_time/spiter))
        axes[1].semilogy(np.linspace(0,max_time,num=nsamp_in),-loglik[:nsamp_in],next(linecycler1),linewidth=1.25)
#         plt_idx=times<=max_time
#         axes[1].semilogy(times[plt_idx],-loglik[plt_idx],next(linecycler1),linewidth=1.25)

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
plt.legend(np.array(alg_names)[found],fontsize=11,loc=2,bbox_to_anchor=(1.02,2.32),labelspacing=4)
fig.tight_layout(rect=[0,0,.85,1])
plt.savefig(folder+'/misfit_t.png')

plt.show()
