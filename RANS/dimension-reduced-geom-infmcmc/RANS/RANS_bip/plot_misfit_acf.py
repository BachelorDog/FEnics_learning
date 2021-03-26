"""
Plot autocorrelation of data-misfits
Shiwei Lan @ U of Warwick, 2016
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

algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','DR-$\infty$-mMALA','DR-$\infty$-mHMC','DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)

folder = './analysis-L25W8-nofreshinit-yesparacont-2000'
fnames=[f for f in os.listdir(folder) if f.endswith('.pckl')]

# plot data-misfit
fig,axes = plt.subplots(num=0,ncols=2,figsize=(12,6))
lines = ["-","--","-.",":"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

for a in range(num_algs):
    for f_i in fnames:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f,encoding='bytes')
                adjuster=-(algs[a]=='DILI')+('aDRinf' in algs[a])
                loglik=f_read[adjuster+3]
                f.close()
                found[a]=True
            except:
                pass
    if found[a]:
        # modify misifits to discern their traceplots
#         misfit=-loglik[500:]-a*3
        misfit=-loglik[500:]-a*10
        axes[0].plot(misfit,next(linecycler0),linewidth=1.25)
        # pd.tools.plotting.autocorrelation_plot(loglik[1000:], ax=axes[1],linestyle=next(linecycler))
        acorr_misfit=autocorr(misfit)
#         axes[1].plot(range(1,21),acorr_misfit[:20],next(linecycler1),linewidth=1.25)
#         plt.xticks(np.arange(2,21,2),np.arange(2,21,2))
        axes[1].plot(range(1,51),acorr_misfit[:50],next(linecycler1),linewidth=1.25)
        plt.xticks(np.arange(5,51,5),np.arange(5,51,5))
        
        plt.axhline(y=0.0, color='r', linestyle='-')

plt.axes(axes[0])
plt.axis('tight')
plt.xlabel('iteration',fontsize=14); plt.ylabel('data-misfit (offset)',fontsize=14)
# plt.legend(np.array(alg_names)[found],fontsize=10.2,loc=3,ncol=num_algs,bbox_to_anchor=(0,1.02,1.,0.102))
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
plt.axis('tight')
plt.xlabel('lag',fontsize=14); plt.ylabel('auto-correlation',fontsize=14)
fig.tight_layout(rect=[0,0,1,.9])
plt.axes(axes[0])
plt.legend(np.array(alg_names)[found],fontsize=10.2,loc=3,ncol=num_algs,bbox_to_anchor=(0,1.02,1.,0.102))
plt.savefig(folder+'/misfit_acf.png')

plt.show()
