"""
Plot Forstner distance, the diagnostic for convergence of LIS
Shiwei Lan @ CalTech, 2017
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from itertools import cycle

# algorithms
algs=('DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)
# preparation for estimates
folder = './analysis-L25W8-nofreshinit-yesparacont-2000'
pckl_names=[f for f in os.listdir(folder) if f.endswith('.pckl')]

# plot residual error of mean
fig,axes = plt.subplots(num=0,nrows=2,figsize=(12,6))
lines = ["-","-.","--","-","--",":","--"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

# get results
for a in range(num_algs):
    for f_i in pckl_names:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f,encoding='bytes') # for python 3
#                 f_read=pickle.load(f)
                dims,dfs=f_read[-3:-1]
                f.close()
                found[a]=True
            except:
                print('Error encountered!')
                pass
    if found[a]:
        axes[0].plot(dims,next(linecycler0),linewidth=1.25)
        axes[1].plot(dfs,next(linecycler1),linewidth=1.25)
#         axes[1].semilogy(dfs,next(linecycler1),linewidth=1.25)

plt.axes(axes[0])
# plt.axis('tight')
plt.xlabel('iteration',fontsize=14); plt.ylabel('dimension of LIS',fontsize=14)
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
# plt.axis('tight')
plt.xlabel('iteration',fontsize=14); plt.ylabel('d_F',fontsize=14)
# plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.subplots_adjust(left=0.08, right=0.83, top=0.94, bottom=0.1, hspace=0.3)
# add legend
h2_pos=axes[1].get_position()
# plt.legend(np.array(alg_names)[found],fontsize=11,loc=2,bbox_to_anchor=(1.02,2.32),labelspacing=15)
plt.legend(np.array(alg_names)[found],fontsize=11,loc=2,bbox_to_anchor=(1.02,2.33),labelspacing=15)
# fig.tight_layout(rect=[0,0,.85,1])
plt.savefig(folder+'/dF.png')

plt.show()
