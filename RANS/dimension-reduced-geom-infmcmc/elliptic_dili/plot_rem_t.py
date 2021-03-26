"""
Plot residual error of mean as function of time
Shiwei Lan @ CalTech, 2017
"""

import os,pickle
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from itertools import cycle
from Elliptic_dili import Elliptic


# define the inverse problem
np.random.seed(2017)
SNR=100
elliptic = Elliptic(nx=40,ny=40,SNR=SNR)

# algorithms
algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','DR-$\infty$-mMALA','DR-$\infty$-mHMC','DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
found = np.zeros(num_algs,dtype=np.bool)
# preparation for estimates
folder = './analysis_f_SNR'+str(SNR)
h5_names=[f for f in os.listdir(folder) if f.endswith('.h5')]
num_samp=2000
pckl_names=[f for f in os.listdir(folder) if f.endswith('.pckl')]

max_iter=1000
max_time=1500

# plot residual error of mean
fig,axes = plt.subplots(num=0,nrows=2,figsize=(12,6))
lines = ["-","-.","--","-","--",":","--"]
linecycler0 = cycle(lines); linecycler1 = cycle(lines);

# get MAP
try:
    f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
    MAP=df.Function(elliptic.pde.V,name="parameter")
    f.read(MAP,"parameter")
    f.close()
except:
    print('Error encountered!')
    pass
# get estimates and REM
REM=np.zeros((num_samp,num_algs))
VAR=np.zeros((num_samp,num_algs))
samp_f=df.Function(elliptic.pde.V,name="parameter")
samp_v=elliptic.prior.gen_vector()
samp_v.zero(); csum=df.Vector(samp_v)
num_read=0
for a in range(num_algs):
    for f_i in h5_names:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                samp_v.zero(); csum.zero()
                for s in range(num_samp):
                    f.read(samp_f,'sample_{0}'.format(s))
                    samp_v[:]=samp_f.vector()
                    if any([j in algs[a] for j in ['DILI','aDRinf']]):
                        samp_v=elliptic.prior.v2u(samp_v)
                    csum.axpy(1.,samp_v)
                    num_read+=1
                    avg=df.Vector(csum)
                    avg/=num_read
                    dif=df.Vector(avg)
                    dif.axpy(-1.,MAP.vector())
#                     REM[s,a]=dif.norm('linf')
                    REM[s,a]=dif.norm('l2')/MAP.vector().norm('l2')
                    dif=df.Vector(avg)
                    dif.axpy(-1.,samp_v)
                    if s>0:
                        VAR[s,a]=((s-1)*VAR[s-1,a]+dif.norm('l2')**2)/s
                f.close()
                found[a]=True
            except:
                print('Error encountered!')
                pass
    for f_i in pckl_names:
        if '_'+algs[a]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
#                 f_read=pickle.load(f,encoding='bytes') # for python 3
                f_read=pickle.load(f)
                adjuster=-(algs[a]=='DILI')+('aDRinf' in algs[a])
                time=f_read[adjuster+5]
#                 times=f_read[adjuster+6]
                f.close()
                found[a]=True
            except:
                print('Error encountered!')
                pass
    if found[a]:
        spiter=time/2000
        nsamp_in=np.int(np.floor(max_time/spiter))
        axes[0].semilogy(np.linspace(0,max_time,num=nsamp_in),REM[:nsamp_in,a],next(linecycler0),linewidth=1.25)
        axes[1].semilogy(np.linspace(0,max_time,num=nsamp_in),VAR[:nsamp_in,a],next(linecycler1),linewidth=1.25)
#         plt_idx=times<=max_time
#         axes[0].semilogy(times[plt_idx],REM[plt_idx],next(linecycler0),linewidth=1.25)
#         axes[1].semilogy(times[plt_idx],VAR[plt_idx],next(linecycler1),linewidth=1.25)

# save for future use
np.savetxt(folder+'/rem_t.txt',REM,delimiter=',',header=','.join(algs))
np.savetxt(folder+'/var_t.txt',VAR,delimiter=',',header=','.join(algs))

plt.axes(axes[0])
plt.axis('tight')
plt.xlabel('time (seconds)',fontsize=14); plt.ylabel('REM',fontsize=14)
plt.ylim(-0.01,1.01)
plt.axes(axes[1])
# plt.axis([0,100,-1,1])
plt.axis('tight')
plt.xlabel('time (seconds)',fontsize=14); plt.ylabel('VAR',fontsize=14)
axes[1].set_ylim(ymin=1000)
# plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# add legend
h2_pos=axes[1].get_position()
plt.legend(np.array(alg_names)[found],fontsize=11,loc=2,bbox_to_anchor=(1.02,2.32),labelspacing=4)
fig.tight_layout(rect=[0,0,.85,1])
plt.savefig(folder+'/rem_t.png')

plt.show()
