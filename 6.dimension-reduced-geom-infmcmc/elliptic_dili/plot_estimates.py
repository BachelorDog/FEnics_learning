"""
Plot estimates of uncertainty field u in Elliptic inverse problem.
Shiwei Lan @ U of Warwick, 2016
"""

import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

from Elliptic_dili import Elliptic
import sys
sys.path.append( "../" )
from util import matplot4dolfin
matplot=matplot4dolfin()

# define the inverse problem
np.random.seed(2017)
SNR=100
elliptic = Elliptic(nx=40,ny=40,SNR=SNR)

# algorithms
algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','DR-$\infty$-mMALA','DR-$\infty$-mHMC','DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
num_algs=len(algs)
# preparation for estimates
folder = './analysis_f_SNR'+str(SNR)
fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
num_samp=2000

# plot
num_rows=3
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(10,7.7))
# sub_figs = [None]*len(axes.flat)

for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot MAP
        try:
            f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join('./result',"MAP_SNR"+str(SNR)+".h5"), "r")
            MAP=df.Function(elliptic.pde.V,name="parameter")
            f.read(MAP,"parameter")
            f.close()
            sub_fig=matplot.plot(MAP)
            ax.set_title('MAP')
        except:
            pass
    elif 1<=i<=num_algs:
        # plot posterior mean
        found=False
        samp_f=df.Function(elliptic.pde.V,name="parameter")
        samp_v=elliptic.prior.gen_vector()
        samp_v.zero()
        num_read=0
        for f_i in fnames:
            if '_'+algs[i-1]+'_' in f_i:
                try:
                    f=df.HDF5File(elliptic.pde.mpi_comm,os.path.join(folder,f_i),"r")
                    samp_v.zero()
                    for s in range(num_samp):
                        f.read(samp_f,'sample_{0}'.format(s))
#                         f.read(samp_f.vector(),'/VisualisationVector/{0}'.format(s),False)
                        samp_v.axpy(1.,samp_f.vector())
                        num_read+=1
                    f.close()
                    found=True
                except:
                    pass
        if found:
            samp_mean=samp_v/num_read
            if any([s in algs[i-1] for s in ['DILI','aDRinf']]):
                samp_mean=elliptic.prior.v2u(samp_mean)
            samp_f.vector()[:]=samp_mean
            sub_fig=matplot.plot(samp_f)
            ax.set_title(alg_names[i-1])
    plt.axis([0, 1, 0, 1])

# set color bar
cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)

# save plot
# fig.tight_layout()
plt.savefig(folder+'/estimates.png',bbox_inches='tight')

plt.show()
