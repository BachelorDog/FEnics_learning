"""
Plot estimates of uncertainty field m before C_mu in inverse RANS problem.
Shiwei Lan @ U of Warwick, 2016
"""

import os
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

from RANS import RANS
import sys
sys.path.append( "../../" )
from util import matplot4dolfin
matplot=matplot4dolfin()

PARAMETER =1
# define the PDE problem
nozz_w=1.;nx=40;ny=80
rans=RANS(nozz_w=nozz_w,nx=nx,ny=ny)
rans.setup(seed=2017)

# algorithms
# algs=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC','DILI','aDRinfmMALA','aDRinfmHMC')
# alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','DR-$\infty$-mMALA','DR-$\infty$-mHMC','DILI','aDR-$\infty$-mMALA','aDR-$\infty$-mHMC')
algs=('pCN','infMALA','DRinfmMALA','DILI','aDRinfmMALA')
alg_names=('pCN','$\infty$-MALA','DR-$\infty$-mMALA','DILI','aDR-$\infty$-mMALA')

num_algs=len(algs)
# preparation for estimates
folder = './analysis-L25W8-nofreshinit-yesparacont-2000'
fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
num_samp=2000

# plot
num_rows=2
fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((1+num_algs)/num_rows)),sharex=True,sharey=True,figsize=(16,6))

for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot MAP
        try:
            f=dl.HDF5File(rans.mpi_comm, os.path.join(folder,"map_solution.h5"), "r")
            MAP=dl.Function(rans.Vh_PARAMETER,name="parameter")
            f.read(MAP,"parameter")
            f.close()
            sub_fig=matplot.plot(MAP)
            ax.set_title('MAP')
        except:
            pass
    elif 1<=i<=num_algs:
        print('Working on '+alg_names[i-1]+' algorithm...')
        # plot posterior mean
        found=False
        samp_f=dl.Function(rans.Vh_PARAMETER,name="parameter")
        samp_v=rans.model_stat.generate_vector(PARAMETER)
        samp_v.zero()
        num_read=0;bad_idx=[]
        for f_i in fnames:
            if '_samp_'+algs[i-1]+'_' in f_i:
                try:
                    f=dl.HDF5File(rans.mpi_comm,os.path.join(folder,f_i),"r")
                    samp_v.zero()
                    for s in range(num_samp):
                        try:
                            f.read(samp_f,'sample_{0}'.format(s))
#                             f.read(samp_f.vector(),'/VisualisationVector/{0}'.format(s),False)
                            samp_v.axpy(1.,samp_f.vector())
                            num_read+=1
                        except:
                            bad_idx.append(s)
                    f.close()
                    if len(bad_idx)>0:
                        print('{0:d} bad samples encountered!'.format(len(bad_idx)))
                    found=True
                except:
                    pass
        if found and num_read>0:
            samp_mean=samp_v/num_read
            if any([s in algs[i-1] for s in ['DILI','aDRinf']]):
                samp_mean=rans.whtprior.v2u(samp_mean)
            samp_f.vector()[:]=samp_mean
            sub_fig=matplot.plot(samp_f)
            ax.set_title(alg_names[i-1])
#     plt.axis('tight')
    plt.axis(rans.box)

# set color bar
cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)

# save plot
# fig.tight_layout()
plt.savefig(folder+'/estimates.png',bbox_inches='tight')

plt.show()
