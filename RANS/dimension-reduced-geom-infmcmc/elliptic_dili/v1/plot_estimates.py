"""
Plot estimates of uncertainty field m before C_mu in inverse RANS problem.
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
np.random.seed(2016)
elliptic = Elliptic(nx=40,ny=40,SNR=10)

# algorithms
algs=('pCN','infMALA','infHMC','infmMALA','infmHMC','drinfmMALA','drinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','$\infty$-mMALA','$\infty$-mHMC','DR$\infty$-mMALA','DR$\infty$-mHMC')
num_algs=len(algs)
# preparation for estimates
folder = './analysis'
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
            f=df.HDF5File(elliptic.pde.mpi_comm, os.path.join(folder,"map_solution.h5"), "r")
            MAP=df.Function(elliptic.pde.V,name="parameter")
            f.read(MAP,"parameter")
            f.close()
            sub_fig=matplot.mplot(MAP)
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
            samp_f.vector()[:]=samp_v/num_read
            sub_fig=matplot.mplot(samp_f)
            ax.set_title(alg_names[i-1])
    plt.axis([0, 1, 0, 1])

# set color bar
cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)

# save plot
# fig.tight_layout()
plt.savefig('./analysis/estimates.png',bbox_inches='tight')

plt.show()
