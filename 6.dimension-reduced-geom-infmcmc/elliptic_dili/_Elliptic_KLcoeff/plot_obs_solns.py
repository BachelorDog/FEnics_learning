"""
Plot observations and some solutions of Elliptic PDE model (DILI; Cui et~al, 2016)
Shiwei Lan @ U of Warwick, 2016
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

from Elliptic import Elliptic

np.random.seed(2016)

# define the PDE problem
elliptic=Elliptic(nx=40,ny=40)
# obtain observations using true coefficient function
obs,idx,loc,_=elliptic.get_obs()

# plot
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,figsize=(14,5))

# plot observations
plt.axes(axes[0])
parameters["plotting_backend"]="matplotlib"
plot(elliptic.mesh)
plt.plot(loc[:,0],loc[:,1],'bo',markersize=10)
plt.axis('tight')
plt.xlabel('x',fontsize=12); plt.ylabel('y',fontsize=12)
plt.title('Observations on selected locations',fontsize=12)

# plot solutions
plt.axes(axes[1])
# elliptic.nx*=2; elliptic.ny*=2
# elliptic.set_FEM(); elliptic.set_forms(elliptic.true_coeff())
# _,_=elliptic.soln_fwd()
u_fwd,_=elliptic.states_fwd.split(True)
sub_fig=plot(u_fwd)
# plt.colorbar(sub_fig)
plt.axis('tight')
plt.xlabel('x',fontsize=12); plt.ylabel('y',fontsize=12)
plt.title('Solution of potential',fontsize=12)

cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(sub_fig, cax=cax, **kw)

# fig.tight_layout()
plt.savefig('./result/obs_solns.png',bbox_inches='tight')

plt.show()
