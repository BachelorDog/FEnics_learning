#!/usr/bin/env python
"""
Add a common color bar to a group of subplots:
- The color limits are read from those subplots;
- Common color limits are set and adjusted for each of the subplots;
- A common color bar is added to the right of the subplots.
Shiwei Lan @ CalTech, 2017
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def common_colorbar(fig,axes,sub_figs):
    clims=np.array([sub_fig.get_clim() for sub_fig in sub_figs if sub_fig])
    common_clim=np.min(clims[:,0]),np.max(clims[:,1])
    for ax,sub_fig in zip(axes.flat,sub_figs):
        plt.axes(ax)
        if sub_fig:
            sub_fig.set_clim(common_clim)
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
#     common_colbar=plt.colorbar(sub_fig, cax=cax,**kw)
    norm = mpl.colors.Normalize(vmin=common_clim[0], vmax=common_clim[1])
    common_colbar = mpl.colorbar.ColorbarBase(cax,norm=norm)
    return fig