"""
Test Karhunen-Loeve expansion of coefficient function
Shiwei Lan @ U of Warwick, 2016
"""


from dolfin import *
import numpy as np
import time
import scipy.sparse as sps
from petsc4py import PETSc
import matplotlib.pyplot as plt
from matplotlib import animation
import os
# plt.rcParams['animation.ffmpeg_path'] = os.path.join(os.getcwd(),'ffmpeg')
plt.rcParams['animation.ffmpeg_path'] = '/Users/LANZI/ffmpeg/ffmpeg'

from Elliptic import Elliptic


# parameters["num_threads"] = 2

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_petsc4py():
    print("DOLFIN has not been configured with petsc4py. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# np.random.seed(2016)

# settings
dim=100
# choice of coefficient definition
# kl_opt='fb'
kl_opt='kf'


# define elliptic model
elliptic=Elliptic(nx=40,ny=40)
theta=np.random.randn(dim)
# get coefficient
coeff=elliptic.coefficient(theta,kl_opt=kl_opt)

# plot
parameters["plotting_backend"]="matplotlib"
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
coeff_fun,_,_=coeff.get_coeff()
heat = plot(coeff_fun)

# animation function.  This is called sequentially
def update(i,ax,fig):
    ax.cla()
    # update coefficient function
    theta=np.random.randn(dim)
    coeff.theta=theta
    coeff_fun,_,_=coeff.get_coeff()
    heat = plot(coeff_fun)
    return heat

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, update, frames=xrange(100), fargs=(ax, fig), interval=100)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# FFwriter = animation.FFMpegWriter()
# anim.save('../result/random_coefficient.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
