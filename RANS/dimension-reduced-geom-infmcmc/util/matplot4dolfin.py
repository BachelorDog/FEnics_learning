#!/usr/bin/env python
"""
Class definition of plotting dolfin type objects using matplotlib.
This has been already implemented after FEniCS 1.6.0.
Shiwei Lan @ U of Warwick, 2016
"""
import os
import dolfin as df
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri

class matplot4dolfin:
    """
    generic function plotting solution over 2D mesh
    by Chris Richardson @ https://bitbucket.org/fenics-project/dolfin/issues/455/add-ipython-compatible-matplotlib-plotting
    already incorporated in version 1.7.0
    """
    def __init__(self,overloaded=None):
        if overloaded is None:
            if df.DOLFIN_VERSION_MAJOR<=1 and df.DOLFIN_VERSION_MINOR<=6:
                self.overloaded=False
            else:
                self.overloaded=True
                print('Warning: plot has been overloaded with matplotlib after version 1.6.0! Check the parameter "plotting_backend".')
        else:
            self.overloaded=overloaded
        if self.overloaded:
            df.parameters["plotting_backend"]="matplotlib"
    
    def mesh2triang(self,mesh):
        xy = mesh.coordinates()
        return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    def mplot_cellfunction(self,cellfn):
        C = cellfn.array()
        tri = self.mesh2triang(cellfn.mesh())
        return plt.tripcolor(tri, facecolors=C)

    def mplot_function(self,f):
        mesh = f.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise AttributeError('Mesh must be 2D')
        # DG0 cellwise function
        if f.vector().size() == mesh.num_cells():
            C = f.vector().array()
            return plt.tripcolor(self.mesh2triang(mesh), C)
        # Scalar function, interpolated to vertices
        elif f.value_rank() == 0:
            C = f.compute_vertex_values(mesh)
            return plt.tripcolor(self.mesh2triang(mesh), C, shading='gouraud')
        # Vector function, interpolated to vertices
        elif f.value_rank() == 1:
            w0 = f.compute_vertex_values(mesh)
            if (len(w0) != 2*mesh.num_vertices()):
                raise AttributeError('Vector field must be 2D')
            X = mesh.coordinates()[:, 0]
            Y = mesh.coordinates()[:, 1]
            U = w0[:mesh.num_vertices()]
            V = w0[mesh.num_vertices():]
            return plt.quiver(X,Y,U,V)

    # Plot a generic dolfin object (if supported)
    def plot(self,obj,**kwargs):
        if self.overloaded:
            return df.plot(obj,**kwargs)
        else:
#             plt.gca().set_aspect('equal')
            if isinstance(obj, df.Function):
                return self.mplot_function(obj)
            elif isinstance(obj, df.CellFunctionSizet):
                return self.mplot_cellfunction(obj)
            elif isinstance(obj, df.CellFunctionDouble):
                return self.mplot_cellfunction(obj)
            elif isinstance(obj, df.CellFunctionInt):
                return self.mplot_cellfunction(obj)
            elif isinstance(obj, df.Mesh):
                if (obj.geometry().dim() != 2):
                    raise AttributeError('Mesh must be 2D')
                return plt.triplot(self.mesh2triang(obj), color='#808080')
            else:
                raise AttributeError('Failed to plot %s'%type(obj))
    
    def save(self,savepath=None,filename='fig.png',**kwargs):
        if not os.path.exists(savepath):
            print('Save path does not exist; created one.')
            savepath=os.path.join(os.getcwd(),'result')
            os.makedirs(savepath)
        plt.savefig(os.path.join(savepath,filename),**kwargs)
    
    def show(self):
        plt.show()