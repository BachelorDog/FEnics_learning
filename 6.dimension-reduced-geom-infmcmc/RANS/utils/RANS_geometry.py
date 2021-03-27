'''
Created on Feb 18, 2016

@author: uvilla
'''

import dolfin as dl
from helpers_meshing import GradingFunctionLin, Remap

class FreeJet_Geometry:
    
    INLET    = 1
    AXIS     = 2
    FARFIELD = 3
    
    DNS = 1
    OUTSIDE = 2
    """
    This class creates a structured anistropic mesh for the turbulent jet problem.
    The mesh is uniform in the x-direction and graded in the y-direction
    (finer closer to the centerline, and coarser far from the centerline)
    
    The geometry is a rectagle with the following boundary labels:
    
                 FARFIELD
           ____________________________
          |                            |
    INLET |                            |  FARFIELD
          |____________________________|
                    AXIS
                    
    The attributes of the class are:
    - box = [x_min, x_max, y_min, y_max]: the bounding box of the rectagle
    - mesh: the finite element mesh
    - boundary_parts: the marker of boundary labels
    - ds: the integrator for each subdomain of the boundary
    """
    
    def __init__(self, box, nx, ny):
        """
        Constructor
        
        INPUTS:
        - box = [x_min, x_max, y_min, y_max]: the bounding box of the computational domain
        - nx, ny: number of elements in the horizontal (axial) and vertical (transversal) direction
        """
        self.box = box
        self.mesh = dl.UnitSquareMesh(nx,ny)
        
        grade = GradingFunctionLin(coordinate=1, cut_point=[.6, .7], slope=6)
        remap = Remap(box=box)
        
        self.mesh.move(grade)
        self.mesh.move(remap)
        
        class InletBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0]-box[0]) < dl.DOLFIN_EPS
            
        class SymmetryBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[1]-box[2]) < dl.DOLFIN_EPS
            
        class FarfieldBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and ( abs(x[0]-box[1]) < dl.DOLFIN_EPS or abs( x[1] - box[3]) < dl.DOLFIN_EPS )
            
        self.boundary_parts = dl.FacetFunction("size_t", self.mesh)
        self.boundary_parts.set_all(0)
        
        Gamma_inlet = InletBoundary()
        Gamma_inlet.mark(self.boundary_parts, self.INLET)
        Gamma_axis = SymmetryBoundary()
        Gamma_axis.mark(self.boundary_parts, self.AXIS)
        Gamma_farfield = FarfieldBoundary()
        Gamma_farfield.mark(self.boundary_parts, self.FARFIELD)
        
        self.ds = dl.Measure("ds")[self.boundary_parts]
        
        class DNSDomain(dl.SubDomain):
            def inside(self,x, on_boundary):
                return x[0] < 20. + dl.DOLFIN_EPS
        
        self.domain_parts = dl.CellFunction("size_t", self.mesh)
        self.domain_parts.set_all(self.OUTSIDE)
        DNS_Domain = DNSDomain()
        DNS_Domain.mark(self.domain_parts, self.DNS)
        
        self.dx = dl.Measure("dx")[self.domain_parts]

class FreeJetSponge_Geometry:
    
    INLET    = 1
    AXIS     = 2
    OUTLET   = 3
    TOP      = 4
    
    PHYSICAL = 1
    SPONGE   = 2
    
    """
    This class creates a structured anistropic mesh for the turbulent jet problem.
    The mesh is uniform in the x-direction and graded in the y-direction
    (finer closer to the centerline, and coarser far from the centerline)
    
    The geometry is a rectagle with the following boundary labels:
    
                 TOP
           ____________________________
          | ______________________     |
    INLET |    Physical           |    |  OUTLET
          |_______________________|____|
                    AXIS
                    
    The attributes of the class are:
    - box = [x_min, x_max, y_min, y_max]: the bounding box of the rectagle
    - mesh: the finite element mesh
    - boundary_parts: the marker of boundary labels
    - ds: the integrator for each subdomain of the boundary
    """
    
    def __init__(self, box, sponge, nx, ny):
        """
        Constructor
        
        INPUTS:
        - box = [x_min, x_max, y_min, y_max]: the bounding box of the computational domain
        - nx, ny: number of elements in the horizontal (axial) and vertical (transversal) direction
        """
        self.box = box
        self.mesh = dl.UnitSquareMesh(nx,ny)
        box_sponge = [box[0], box[1]+sponge[0], box[2], box[3]+sponge[1]]
        
        grade = GradingFunctionLin(coordinate=1, cut_point=[.6, .7], slope=6)
        remap = Remap(box=box_sponge)
        
        self.mesh.move(grade)
        self.mesh.move(remap)
        
        class InletBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0]-box_sponge[0]) < dl.DOLFIN_EPS
            
        class SymmetryBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[1]-box_sponge[2]) < dl.DOLFIN_EPS
            
        class OutletBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs(x[0]-box_sponge[1]) < dl.DOLFIN_EPS
            
        class TopBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and abs( x[1] - box_sponge[3]) < dl.DOLFIN_EPS
            
        self.boundary_parts = dl.FacetFunction("size_t", self.mesh)
        self.boundary_parts.set_all(0)
        
        Gamma_inlet = InletBoundary()
        Gamma_inlet.mark(self.boundary_parts, self.INLET)
        Gamma_axis = SymmetryBoundary()
        Gamma_axis.mark(self.boundary_parts, self.AXIS)
        Gamma_outlet = OutletBoundary()
        Gamma_outlet.mark(self.boundary_parts, self.OUTLET)
        Gamma_top = TopBoundary()
        Gamma_top.mark(self.boundary_parts, self.TOP)
        
        self.ds = dl.Measure("ds")[self.boundary_parts]
        
        class PhysicalDomain(dl.SubDomain):
            def inside(self,x, on_boundary):
                return x[0] < box[1] + dl.DOLFIN_EPS and x[1] < box[3] + dl.DOLFIN_EPS
        
        self.domain_parts = dl.CellFunction("size_t", self.mesh)
        self.domain_parts.set_all(self.SPONGE)
        
        P_Domain = PhysicalDomain()
        P_Domain.mark(self.domain_parts, self.PHYSICAL)
        
        self.dx = dl.Measure("dx")[self.domain_parts]
        
        self.xfun, self.yfun = dl.MeshCoordinates(self.mesh)
        self.x_start = dl.Constant(box[1]+.5*sponge[0])
        self.x_width = dl.Constant(.5*sponge[0])
        self.y_start = dl.Constant(box[3]+.5*sponge[1])
        self.y_width = dl.Constant(.5*sponge[1])
        
        self.s_x = dl.Constant(1.) + ( (dl.Constant(100)/self.x_width)*dl.max_value(self.xfun - self.x_start, dl.Constant(0.)) )**2
        self.s_y = dl.Constant(1.) + ( (dl.Constant(100)/self.y_width)*dl.max_value(self.yfun - self.y_start, dl.Constant(0.)) )**2
        self.sponge_fun = self.s_x * self.s_y

