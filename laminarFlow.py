from dataclasses import dataclass
from fenics import *
from dolfin import *
from mshr import *
import numpy as np

@dataclass
class caseData:
    Lx : float = -10
    Ly : float = 8
    nx : int = 40
    ny : int = 30

    inletLabel  : int = 1
    outletLabel : int = 2
    wallLabel   : int = 3

def meshGeneration(caseData):
    caseData.mesh = RectangleMesh(Point(0.0, -0.5*caseData.Ly), Point(caseData.Lx, 0.5*caseData.Ly), caseData.nx, caseData.ny)

    caseData.inlet = AutoSubDomain(lambda x, on_boundary: near(x[0], 0.0) and on_boundary)
    caseData.outlet = AutoSubDomain(lambda x, on_boundary: near(x[0], caseData.Ly) and on_boundary)
    caseData.wall = AutoSubDomain(lambda x, on_boundary: near(abs(x[1]), 0.5*caseData.Ly) and on_boundary)

    caseData.boundaries = MeshFunction("size_t", caseData.mesh, caseData.mesh.topology().dim()-1, 0)
    caseData.ds = ds(subdomain_data = caseData.boundaries)

    caseData.inlet.mark(caseData.boundaries, caseData.inletLabel)
    caseData.outlet.mark(caseData.boundaries, caseData.outletLabel)
    caseData.wall.mark(caseData.boundaries, caseData.wallLabel)

    caseData.v0 = FunctionSpace(caseData.mesh, "CG", 2)
    v_elem = VectorElement("CG", caseData.mesh.ufl_cell(), 1, 2)
    p_elem = FiniteElement("CG", caseData.mesh.ufl_cell(), 1)
    l_elem = FiniteElement("CG", caseData.mesh.ufl_cell(), 1)

    caseData.MixedSpace = FunctionSpace(caseData.mesh, MixedElement([v_elem, p_elem, l_elem]))

    bc_wall = DirichletBC(caseData.MixedSpace.sub(0).sub(1), Constant(0.0), caseData.boundaries, caseData.wallLabel)
    bc_lagrange = DirichletBC(caseData.MixedSpace.sub(2), Constant(0.0), "fabs(x[0])>2.0*DOLFIN_EPS")

    caseData.ess_bc = [bc_wall, bc_lagrange]

    def homogenize(bc):
    	bc_copy = DirichletBC(bc)
    	bc_copy.homogenize()
    	return bc_copy
    caseData.adj_bcs = [homogenize(bc) for bc in caseData.ess_bc]

    bc_inlet = DirichletBC(caseData.v0, 1, caseData.inlet)
    caseData.idx_inlet = np.array(bc_inlet.get_boundary_values().keys(), dtype = np.int32)

    try:
    	dof_coordinate = caseData.v0.tabulate_dof_coordinate()
    except AttributeError:
        print('Warning: tabulate_all_coordinates of GenericDofMap object will be deprecated after version 1.6.0!')
        dof_coordinates = caseData.v0.dofmap().tabulate_all_coordinates(caseData.mesh)
    dof_coordinates.resize((caseData.v0.dim(), caseData.mesh.geometry().dim()))
    caseData.dof_coords_inlet = dof_coordinates[caseData.idx_inlet, 1]

if __name__=="__main__":
    laminarJet = caseData()
    meshGeneration(laminarJet)



