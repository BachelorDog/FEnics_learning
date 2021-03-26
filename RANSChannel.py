from dataclasses import dataclass
from fenics import *
from fenics_adjoint import *
from numpy import arctan, array, power
import matplotlib.pyplot as plt

@dataclass
class caseData:
    mesh : Mesh = Mesh()

    inletLabel  : int = 1
    outletLabel : int = 2
    wallLabel   : int = 3

    frictionVelocity : float = 5.43496e-2
    mu               : float = 1e-4
    beta             : float = 3/40
    betaStar         : float = 0.09
    sigma            : float = 0.5
    sigmaStar        : float = 0.5
    gamma            : float = 5/9


def meshGenerate(caseData):
    caseData.mesh = UnitIntervalMesh(198)
    yCoordinate = caseData.mesh.coordinates()
    yCoordinate[:] = yCoordinate[:]*2-1
    yCoordinate[yCoordinate<0] = (power(1+550, yCoordinate[yCoordinate<0]+1)-1)/550 - 1
    yCoordinate[yCoordinate>0] = -(power(1+550, 1-yCoordinate[yCoordinate>0])-1)/550 + 1

channelFlow = caseData()
meshGenerate(channelFlow)
n = FacetNormal(channelFlow.mesh)

def boundary(x, on_boundary):
    return on_boundary

def wboundary(x, on_boundary):
    tol = 0.005
    return on_boundary or near(x[0], -1, tol) or near(x[0], 1, tol)

V = FunctionSpace(channelFlow.mesh, 'CG', 2)
u_trial = TrialFunction(V)
u_test = TestFunction(V)
u_n = Function(V)
u_ = Function(V)

k_trial = TrialFunction(V)
k_test = TestFunction(V)
k_n = Function(V)
k_ = Function(V)

w_trial = TrialFunction(V)
w_test = TestFunction(V)
w_n = Function(V)
w_ = Function(V)

u_n = project(Expression("0.0", degree = 1), V)
k_n = project(Expression("1e-7", degree = 1), V)
w_n = project(Expression("32000", degree = 1), V)

nut = Function(V)


#control = Control(yEpsilon)
relax = 0.6

gradP = Constant(channelFlow.frictionVelocity**2)

mu = Constant(channelFlow.mu)
beta = Constant(channelFlow.beta)
betaStar = Constant(channelFlow.betaStar)
sigma = Constant(channelFlow.sigma)
sigmaStar = Constant(channelFlow.sigmaStar)
gamma = Constant(channelFlow.gamma)
control = Control(gamma)
yEpsilon = project(Constant("1.0"), V)

bc_u = DirichletBC(V, Constant(0.0), boundary)
bc_k = DirichletBC(V, Constant(0.0), boundary)
bc_w = DirichletBC(V, Expression("min(2.352e14,  6*mu/(beta*pow(abs(x[0])-1, 2)))", degree = 1, mu = channelFlow.mu, beta = channelFlow.beta), wboundary)

F1 = (mu+k_n/w_n)*dot(grad(u_trial), grad(u_test))*dx - gradP*u_test*dx
F2 = (k_n/w_n)*dot(dot(grad(u_n), grad(u_n)), k_test)*dx - betaStar*k_n*w_n*k_test*dx - (mu + sigmaStar*k_n/w_n)*dot(grad(k_trial), grad(k_test))*dx + (mu + sigmaStar*k_n/w_n)*dot(grad(k_trial), (k_test*n))*ds
F3 = gamma*dot(dot(grad(u_n), grad(u_n)), w_test)*dx - beta*w_n*w_n*w_test*dx - (mu + sigma*k_n/w_n)*dot(grad(w_trial), grad(w_test))*dx + (mu + sigma*k_n/w_n)*dot(grad(w_trial), (w_test*n))*ds

a1 = lhs(F1)
L1 = rhs(F1)

a2 = lhs(F2)
L2 = rhs(F2)

a3 = lhs(F3)
L3 = rhs(F3)

for i in range(10):
    solve(a1 == L1, u_, bc_u)
    plot(u_)
    plt.show()
    u_n.assign((1-relax)*u_ + relax*u_n)

    solve(a3 == L3, w_, bc_w)
    plot(w_)
    plt.show()
    #w_n.assign((1-relax)*w_ + relax*w_n)

    solve(a2 == L2, k_, bc_k)
    plot(k_)
    plt.show()
    k_n.assign((1-relax)*k_ + relax*k_n)

    nut = project(k_/w_, V)


#J = assemble(dot(u, u)*dx)
#dJdEpsilon = compute_gradient(J, control)