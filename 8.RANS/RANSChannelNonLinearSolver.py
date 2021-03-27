from dataclasses import dataclass
from fenics import *
from fenics_adjoint import *
import numpy as np
from numpy import arctan, array, power
import matplotlib.pyplot as plt

u_obs = np.loadtxt(open("550.csv", "rb"), delimiter=',', skiprows=0)

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
nut = Function(V)

P = FiniteElement('CG', interval, 2)
element = MixedElement([P, P, P])
T = FunctionSpace(channelFlow.mesh, element)

u_test, k_test, w_test = TestFunctions(T)
u, k, w = TrialFunctions(T)

T_n = Function(T)
u_n, k_n, w_n = split(T_n)
T_ = project(Expression(("0.0", "1e-7", "32000"), degree = 1), T)
u_, k_, w_ = split(T_)

relax = 0.6

gradP = Constant(channelFlow.frictionVelocity**2)

mu = Constant(channelFlow.mu)
beta = Constant(channelFlow.beta)
betaStar = Constant(channelFlow.betaStar)
sigma = Constant(channelFlow.sigma)
sigmaStar = Constant(channelFlow.sigmaStar)
gamma = Constant(channelFlow.gamma)
yEpsilon = project(Expression("1.0", name='Control', degree=1), V)

control = Control(yEpsilon)

bc_u = DirichletBC(T.sub(0), Constant(0.0), boundary)
bc_k = DirichletBC(T.sub(1), Constant(0.0), boundary)
bc_w = DirichletBC(T.sub(2), Expression("min(2.352e14,  6*mu/(beta*pow(abs(x[0])-1, 2)))", degree = 1, mu = channelFlow.mu, beta = channelFlow.beta), wboundary)
bc = [bc_u, bc_k, bc_w]

F1 = (mu+k_/w_)*dot(grad(u), grad(u_test))*dx - gradP*u_test*dx
F2 = (k_/w_)*dot(dot(grad(u_), grad(u_)), k_test)*dx - betaStar*k*w_*k_test*dx - (mu + sigmaStar*k_/w_)*dot(grad(k), grad(k_test))*dx + (mu + sigmaStar*k_/w_)*dot(grad(k), (k_test*n))*ds
F3 = yEpsilon*gamma*dot(dot(grad(u_), grad(u_)), w_test)*dx - beta*w*w_*w_test*dx - (mu + sigma*k_/w_)*dot(grad(w), grad(w_test))*dx + (mu + sigma*k_/w_)*dot(grad(w), (w_test*n))*ds

F = F1 + F2 + F3

a = lhs(F)
L = rhs(F)

for i in range(200):
    solve(a == L, T_n, bc)
    T_.assign(project(as_vector((0.8*u_+0.2*u_n, 0.8*k_+0.2*k_n, 0.8*w_+0.2*w_n)), T))

    # update = project(u_n-u_, V)
    # error = np.linalg.norm(update.vector().get_local())
    # if error < 1e-6:
    #     break

plot(u_)
plt.show()
J = assemble(inner(u_, u_)*dx)
dJdEpsilon = compute_gradient(J, control)
plot(dJdEpsilon)
plt.show()

