from dataclasses import dataclass
from fenics import *
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
u_test = TestFunction(V)
u = TrialFunction(V)
u_n = Function(V)
u_ = Function(V)
nut = Function(V)

P = FiniteElement('CG', interval, 2)
element = MixedElement([P, P])
T = FunctionSpace(channelFlow.mesh, element)

k_test, w_test = TestFunctions(T)
k, w = TrialFunctions(T)

T_n = Function(T)
k_n, w_n = T_n.split()
T_ = Function(T)
k_, w_ = T_.split()

u_ = project(Constant("0.0"), V)
k_ = project(Constant("1e-7"), V)
w_ = project(Constant("32000"), V)

dt = 0.01
delta = Constant(dt)

gradP = Constant(channelFlow.frictionVelocity**2)

mu = Constant(channelFlow.mu)
beta = Constant(channelFlow.beta)
betaStar = Constant(channelFlow.betaStar)
sigma = Constant(channelFlow.sigma)
sigmaStar = Constant(channelFlow.sigmaStar)
gamma = Constant(channelFlow.gamma)

bc_u = DirichletBC(V, Constant(0.0), boundary)
bc_k = DirichletBC(T.sub(0), Constant(0.0), boundary)
bc_w = DirichletBC(T.sub(1), Expression("min(2.352e14,  6*mu/(beta*pow(abs(x[0])-1, 2)))", degree = 1, mu = channelFlow.mu, beta = channelFlow.beta), wboundary)


for i in range(100):

    F1 = (mu+k_/w_)*dot(grad(u), grad(u_test))*dx - gradP*u_test*dx

    a1 = lhs(F1)
    L1 = rhs(F1)

    A1 = assemble(a1)
    b1 = assemble(L1)

    bc_u.apply(A1)
    bc_u.apply(b1)

    solve(A1, u_n.vector(), b1)
    u_ = project(0.6*u_+0.4*u_n, V)

    for j in range(20):
        F2 = (k_/w_)*dot(dot(grad(u_), grad(u_)), k_test)*dx - betaStar*k*w_*k_test*dx - (mu + sigmaStar*k_/w_)*dot(grad(k), grad(k_test))*dx + (mu + sigmaStar*k_/w_)*dot(grad(k), (k_test*n))*ds
        F3 = gamma*dot(dot(grad(u_), grad(u_)), w_test)*dx - beta*w*w_*w_test*dx - (mu + sigma*k_/w_)*dot(grad(w), grad(w_test))*dx + (mu + sigma*k_/w_)*dot(grad(w), (w_test*n))*ds

        F = F2 + F3

        a2 = lhs(F)
        L2 = rhs(F)

        A2 = assemble(a2)
        b2 = assemble(L2)

        bc_k.apply(A2)
        bc_w.apply(A2)
        bc_k.apply(b2)
        bc_w.apply(b2)

        solve(A2, T_n.vector(), b2)

        T_.assign(T_n)
        k_n, w_n = T_.split()

        w_ = project(0.6*w_+0.4*w_n, V)
        k_ = project(0.6*k_+0.4*k_n, V)        
        nut = project(k_/w_, V)
        
        res = residual(A2, T_n.vector(), b2)
        #print("residual: ", res)

    plot(u_)
    plt.show()
    # plot(k_)
    # plt.show()
    # plot(w_)
    # plt.show()
    plot(nut)
    plt.show()

    print(nut(0))
