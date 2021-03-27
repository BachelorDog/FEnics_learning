"""This program solves the Elliptic PDE in the DILI paper by Cui et~al (2016)
Shiwei Lan @ University of Warwick, 2016
"""

from dolfin import *
# from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Geometry
nx = 100
ny = 100

mesh = UnitSquareMesh(nx, ny)

# boundaries
boundaries = FacetFunction("size_t", mesh)
ds = ds(subdomain_data=boundaries)

# 2. Define the finite element spaces
V = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, V * R)

# 3. Define boundary conditions
# subclass of Expression with varying parameters
class coeff(Expression):
    def __init__(self,theta,sigma=1.0,alpha=0.01,s=1.1,**kwargs):
        self.theta=theta
        self.sigma=sigma
        self.alpha=alpha
        self.s=s
        self.l=len(self.theta)
        self.lx=np.ceil(sqrt(self.l))
        self.ly=np.ceil(self.l/self.lx)
    # K-L expansion of theta ~ GP(0,C)
    def eval(self,value,x):
        seq0lx = np.arange(self.lx,dtype=np.float)
        seq0ly = np.arange(self.ly,dtype=np.float)
        eigv = self.sigma*pow(self.alpha+pi**2*(seq0lx[:,None]**2+seq0ly[None,]**2),-self.s/2)
        eigf = np.cos(pi*x[0]*seq0lx[None,:,None]) * np.cos(pi*x[1]*seq0ly[None,None,])
        value[0] = np.exp((np.reshape(eigf,(-1,self.l))*eigv.flatten()).dot(self.theta))

# theta = np.random.randn(100)
# kappa = coeff(theta,degree=2)
# kappa = Expression('sin(x[0])', degree=2)

truth_area1 = lambda x: .6<= x[0] <=.8 and .2<= x[1] <=.4
truth_area2 = lambda x: (.8-.3)**2<= (x[0]-.8)**2+(x[1]-.2)**2 <=(.8-.2)**2 and x[0]<=.8 and x[1]>=.2
class truth(Expression):
    def eval(self,value,x):
        if truth_area1(x) or truth_area2(x):
            value[0] = exp(-1)
        else:
            value[0] = 1
kappa = truth(degree=0)

# constraint on Lagrange multiplier
bc_lagrange = DirichletBC(W.sub(1), Constant(0.0), "fabs(x[0])>2.0*DOLFIN_EPS & fabs(x[0]-1.0)>2.0*DOLFIN_EPS & fabs(x[1])>2.0*DOLFIN_EPS & fabs(x[1]-1.0)>2.0*DOLFIN_EPS")

ess_bc = [bc_lagrange]

# 4. Define variational problem
class source_term(Expression):
    def __init__(self,mean=np.array([[0.3,0.3], [0.7,0.3], [0.7,0.7], [0.3,0.7]]),sd=0.05,wts=[2,-3,3,-2],**kwargs):
        self.mean=mean
        self.sd=sd
        self.wts=wts
    def eval(self,value,x):
        pdfs=1/(2*pi*self.sd**2)*np.exp(-.5*np.sum((self.mean-x)**2,axis=1)/self.sd**2)
        value[0]=pdfs.dot(self.wts)

(p, c) = TrialFunction(W)
(v, d) = TestFunctions(W)
f = source_term(degree=2)
a = kappa*inner(grad(p), grad(v))*dx + (c*v + p*d)*ds
L = f*v*dx

# 5. Compute solution
w = Function(W)
solve(a == L, w, ess_bc)
(p, c) = w.split(True)

# Save to file
# File('./result/states.xml')<<p

# Plot solution
parameters["plotting_backend"]="matplotlib"
plt.figure(0)
fig=plot(p)
plt.colorbar(fig)
# plot(p)

# Hold plot
plt.show()
# interactive()
