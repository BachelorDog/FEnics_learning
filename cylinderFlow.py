from fenics import *
from mshr import *
import numpy as np 

T = 5.0
num_steps = 5000
dt = T / num_steps
mu = 0.001
rho = 1

channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 2.2)'
walls = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1] < 0.3'

inflow_profile = ('4.0*1.5*x[1]*(0.41-x[1])/pow(0.41, 2)', '0')

bcu_inflow = DirichletBC(V,Expression(inflow_profile, degree = 2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cyliner = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cyliner]
bcp = [bcp_outflow]

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

F1 = rho*dot((u - u_n) / k, v)*dx + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx + inner(sigma(U, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

timeseries_u = TimeSeries('navier_stokes_cylinder/velocity_series')
timeseries_p = TimeSeries('navier_stokes_cylinder/pressure_series')

File('navier_stokes_cylinder/cylinder.xml.gz') << mesh
progress = Progress('Time-stepping')
set_log_level(40)

t = 0
for n in range(num_steps):
    t += dt

    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    plot(u_, title='Velocity')
    plot(p_, title='Pressure')

    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    u_n.assign(u_)
    p_n.assign(p_)

    print('u max: ', u_.vector().get_local().max())

