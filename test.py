import matplotlib.pyplot as plt
from dolfin import *;
from mshr import *;
import time
import logging;
logging.getLogger('FFC').setLevel(logging.WARNING)


######## Important code begins here ########

# Define domain and mesh
XMIN, XMAX = -2., 2.;
YMIN, YMAX = -2., 2.;
G = [XMIN, XMAX, YMIN, YMAX];
mresolution = 20;
mesh = generate_mesh(Rectangle(Point(G[0], G[2]), Point(G[1], G[3])), mresolution)

# Define finite element function space
degree = 1;
V = FunctionSpace(mesh, "CG", degree);

# Finite element functions
v = TestFunction(V);
u = Function(V);

# Define boundary conditions
def boundary(x, on_boundary):
    return on_boundary

# Time parameters
theta = 1.0 # Implicit Euler
k = 0.5; # Time step
t, T = 0., 20.; # Start and end time

# Exact solution
k_coeff = 1
ue = Expression("exp(-(x[0]*x[0]+x[1]*x[1])/(4*a*t))/(4*pi*a*t)", degree=2, a=k_coeff, t=1e-7, domain=mesh)
u0 = ue;

bc = DirichletBC(V, ue, boundary)

# Inititalize time stepping
pl, ax = plt.subplots();
stepcounter = 0;
timer0 = time.clock()

# Time-stepping loop
while t < T:
    # Time scheme
    um = theta*u + (1.0-theta)*u0

    # Weak form of the heat equation in residual form
    r = (u - u0)/k*v*dx + k_coeff*inner(grad(um), grad(v))*dx

    # Solve the Heat equation (one timestep)
    solve(r==0, u, bc)

    # Shift to next timestep
    t += k; u0 = project(u, V);
    ue.t = t;
    stepcounter += 1

print("elapsed CPU time: ", (time.clock() - timer0))
