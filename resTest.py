from fenics import *
from dolfin import *

mesh = UnitIntervalMesh(1)
V = FunctionSpace(mesh,"CG",1)
A = assemble(TrialFunction(V)*TestFunction(V)*dx)
b = assemble(TestFunction(V)*dx)
x = Function(V).vector()

print(residual(A,x,b))
print(norm(A*x-b))
