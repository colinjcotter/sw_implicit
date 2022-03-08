from firedrake import *
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
mesh = PeriodicUnitSquareMesh(20,20)

degree = 2
BDM = FiniteElement("BDM", triangle, degree, variant="integral")
V1 = FunctionSpace(mesh, BDM)
V2 = FunctionSpace(mesh, "DG", degree-1)
Tr = FunctionSpace(mesh, "HDiv Trace", degree)
bNed = BrokenElement(FiniteElement("N1curl", triangle,
                                   degree-1, variant="integral"))
U = FunctionSpace(mesh, bNed)

hbar = Function(V2)
x, y = SpatialCoordinate(mesh)
hbar.interpolate(exp(sin(2*pi*x)*cos(2*pi*y)))
ubar = Function(V1, name="ubar")
ubar.interpolate(as_vector([-sin(2*pi*x)*cos(pi*y),
                           cos(2*pi*x)*sin(pi*y)]))
n = FacetNormal(mesh)
Upwind = 0.5 * (sign(dot(ubar, n)) + 1)
hup = Upwind('+')*hbar('+') + Upwind('-')*hbar('-')

u0 = Function(V1, name="u0")
u0.interpolate(as_vector([-sin(pi*x)*cos(pi*y),
                           cos(pi*x)*sin(pi*y)]))
W = Tr * U

gamma, w = TestFunctions(W)
F = TrialFunction(V1)

a = gamma('+')*inner(F('+'), n('+'))*dS + \
    gamma*inner(F, n)*ds
a += inner(w, F)*dx
L = gamma('+')*inner(u0('+'), n('+'))*hup*dS
L += inner(hbar*w, u0)*dx

F0 = Function(V1, name="F0")
myprob = LinearVariationalProblem(a, L, F0)
solver_parameters={
    "mat_type":"aij",
    "ksp_type":"preonly",
    "pc_type":"lu",
    "pc_factor_mat_solver_type" : "mumps"}

mysolver = LinearVariationalSolver(myprob,
                                   solver_parameters=solver_parameters)
mysolver.solve()

phi = TestFunction(V2)
un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

hform = (
    -inner(grad(phi), ubar*hbar)*dx
    + (phi('+') - phi('-'))*(un('+')*hbar('+') - un('-')*hbar('-'))*dS
)

Ftest = phi*div(F0)*dx - action(derivative(hform, ubar), u0)

Fass = assemble(Ftest)

import numpy as np
print(np.abs(Fass.dat.data[:]).max())
