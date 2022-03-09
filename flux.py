from firedrake import *
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
mesh = PeriodicUnitSquareMesh(20,20)

degree = 2
BDM = FiniteElement("BDM", triangle, degree, variant="integral")
V1 = FunctionSpace(mesh, BDM)
V1b = FunctionSpace(mesh, BrokenElement(BDM))
V2 = FunctionSpace(mesh, "DG", degree-1)
bNed = BrokenElement(FiniteElement("N1curl", triangle,
                                   degree-1, variant="integral"))
Tr = FunctionSpace(mesh, "HDivT", degree)
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
W = V1b * U * Tr

def both(e):
    return e('+')+e('-')

w, r, gamma = TestFunctions(W)
F, v, lambda0 = TrialFunctions(W)
a = both(inner(w, n)*inner(F, n))*dS
a += inner(w, n)*inner(F, n)*ds
a += inner(w, v)*dx
a += inner(F, r)*dx
a += jump(w, n)*lambda0('+')*dS
a += inner(w, n)*lambda0*ds
a += jump(F, n)*gamma('+')*dS
a += inner(F, n)*gamma*ds

L = both(inner(w, n)*inner(u0, n))*hup*dS
L += inner(w, n)*inner(u0*hbar, n)*ds
L += inner(u0*hbar, r)*dx

w0 = Function(W)
myprob = LinearVariationalProblem(a, L, w0)
solver_parameters={
    "mat_type":"matfree",
    "ksp_type":"preonly",
    'pmat_type':'matfree',
    'pc_type':'python',
    'pc_python_type':'firedrake.SCPC',
    'pc_sc_eliminate_fields': '0, 1',
    'condensed_field': {'ksp_type': 'gmres',
                        'ksp_monitor': None,
                        'ksp_converged_reason': None,
                        'pc_type': 'ilu'}}


mysolver = LinearVariationalSolver(myprob,
                                   solver_parameters=solver_parameters)
mysolver.solve()

phi = TestFunction(V2)
un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

hform = (
    -inner(grad(phi), ubar*hbar)*dx
    + (phi('+') - phi('-'))*(un('+')*hbar('+') - un('-')*hbar('-'))*dS
)

F0, v0 = wo.split()

Ftest = phi*div(F0)*dx - action(derivative(hform, ubar), u0)

Fass = assemble(Ftest)

import numpy as np
print(np.abs(Fass.dat.data[:]).max())
