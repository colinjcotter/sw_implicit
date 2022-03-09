from firedrake import *
from petsc4py import PETSc
import numpy as np
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
W = V1b * U

def both(e):
    return e('+')+e('-')

w, r = TestFunctions(W)
F, v = TrialFunctions(W)
a = both(inner(w, n)*inner(F, n))*dS
a += inner(w, n)*inner(F, n)*ds
a += inner(w, v)*dx
a += inner(F, r)*dx

L = both(inner(w, n)*inner(u0, n))*hup*dS
L += inner(w, n)*inner(u0*hbar, n)*ds
L += inner(u0*hbar, r)*dx

w0 = Function(W)
myprob = LinearVariationalProblem(a, L, w0)
solver_parameters={
    "ksp_type":"preonly",
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'}

mysolver = LinearVariationalSolver(myprob,
                                   solver_parameters=solver_parameters)
mysolver.solve()

phi = TestFunction(V2)
un = 0.5*(dot(ubar, n) + abs(dot(ubar, n)))

hform = (
    -inner(grad(phi), ubar*hbar)*dx
    + (phi('+') - phi('-'))*(un('+')*hbar('+') - un('-')*hbar('-'))*dS
)

F0, v0 = w0.split()

F1 = Function(V1)

shapes = (V1.finat_element.space_dimension(),
          np.prod(V1.shape))
domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
instructions = """
for i, j
w[i,j] = w[i,j] + 1
end
"""
weight = Function(V1)
par_loop((domain, instructions), dx, {"w": (weight, INC)},
         is_loopy_kernel=True)

instructions = """
for i, j
vec_out[i,j] = vec_out[i,j] + vec_in[i,j]/w[i,j]
end
"""
average_kernel = (domain, instructions)

par_loop(average_kernel, dx,
         {"w": (weight, READ),
          "vec_in": (F0, READ),
          "vec_out": (F1, INC)},
         is_loopy_kernel=True)

Ftest = phi*div(F1)*dx - action(derivative(hform, ubar), u0)

Fass = assemble(Ftest)

import numpy as np
print(np.abs(Fass.dat.data[:]).max())
