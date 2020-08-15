import firedrake as fd

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
ref_level = 3

mesh = fd.IcosahedralSphereMesh(radius=R0,
                                refinement_level=ref_level, degree=3)
R0 = fd.Constant(R0)
cx = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(cx)

cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


V1 = fd.FunctionSpace(mesh, "BDM", 2)
V2 = fd.FunctionSpace(mesh, "DG", 1)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)

# Set up the exponential operator
operator_in = fd.Function(W)
u_in, eta_in = fd.split(operator_in)

# D = eta + b

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Unp1)
uh = 0.5*(u0 + u1)
hh = 0.5*(h0 + h1)
n = fd.FacetNormal(mesh)
Upwind = 0.5 * (fd.sign(fd.dot(uh, n)) + 1)


def both(u):
    return 2*fd.avg(u)


K = 0.5*fd.inner(uh, uh)
uup = 0.5 * (fd.dot(uh, n) + abs(fd.dot(uh, n)))
dT = fd.Constant(0.)
dS = fd.dS

vector_invariant = True
if vector_invariant:
    eqn = (
        fd.inner(v, u1 - u0)*dx + fd.inner(v, f*perp(uh))*dx
        + dT*fd.inner(perp(fd.grad(fd.inner(v, perp(uh)))), uh)*dx
        + dT*fd.inner(both(perp(n)*fd.inner(v, perp(uh))),
                      both(Upwind*uh))*dS
        - dT*fd.div(v)*(g*(hh - b) + K)*dx
        + phi*(h1 - h0)*dx
        - dT*fd.inner(fd.grad(phi), uh)*hh*dx
        + dT*fd.jump(phi)*(uup('+')*hh('+')
                           - uup('-')*hh('-'))*dS
        )

solver_parameters = {'mat_type': 'aij',
                     'snes_monitor': None,
                     'ksp_type': 'preonly',
                     'pc_type': 'lu',
                     'pc_factor_mat_solver_type': 'mumps'}

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=solver_parameters)

hours = 0.2
dt = 60*60*hours
dT.assign(dt)
t = 0.
hmax = 24
tmax = 60.*60.*hmax
hdump = 1
dumpt = hdump*60.*60.
tdump = 0.

x = fd.SpatialCoordinate(mesh)
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

# Topography
rl = fd.pi/9.0
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

u0, h0 = Un.split()
u0.assign(un)
h0.assign(etan + H - b)

name = "sw_imp"
file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H - b)
un.assign(u0)
file_sw.write(un, etan)

print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H - b)
        un.assign(u0)
        file_sw.write(un, etan)
        tdump -= dumpt
