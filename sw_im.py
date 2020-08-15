import firedrake as fd

hours = 0.2
dt = 60*60*hours

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
ref_level = 3

mesh = fd.IcosahedralSphereMesh(radius=R0,
                                refinement_level=ref_level, degree=3)
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
uup = 0.5 * (fd.dot(uh, n) + fd.abs(fd.dot(uh, n)))
dT = fd.Constant(dt)
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

solver_parameters = {'mat_type':'aij',
                     '

GOT THIS FAR!
NEED TO BUILD NONLINEAR SOLVER

t = 0.
hmax = 24
tmax = 60.*60.*hmax
hdump = 1
dumpt = hdump*60.*60.
tdump = 0.

u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = Constant(u_0)
u_expr = as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = Function(V1, name="Velocity").project(u_expr)
etan = Function(V2, name="Elevation").project(eta_expr)

# Topography
rl = pi/9.0
lambda_x = atan_2(x[1]/R0, x[0]/R0)
lambda_c = -pi/2.0
phi_x = asin(x[2]/R0)
phi_c = pi/6.0
minarg = Min(pow(rl, 2), pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - sqrt(minarg)/rl)
b.interpolate(bexpr)

U = Function(W)
eU = Function(W)
DU = Function(W)
V = Function(W)

u0, h0 = U.split()
u0.assign(un)
h0.assign(etan + H - b)

name = "sw_imp"
file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H - b)
file_sw.write(un, hn, etan)

print ('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    #Average the nonlinearity
    cheby.apply(U, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Step forward V by half a step
    V.assign(U + 0.5*V)

    #transform forwards to U^{n+1/2}
    cheby2.apply(V, USlow_in, dt/2)

    #Average the nonlinearity
    cheby.apply(V, USlow_in, expt)
    SlowSolver.solve()
    cheby.apply(USlow_out, DU, -expt)
    DU *= wt
    ensemble.allreduce(DU, V)
    #Advance U by half a step
    cheby2.apply(U, DU, dt/2)
    V.assign(DU + V)

    #transform forwards to next timestep
    cheby2.apply(V, U, dt/2)

    if rank == 0:
        if tdump > dumpt - dt*0.5:
            un.assign(U_u)
            etan.assign(U_eta)
            file_sw.write(un, etan, b)
            tdump -= dumpt
