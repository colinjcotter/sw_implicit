import firedrake as fd

#get command arguments
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 24.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 1.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5semi')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--k_max', type=int, default=4, help='Nonlinear iterations. Default 4.')
args = parser.parse_known_args()
args = args[0]

if args.show_args:
    print(args)

name = args.filename
    
# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
deg = args.coords_degree
k_max = args.k_max
mesh = fd.IcosahedralSphereMesh(radius=R0,
                                refinement_level=args.ref_level, degree=deg)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDFM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
V = V1 * V2    # Mixed space for velocity and depth

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)

def both(u):
    return 2*fd.avg(u)

dT = fd.Constant(0.)
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

x = fd.SpatialCoordinate(mesh)
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

up = fd.Function(V1)
hp = fd.Function(V2)

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)
hn = fd.Function(V2).assign(etan + H - b)

##############################################################################
# Set up depth advection solver (DG upwinded scheme)
##############################################################################

hps = fd.Function(V2)
h = fd.TrialFunction(V2)
phi = fd.TestFunction(V2)
hh = 0.5 * (hn + h)
uh = 0.5 * (un + up)
n = fd.FacetNormal(mesh)
dx = fd.dx
dS = fd.dS
uup = 0.5 * (fd.dot(uh, n) + abs(fd.dot(uh, n)))
Heqn = ((h - hn)*phi*dx - dT*fd.inner(fd.grad(phi), uh*hh)*dx
        + dT*fd.jump(phi)*(uup('+')*hh('+')-uup('-')*hh('-'))*dS)
Hproblem = fd.LinearVariationalProblem(fd.lhs(Heqn), fd.rhs(Heqn), hps)
ilu_params = {'ksp_type': 'gmres',
              'pc_type': 'bjacobi',
              'sub_pc_type':'ilu',
              'ksp_converged_reason':None}
Hsolver = fd.LinearVariationalSolver(Hproblem,
                                     solver_parameters=ilu_params,
                                     options_prefix="H-advection")

##############################################################################
# Velocity advection (Natale et. al (2016) extended to SWE)
##############################################################################

ups = fd.Function(V1)
u = fd.TrialFunction(V1)
v = fd.TestFunction(V1)
hh = 0.5 * (hn + hp)
ubar = 0.5 * (un + up)
uup = 0.5 * (fd.dot(ubar, n) + abs(fd.dot(ubar, n)))
uh = 0.5 * (un + u)
Upwind = 0.5 * (fd.sign(fd.dot(ubar, n)) + 1)
K = 0.5 * (fd.inner(0.5 * (un + up), 0.5 * (un + up)))
both = lambda u: 2*fd.avg(u)
outward_normals = fd.CellNormal(mesh)
perp = lambda arg: fd.cross(outward_normals, arg)
Ueqn = (fd.inner(u - un, v)*dx + dT*fd.inner(perp(uh)*f, v)*dx
        - dT*fd.inner(perp(fd.grad(fd.inner(v, perp(ubar)))), uh)*dx
        + dT*fd.inner(both(perp(n)*fd.inner(v, perp(ubar))),
                   both(Upwind*uh))*dS
        - dT*fd.div(v)*(g*(hh + b) + K)*dx)
Uproblem = fd.LinearVariationalProblem(fd.lhs(Ueqn), fd.rhs(Ueqn), ups)
Usolver = fd.LinearVariationalSolver(Uproblem,
                                  solver_parameters=ilu_params,
                                  options_prefix="U-advection")

##############################################################################
# Linear solver for incremental updates
##############################################################################

HU = fd.Function(V)
deltaU, deltaH = HU.split()
w, phi = fd.TestFunctions(V)
du, dh = fd.TrialFunctions(V)
alpha = fd.Constant(0.5)
HUlhs = (fd.inner(w, du + alpha*dT*f*perp(du))*dx
         - alpha*dT*fd.div(w)*g*dh*dx
         + phi*(dh + alpha*dT*H*fd.div(du))*dx)
HUrhs = -fd.inner(w, up - ups)*dx - phi*(hp - hps)*dx
HUproblem = fd.LinearVariationalProblem(HUlhs, HUrhs, HU)
params = {'mat_type': 'matfree',
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.HybridizationPC',
          'hybridization': {'ksp_type': 'cg',
                            'pc_type': 'gamg',
                            'ksp_converged_reason': None
          }}

HUsolver = fd.LinearVariationalSolver(HUproblem,
                                      solver_parameters=params,
                                      options_prefix="impl-solve")

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File(name+'.pvd')
etan.assign(hn - H + b)
qsolver.solve()
file_sw.write(un, etan, qn)

print('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    up.assign(un)
    hp.assign(hn)

    # Start picard cycle
    for i in range(k_max):
        # Advect to get candidates
        Hsolver.solve()
        Usolver.solve()

        # Linear solve for updates
        HUsolver.solve()

        # Increment updates
        up += deltaU
        hp += deltaH

    # Update fields for next time step
    un.assign(up)
    hn.assign(hp)

    if tdump > dumpt - dt*0.5:
        etan.assign(hn - H + b)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
