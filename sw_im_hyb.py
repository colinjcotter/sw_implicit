import firedrake as fd
#get command arguments
from petsc4py import PETSc

import mg
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5lu')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--time_scheme', type=int, default=0, help='Timestepping scheme. 0=Crank-Nicholson. 1=Implicit midpoint rule.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
nrefs = args.ref_level
name = args.filename
deg = args.coords_degree

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
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V1_el = fd.FiniteElement("BDM", fd.triangle, degree+1)
V1b_el = fd.BrokenElement(V1_el)
V1b = fd.FunctionSpace(mesh, V1b_el)
V2 = fd.FunctionSpace(mesh, "DG", degree)
Tr = fd.FunctionSpace(mesh, "HDivT", degree+1)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))
Wtr = V1b * V2 * Tr

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)

# D = eta + b

One = fd.Function(V2).assign(1.0)

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Unp1)
n = fd.FacetNormal(mesh)


def both(u):
    return 2*fd.avg(u)


dT = fd.Constant(0.)
dS = fd.dS


def u_op(v, u, h):
    Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    K = 0.5*fd.inner(u, u)
    return (fd.inner(v, f*perp(u))*dx
            - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*dx
            + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                          both(Upwind*u))*dS
            - fd.div(v)*(g*(h + b) + K)*dx)


def h_op(phi, u, h):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.grad(phi), u)*h*dx
            + fd.jump(phi)*(uup('+')*h('+')
                            - uup('-')*h('-'))*dS)


if args.time_scheme == 1:
    "implicit midpoint rule"
    uh = 0.5*(u0 + u1)
    hh = 0.5*(h0 + h1)

    testeqn = (
        fd.inner(v, u1 - u0)*dx
        + dT*u_op(v, uh, hh)
        + phi*(h1 - h0)*dx
        + dT*h_op(phi, uh, hh))
    # the extra bit
    eqn = testeqn
    
elif args.time_scheme == 0:
    "Crank-Nicholson rule"
    half = fd.Constant(0.5)

    testeqn = (
        fd.inner(v, u1 - u0)*dx
        + half*dT*u_op(v, u0, h0)
        + half*dT*u_op(v, u1, h1)
        + phi*(h1 - h0)*dx
        + half*dT*h_op(phi, u0, h0)
        + half*dT*h_op(phi, u1, h1))
    # the extra bit
    eqn = testeqn
else:
    raise NotImplementedError
    
# U_t + N(U) = 0
# IMPLICIT MIDPOINT
# U^{n+1} - U^n + dt*N( (U^{n+1}+U^n)/2 ) = 0.

# TRAPEZOIDAL RULE
# U^{n+1} - U^n + dt*( N(U^{n+1}) + N(U^n) )/2 = 0.
    
# Newton's method
# f(x) = 0, f:R^M -> R^M
# [Df(x)]_{i,j} = df_i/dx_j
# x^0, x^1, ...
# Df(x^k).xp = -f(x^k)
# x^{k+1} = x^k + xp.

# solver options
    
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

PETSc.Sys.popErrorHandler()

# PC forming approximate hybridisable system (without advection)
# solve it using hybridisation and then return the DG part
# (for use in a Schur compement setup)
class ApproxHybridPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        
        #input and output functions
        self.xfstar = fd.Cofunction(V2.dual())

        self.yf = fd.Function(V2) # the preconditioned residual

        # hybridised system
        u, p, ll = fd.TrialFunctions(Wtr)
        v, q, dll = fd.TestFunctions(Wtr)
        w0 = fd.Function(Wtr)
        _, self.p0, _ = fd.split(w0)

        n = fd.FacetNormal(mesh)
        eqn = (
            fd.inner(u, v) + dT*fd.inner(v, 0.5*f*perp(u))
            - 0.5*dT*g*fd.div(v)*p
            + q*p + 0.5*dT*H*fd.div(u)*q
        )*fd.dx

        # trace bits
        eqn += (
            fd.jump(v, n)*ll + fd.jump(u, n)*dll
        )*fd.dS

        condensed_params = {'ksp_type':'preonly',
                            'pc_type':'lu',
                            'pc_factor_mat_solver_type':'mumps'}

        hbps = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            'pc_sc_eliminate_fields': '0, 1',
            'condensed_field':condensed_params
        }

        prob = fd.LinearVariationalProblem(eqn, self.xfstar, w0)
        self.solver = fd.LinearVariationalSolver(
            prob, solver_parameters=hbps)
        
    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def apply(self, pc, x, y):
        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
        
        self.solver.solve()
        self.yf.assign(self.p0)

        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)

sparameters = {
    "ksp_view": None,
    "ksp_max_it": 3,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": __name__+ ".ApproxHybridPC",
}

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters)

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

x = fd.SpatialCoordinate(mesh)
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.min_value(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

u0, h0 = Un.split()
u0.assign(un)
h0.assign(etan + H - b)

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)
Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)
    
    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount,
                "dt", dt, "tlblock", args.tlblock, "ref_level", args.ref_level, "dmax", args.dmax)
