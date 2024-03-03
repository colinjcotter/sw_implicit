import firedrake as fd
#get command arguments
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()

from utils.planets import earth
from utils import shallow_water as swe
from utils.shallow_water.williamson1992 import case5 as w5

from utils.mg import ManifoldTransferManager

import argparse
parser = argparse.ArgumentParser(description='Moist Williamson 5 testcase')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default XXX')
parser.add_argument('--filename', type=str, default='w5moist')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspmg', type=int, default=3, help='Max number of KSP iterations in the MG levels. Default 3.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = earth.Radius
H = w5.H0
base_level = args.base_level
nrefs = args.ref_level - base_level
name = args.filename

mesh = swe.create_mg_globe_mesh(base_level=args.base_level,
                                ref_level=args.ref_level,
                                coords_degree=1)

R0 = fd.Constant(R0)
x, y, z = fd.SpatialCoordinate(mesh)
outward_normals = fd.CellNormal(mesh)

def perp(u):
    return fd.cross(outward_normals, u)


Vdry = swe.default_function_space(mesh, degree=args.degree)
V1, V2 = Vdry.subfunctions
V0 = fd.FunctionSpace(mesh, "CG", args.degree+2)
W = fd.MixedFunctionSpace((V1, V2, V2, V2, V2, V2))
# velocity, depth, temperature, vapour, cloud, rain

Omega = earth.Omega  # rotation rate
f = w5.coriolis_expression(x, y, z)  # Coriolis parameter
g = earth.Gravity  # Gravitational constant
b = fd.Function(V2, name="Topography")

# h = D + b

# finite element forms

n = fd.FacetNormal(mesh)
outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


def form_advection(u, v):
    def both(u):
        return 2*fd.avg(u)
    upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    k = 0.5*fd.inner(u, u)
    return (fd.inner(v, f*perp(u))*fd.dx
            - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
            - fd.div(v)*k*fd.dx
            + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                       both(upwind*u))*fd.dS)


def form_transport(u, q, p, conservative=False):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    flux = fd.jump(p)*(uup('+')*q('+')
                       - uup('-')*q('-'))*fd.dS
    if conservative:
        volume = -fd.inner(fd.grad(p), u)*q*fd.dx
    else:
        volume = -fd.inner(fd.div(p*u), q)*fd.dx
    return volume + flux


def form_function_velocity_dry(g, b, f, u, h, v):
    A = form_advection(u, v)
    P = -fd.div(v)*g*(h + b)*fd.dx
    return A + P


def form_function_velocity_moist(g, b, f, u, h, B, v):
    A = form_advection(u, v)
    # Thermal terms from Nell
    # b grad (D+B) + D/2 grad b
    # integrate by parts
    # -<div(bv), (D+B)> + << jump(vb, n), {D+B} >>
    # -<div(Dv), b/2> + <<jump(Dv, n), {b/2} >>
    P = (- fd.div(B*v)*(h + b)*fd.dx
         + fd.jump(B*v, n)*fd.avg(h + b)*fd.dS
         - fd.div(h*v)*(B/2)*fd.dx
         + fd.jump(h*v, n)*fd.avg(B/2)*fd.dS
    )
    return A + P


def form_function_depth(u, h, p):
    return form_transport(u, h, p, conservative=True)

def u_op(v, u, h, B):
    return form_function_velocity_moist(g, b, f, u, h, B, v)

def h_op(phi, u, h):
    return form_function_depth(u, h, phi)

def q_op(phi, u, q):
    return form_transport(u, q, phi, conservative=False)

Un = fd.Function(W)
Unp1 = fd.Function(W)
u0, h0, B0, qv0, qc0, qr0 = fd.split(Un)
u1, h1, B1, qv1, qc1, qr1  = fd.split(Unp1)
du, dh, dB, dqv, dqc, dqr = fd.TestFunctions(W)

dT = fd.Constant(0.)

"implicit midpoint rule"
uh = 0.5*(u0 + u1)
hh = 0.5*(h0 + h1)
Bh = 0.5*(B0 + B1)
qvh = 0.5*(qv0 + qv1)
qch = 0.5*(qc0 + qc1)
qrh = 0.5*(qr0 + qr1)


q0 = fd.Constant(135)
qsat = q0/g/(hh + b)*fd.exp(20*(1-Bh/g))
L = fd.Constant(10)
gamma_r = fd.Constant(1.0e-3)
gamma_v = 1/(1 + L*(20*q0/g/(hh + b))*fd.exp(20*(1-Bh/g)))
q_precip = fd.Constant(1.0e-4)

dx0 = fd.dx('everywhere', metadata = {'quadrature_degree': 6,
                                      'representation': 'quadrature'})

def del_qv(qv):
    return fd.max_value(0, gamma_v*(qv - qsat))/dT

def del_qc(qc, qv):
    return fd.min_value(qc, fd.max_value(0, gamma_v*(qsat - qv)))/dT

def del_qr(qc):
    return fd.max_value(0, gamma_r*(qc - q_precip))/dT

eqn = (
    # velocity
    fd.inner(du, u1 - u0)*fd.dx
    + dT*u_op(du, uh, hh, Bh)

    # depth
    + dh*(h1 - h0)*fd.dx
    + dT*h_op(dh, uh, hh)

    # buoyancy / temperature
    + dB*(B1 - B0)*fd.dx
    + dT*q_op(dB, uh, Bh)
    + dT*g*L*dB*(del_qv(qvh) - del_qc(qch, qvh))*dx0 #  buoyancy source

    # vapour
    + dqv*(qv1 - qv0)*fd.dx
    + dT*q_op(dqv, uh, qvh)
    - dT*dqv*(del_qc(qch, qvh) - del_qv(qvh))*dx0 #  qv source

    # cloud
    + dqc*(qc1 - qc0)*fd.dx
    + dT*q_op(dqc, uh, qch)
    - dT*dqc*(del_qv(qvh) - del_qc(qch, qvh) - del_qr(qch))*dx0 #  qc source

    # rain
    + dqr*(qr1 - qr0)*fd.dx
    + dT*q_op(dqr, uh, qrh)
    - dT*dqr*del_qr(qch)*dx0 #  qr source
)
    
# U_t + N(U) = 0
# IMPLICIT MIDPOINT
# U^{n+1} - U^n + dt*N( (U^{n+1}+U^n)/2 ) = 0.

# Newton's method
# f(x) = 0, f:R^M -> R^M
# [Df(x)]_{i,j} = df_i/dx_j
# x^0, x^1, ...
# Df(x^k).xp = -f(x^k)
# x^{k+1} = x^k + xp.

# monolithic solver options

atol = 1e5
sparameters = {
    "snes": {
        "monitor": None,
        "converged_reason": None,
        'atol': atol,
        'rtol': 1e-8,
        'ksp_ew': None,
        'ksp_ew_version': 1,
    },
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp": {
        "monitor": None,
        "converged_rate": None,
        "atol": atol,
        "rtol": 1e-3,
        "max_it": 30,
    },
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg": {
        "transfer_manager": f"{__name__}.ManifoldTransferManager",
        "levels": {
            "ksp_type": "gmres",
            "ksp_max_it": 3,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch": {
                "pc_patch": {
                    "save_operators": True,
                    "partition_of_unity": True,
                    "sub_mat_type": "seqdense",
                    "construct_dim": 0,
                    "construct_type": "star",
                    "local_type": "additive",
                    "precompute_element_tensors": True,
                    "symmetrise_sweep": False,
                },
                "sub": {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_shift_type": "nonzero",
                },
            },
        },
        "coarse": {
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "lu",
            "assembled_pc_factor_mat_solver_type": "mumps",
        },
    },
}

dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters)

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

u_expr = w5.velocity_expression(x, y, z)
eta_expr = w5.elevation_expression(x, y, z)

un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

# Topography.
bexpr = w5.topography_expression(x, y, z)

b.interpolate(bexpr)

u0, h0, B0, qv0, qc0, qr0n = Un.subfunctions
u0.assign(un)
h0.assign(etan + H - b)

# The below is from Nell Hartney
eps = fd.Constant(1.0/300)
EQ = 30*eps
SP = -40*eps
NP = -20*eps
mu1 = fd.Constant(0.05)
mu2 = fd.Constant(0.98)

# expression for initial buoyancy - note the bracket around 1-mu
lambda_x = fd.atan2(y/R0, x/R0) #  longitude
phi_x = fd.asin(z/R0) #  latitude

F = (2/(fd.pi**2))*(phi_x*(phi_x-fd.pi/2)*SP -
                 2*(phi_x+fd.pi/2)*(phi_x-fd.pi/2)*(1-mu1)*EQ
                 + phi_x*(phi_x+fd.pi/2)*NP)
theta_expr = F + mu1*EQ*fd.cos(phi_x)*fd.sin(lambda_x)
buoyexpr = g * (1 - theta_expr)
B0.interpolate(buoyexpr)

# The below is from Nell Hartney
# expression for initial water vapour depends on initial saturation
initial_msat = q0/(g*h0 + g*bexpr) * fd.exp(20*theta_expr)
vexpr = mu2 * initial_msat
qv0.interpolate(vexpr)
# cloud and rain initially zero

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*fd.dx + fd.inner(perp(fd.grad(p)), un)*fd.dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
qvn = fd.Function(V2, name="Water Vapour")
qcn = fd.Function(V2, name="Cloud Vapour")
qrn = fd.Function(V2, name="Rain")
qvn.interpolate(qv0)
qcn.interpolate(qc0)
qrn.interpolate(qr0)
file_sw.write(un, etan, qn, qvn, qcn, qrn)
Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\n=== --- Timestep {stepcount} at time {t} --- ===\n")
    t += dt
    tdump += dt

    with PETSc.Log.Event("nsolver"):
        nsolver.solve()
    Un.assign(Unp1)
    
    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        qvn.interpolate(qv0)
        qcn.interpolate(qc0)
        qrn.interpolate(qr0)
        file_sw.write(un, etan, qn, qvn, qcn, qrn)
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("\n=== --- Iteration counts --- ===\n")
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount,
                "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)
