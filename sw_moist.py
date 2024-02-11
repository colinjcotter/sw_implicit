import firedrake as fd
#get command arguments
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import mg
import argparse
parser = argparse.ArgumentParser(description='Moist Williamson 5 testcase')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
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
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
name = args.filename
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
#distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.FACET, 2)}


def high_order_mesh_hierarchy(mh, degree, R0):
    meshes = []
    for m in mh:
        X = fd.VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = fd.interpolate(m.coordinates, X)
        x, y, z = new_coords
        r = (x**2 + y**2 + z**2)**0.5
        new_coords.interpolate(R0*new_coords/r)
        new_mesh = fd.Mesh(new_coords)
        meshes.append(new_mesh)

    return fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)

basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=base_level,
                                    degree=1,
                                    distribution_parameters = distribution_parameters)
del basemesh._radius
mh = fd.MeshHierarchy(basemesh, nrefs)
mh = high_order_mesh_hierarchy(mh, deg, R0)
for mesh in mh:
    xf = mesh.coordinates
    mesh.transfer_coordinates = fd.Function(xf)
    x = fd.SpatialCoordinate(mesh)
    r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    xf.interpolate(R0*xf/r)
    mesh.init_cell_orientations(x)
mesh = mh[-1]

R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)
outward_normals = fd.CellNormal(mesh)

def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2, V2, V2, V2, V2))
# velocity, depth, temperature, vapour, cloud, rain

du, dD, dbuoy, dqv, dqc, dqr = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)

# h = D + b

One = fd.Function(V2).assign(1.0)

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, D0, buoy0, qv0, qc0, qr0 = fd.split(Un)
u1, D1, buoy1, qv1, qc1, qr1  = fd.split(Unp1)
n = fd.FacetNormal(mesh)


def both(u):
    return 2*fd.avg(u)

dT = fd.Constant(0.)
dS = fd.dS

def u_op(v, u, D, buoy):
    Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    K = 0.5*fd.inner(u, u)
    return (
        fd.inner(v, f*perp(u))*dx
        - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*dx
        + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                   both(Upwind*u))*dS
        - fd.div(v)*K*dx
        # Thermal terms from Nell
        # b grad (D+B) + D/2 grad b
        # integrate by parts
        # -<div(bv), (D+B)> + << jump(vb, n), {D+B} >>
        # -<div(Dv), b/2> + <<jump(Dv, n), {b/2} >>
        - fd.div(buoy*v)*(D+b)*dx
        - fd.jump(buoy*v, n)*fd.avg(D+b)*dS
        + fd.div(D*v)*b/2*dx
        - fd.jump(D*v, n)*fd.avg(b/2)*dS
    )

def h_op(phi, u, h):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.grad(phi), u)*h*dx
            + fd.jump(phi)*(uup('+')*h('+')
                            - uup('-')*h('-'))*dS)

def q_op(phi, u, q):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.grad(phi*q), u)*dx
            + fd.jump(phi)*(uup('+')*q('+')
                            - uup('-')*q('-'))*dS)

"implicit midpoint rule"
uh = 0.5*(u0 + u1)
DD = 0.5*(D0 + D1)
buoyh = 0.5*(buoy0 + buoy1)
qvh = 0.5*(qv0 + qv1)
qch = 0.5*(qc0 + qc1)
qrh = 0.5*(qr0 + qr1)

# du, dD, dbuoy, dqv, dqc, dqr = fd.TestFunctions(W)
# u0, D0, buoy0, qv0, qc0, qr0 = fd.split(Un)
# u1, D1, buoy1, qv1, qc1, qr1  = fd.split(Unp1)

q0 = fd.Constant(135)
qsat = q0/g/(Dh + b)*fd.exp(20*(1-buoyh/g))
L = fd.Constant(10)
gamma_r = fd.Constant(1.0e-3)
gamma_v = 1/(1 + L*(20*q0/g/(Dh + b))*fd.exp(20*(1-buoyh/g)))
q_precip = fd.Constant(1.0e-4)

def del_qv(qv):
    return MaxValue(0, gamma_v*(qv - qsat))/dT

def del_qc(qc, qv):
    return MinValue(qc, MaxValue(0, gamma_v*(qsat - qv)))/dT

def del_qr(qc):
    return MaxValue(0, gamma_r*(qc - q_precip))/dT

eqn = (
    fd.inner(du, u1 - u0)*dx
    + dT*u_op(du, uh, DD)
    + dD*(D1 - D0)*dx
    + dT*h_op(dD, uh, DD)
    + dbuoy*(buoy1 - buoy0)*dx
    + dT*q_op(dbuoy, uh, buoyh)*dx
    - dT*g*L*dbuoy*(del_qv(qvh) - del_qc(qch, qvh))*dx #  bouyancy source
    + dqv*(qv1 - qv0)*dx
    + dT*q_op(dqv, uh, qvh)*dx
    - dT*dqv*(del_qc(qch, qvh) - del_qv(qvh))*dx #  qv source
    + dqc*(qc1 - qc0)*dx
    + dT*q_op(dqc, uh, qch)*dx
    - dT*dqc*(del_qv(qvh) - del_qc(qch, qvh) - del_qr(qch))*dx #  qc source
    + dqr*(qr1 - qr0)*dx
    + dT*q_op(dqr, uh, qrh)*dx
    - dT*dqr*del_qr(qch)*dx #  qr source
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

sparameters = {
    "snes_monitor": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    #"mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_local_type": "additive",
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}
    
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
ctx = {"mu": gamma*2/g/dt}
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters,
                                        appctx=ctx)
vtransfer = mg.ManifoldTransfer()
tm = fd.TransferManager()
transfers = {
    V1.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
                       vtransfer.inject),
    V2.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
                       vtransfer.inject)
}
transfermanager = fd.TransferManager(native_transfers=transfers)
nsolver.set_transfer_manager(transfermanager)

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
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0) #  longitude
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0) #  latitude
phi_c = fd.pi/6.0
minarg = fd.min_value(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

u0, D0, bouy0, qv0, qc0, qr0 = U0.subfunctions
u0.assign(un)
D0.assign(etan + H - b)

# The below is from Nell Hartney
# expression for initial buoyancy - note the bracket around 1-mu
F = (2/(pi**2))*(phi_x*(phi_x-pi/2)*SP -
                 2*(phi_x+pi/2)*(phi_x-pi/2)*(1-mu1)*EQ
                 + phi_x*(phi_x+pi/2)*NP)
theta_expr = F + mu1*EQ*cos(phi_x)*sin(lambda_x)
buoyexpr = g * (1 - theta_expr)
bouy0.interpolate(bouyexpr)

# The below is from Nell Hartney
# expression for initial water vapour depends on initial saturation
initial_msat = q0/(g*D0 + g*tpexpr) * exp(20*theta_expr)
vexpr = mu2 * initial_msat
qv.interpolate(vexpr)
# cloud and rain initially zero

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

eps = Constant(1.0/300)
theta_EQ = 30*eps
theta_SP = -40*eps
theta_NP = -20*eps
mu1 = fd.Constant(0.05)
mu2 = fd.Constant(0.98)

def ma_F(f1, f2, f3, phi):
    return phi*(phi - fd.pi/2)*f1 - \
        2*(phi + fd.pi/2)*(phi - fd/pi/2)*f2 + \
        phi*(phi + fd.pi/2)*f3

Phi00 = fd.Constant(
Phi0 = 

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

    with PETSc.Lgo.Event("nsolver"):
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
