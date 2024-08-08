import firedrake as fd
#get command arguments
from petsc4py import PETSc
from firedrake.output import VTKFile

PETSc.Sys.popErrorHandler()
import mg
import argparse
parser = argparse.ArgumentParser(description='Moist Williamson 5 testcase using VI formulation.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1')
parser.add_argument('--filename', type=str, default='w5moistvi')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspmg', type=int, default=3, help='Max number of KSP iterations in the MG levels. Default 3.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--bounds', action='store_true', help='Apply the bounds constraints.')

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
        new_coords = fd.Function(X).interpolate(m.coordinates)
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
Vn = fd.VectorFunctionSpace(mesh, "CG", deg)
outward_normals = fd.Function(Vn).interpolate(fd.CellNormal(mesh))

def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2, V2,
                           V2, V2,
                           V2, V2))
# velocity, depth, total temperature, vapour, evaporation rate, cloud condensation rate, cloud
# NO RAIN AT THE MOMENT

du, dD, dbuoy, dqv, dRe, dRc, dqc = fd.TestFunctions(W)

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

u0, D0, buoy0, qv0, Re0, qc0, Rc0 = fd.split(Un)
u1, D1, buoy1, qv1, Re1, qc1, Rc1 = fd.split(Unp1)
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
        + fd.jump(buoy*v, n)*fd.avg(D+b)*dS
        - fd.div(D*v)*buoy/2*dx
        + fd.jump(D*v, n)*fd.avg(buoy/2)*dS
    )

def h_op(phi, u, h):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.grad(phi), u)*h*dx
            + fd.jump(phi)*(uup('+')*h('+')
                            - uup('-')*h('-'))*dS)

def q_op(phi, u, q):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.div(phi*u), q)*dx
            + fd.jump(phi)*(uup('+')*q('+')
                            - uup('-')*q('-'))*dS)

q0 = fd.Constant(135)
L = fd.Constant(10)
gamma_r = fd.Constant(1.0e-3)
q_precip = fd.Constant(1.0e-4)

dx0 = dx('everywhere', metadata = {'quadrature_degree': 4,
                                   'representation': 'quadrature'})
#dx0 = dx
#dS = dS('everywhere', metadata = {'quadrature_degree': 4,
#                                  'representation': 'quadrature'})

def qsat(D, buoy):
    return q0/g/(D + b)*fd.exp(20*(1-buoy/g))

#u0, D0, buoy0, qv0, Re0, qc0, Rc0 = fd.split(Un)
#u1, D1, buoy1, qv1, Re1, qc1, Rc1 = fd.split(Unp1)
#du, dD, dbuoy, dqv, dRe, dRc, dqc = fd.TestFunctions(W)

# b = g(1-theta)
# theta_T = theta + Lq_v
# theta = theta_T - Lq_v
# b = g(1-theta_T) + gLq_v
buoyT0 = buoy0 + g*L*qv0
buoyT1 = buoy1 + g*L*qv1

alpha = fd.Constant(0.5) # offcentering parameter (1.0 = BE)

eqn = (
    fd.inner(du, u1 - u0)*dx
    + (1-alpha)*dT*u_op(du, u0, D0, buoy0)
    + alpha*dT*u_op(du, u1, D1, buoy1)

    + dD*(D1 - D0)*dx
    + (1-alpha)*dT*h_op(dD, u0, D0)
    + alpha*dT*h_op(dD, u1, D1)

    + dbuoy*(buoyT1 - buoyT0)*dx
    + (1-alpha)*dT*q_op(dbuoy, u0, buoyT0)
    + alpha*dT*q_op(dbuoy, u1, buoyT1)

    + dRe*(qsat(D1, buoy1) - qv1)*fd.dx

    + dqv*(qv1 - qv0)*dx
    + (1-alpha)*dT*q_op(dqv, u0, qv0)
    + alpha*dT*q_op(dqv, u1, qv1)
    + dT*Re1*fd.dx  #  evaporation rate

    + dRc*

    + dqc*(qcv1 - qcv0)*dx
    + (1-alpha)*dT*q_op(dqv, u0, qcv0)
    + alpha*dT*q_op(dqv, u1, qcv1)
    + dT*Rc1*fd.dx
)

# monolithic solver options

sparameters = {
    "snes_monitor": None,
    "snes_vi_monitor": None,
    #"mat_type": "matfree",
    "ksp_type": "gmres",
    #"ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    #"ksp_view": None,
    "snes_stol": 1e-50,
    "snes_atol": 1e-50,
    "snes_rtol": 1e-10,
    "ksp_atol": 1e-50,
    "ksp_rtol": 1e-6,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "richardson",
    "mg_levels_ksp_richardson_scale": 0.95,
    "mg_levels_ksp_max_it": 1,
    "mg_levels_ksp_convergence_test": "skip",
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
    "mg_levels_patch_sub_pc_type": "ilu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}

sparameters = {
    "snes_monitor": None,
    "snes_vi_monitor": None,
    "ksp_type": "gmres",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}


lbound = fd.Function(W).assign(PETSc.NINFINITY)
ubound = fd.Function(W).assign(PETSc.INFINITY)

# 0   1   2      3     4    5      6
# u0, D0, buoy0, qsat0, qvp0, qc0, qr0 = fd.split(Un)

if args.bounds:
    sparameters["snes_type"] = "vinewtonrsls"
    ubound.sub(4).assign(0.) #  qprime <= 0
    ubound.sub(5).assign(q_precip) #  qc <= q_precip

dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        options_prefix="stepper",
                                        solver_parameters=sparameters)
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
Dn = fd.Function(V2, name="Depth")

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan2(x[1]/R0, x[0]/R0) #  longitude
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0) #  latitude
phi_c = fd.pi/6.0
minarg = fd.min_value(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

u0, D0, buoy0, qsat0, qvp0, qc0, qr0n = Un.subfunctions
u0.assign(un)
D0.assign(etan + H - b)

eps = fd.Constant(1.0/300)
EQ = 30*eps
SP = -40*eps
NP = -20*eps
mu1 = fd.Constant(0.05)
mu2 = fd.Constant(0.98)

# The below is from Nell Hartney
# expression for initial buoyancy - note the bracket around 1-mu
problem = "W5"
if problem == "W5":
    F = (2/(fd.pi**2))*(phi_x*(phi_x-fd.pi/2)*SP -
                        2*(phi_x+fd.pi/2)*(phi_x-fd.pi/2)*(1-mu1)*EQ
                        + phi_x*(phi_x+fd.pi/2)*NP)
    theta_expr = F + mu1*EQ*fd.cos(phi_x)*fd.sin(lambda_x)
elif problem == "W2":
    # The below is from also Nell Hartney
    omega = Omega*R0*u_max + u_max**2/2
    sigma = omega/fd.Constant(10)
    import numpy as np
    Phi0 = fd.Constant(3*np.exp(4))
    theta0 = fd.Constant(1.0) #  eps*Phi0**2
    brk0 = fd.Constant(1.0) # (omega + sigma)*fd.cos(phi_x)**2 + 2*(Phi0 - omega - sigma)
    num0 = theta0 + sigma# *fd.cos(phi_x)**2*brk0
    #den0 = (Phi0**2 + (omega + sigma)**2*fd.sin(phi_x)**4
    #        - 2*Phi0*(omega + sigma)*fd.sin(phi_x)**2)
    den0 = fd.Constant(1.0)
    theta_expr = num0/den0
else:
    raise NotImplementedError

buoyexpr = g * (1 - theta_expr)
# b = g(1- theta)

buoy0.interpolate(buoyexpr)

# The below is from Nell Hartney
# expression for initial water vapour depends on initial saturation
initial_msat = q0/(g*D0 + g*bexpr) * fd.exp(20*theta_expr)
vexpr = mu2 * initial_msat
qsat0.project(qsat(D0, buoy0))
qvp0.project(vexpr - qsat0)
# cloud and rain initially zero

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = VTKFile(name+'.pvd')
etan.assign(D0 - H + b)
un.assign(u0)
qsolver.solve()
buoyn = fd.Function(V2, name="Buoyancy")
qvn = fd.Function(V2, name="Water Vapour")
qcn = fd.Function(V2, name="Cloud Vapour")
qrn = fd.Function(V2, name="Rain")
Dn.interpolate(D0)
qvn.project(qvp0 + qsat(D0, buoy0))
qcn.interpolate(qc0)
qrn.interpolate(qr0)
buoyn.interpolate(buoy0)
file_sw.write(un, etan, buoyn, qn, qvn, qcn, qrn)
Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
stepcount = 0

One = fd.Function(V2).assign(1.0)
Area = fd.assemble(One*fd.dx)

while t < tmax + 0.5*dt:
    PETSc.Sys.Print(t)
    PETSc.Sys.Print("rain", fd.norm(qr0)/Area, "cloud", fd.norm(qc0)/Area)
    t += dt
    tdump += dt

    with PETSc.Log.Event("nsolver"):
        if args.bounds:
            nsolver.solve(bounds=(lbound, ubound))
        else:
            nsolver.solve()
    Un.assign(Unp1)
    
    if tdump > dumpt - dt*0.5:
        etan.assign(D0 - H + b)
        un.assign(u0)
        qsolver.solve()
        buoyn.interpolate(buoy0)
        qvn.project(qvp0 + qsat0)
        qcn.interpolate(qc0)
        qrn.interpolate(qr0)
        file_sw.write(un, etan, buoyn, qn, qvn, qcn, qrn)
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount,
                "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)