import firedrake as fd
#get command arguments
from petsc4py import PETSc
from firedrake.__future__ import interpolate
from firedrake.output import VTKFile
print = PETSc.Sys.Print
PETSc.Sys.popErrorHandler()
import mg
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--gamma', type=float, default=0., help='Augmented Lagrangian scaling parameter. Default 0.')
parser.add_argument('--solver_mode', type=str, default='monolithic', help='Solver strategy. monolithic=use monolithic MG with Schwarz smoothers. schurU=eliminate down to u and do a direct solve. block=use Hdiv-style block preconditioner (requires gamma>1). splitdirect=multiplicative composition of patch and direct solve on linear SWE. Default = monolithic')
parser.add_argument('--schur_complement', type=str, default='use_u', help='Approximation of Schur complement to use. use_u=use the u-u block (requires gamma > 0). approx_sub_D=use an approximate substitution of D neglecting transport of dD by Ubar.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspmg', type=int, default=3, help='Max number of KSP iterations in the MG levels. Default 3.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--linear', action='store_true', help='Solve the linear equations.')
parser.add_argument('--one_step', action='store_true', help='Do one timestep and exit (overriding dmax).')
parser.add_argument('--time_scheme', type=int, default=0, help='Timestepping scheme. 0=Crank-Nicholson (default). 1=Implicit midpoint rule.')

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
nonlinear = not args.linear

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

if args.solver_mode == "monolithic":
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
else:
    mesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=args.ref_level, degree=deg,
                                    distribution_parameters = distribution_parameters)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)
Vnormals = fd.VectorFunctionSpace(mesh, "DG", deg)
outward_normals = fd.Function(Vnormals).interpolate(outward_normals)

def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)
gamma0 = args.gamma
gamma = fd.Constant(gamma0)

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
    if nonlinear:
        return (fd.inner(v, f*perp(u))*dx
                - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*dx
                + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                           both(Upwind*u))*dS
                - fd.div(v)*(g*(h + b) + K)*dx)
    else:
        return (fd.inner(v, f*perp(u))*dx
                - fd.div(v)*g*(h + b)*dx)

def h_op(phi, u, h):
    if nonlinear:
        uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
        return (- fd.inner(fd.grad(phi), u)*h*dx
                + fd.jump(phi)*(uup('+')*h('+')
                                - uup('-')*h('-'))*dS
                )
    else:
        return H*phi*fd.div(u)*dx


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
    eqn = testeqn \
        + gamma*(fd.div(v)*(h1 - h0)*dx
                 + dT*h_op(fd.div(v), uh, hh))

    
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
    eqn = testeqn \
        + gamma*(fd.div(v)*(h1 - h0)*dx
                 + half*dT*h_op(fd.div(v), u0, h0)
                 + half*dT*h_op(fd.div(v), u1, h1))
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

# linear shallow water operator
class HelmholtzPC(fd.AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        u, p = fd.split(trial)
        v, q = fd.split(test)
        inner = fd.inner; div = fd.div
        a = (
            fd.inner(u, v) + fd.inner(p, q) + 0.5*dT*inner(v, f*perp(u))
            - 0.5*g*dT*div(v)*p
            + 0.5*H*dT*div(u)*q
        )*fd.dx
        #Returning None as bcs
        return (a, None)

# approximate Schur complement
class ApproxUSchurPC(fd.AuxiliaryOperatorPC):
    def form(self, pc, vf, uf):
        # only hand coded for CN
        assert(args.time_scheme == 0)

        u1, h1 = fd.split(Unp1)
        # The original form for u equation
        #(fd.inner(v, f*perp(u))*dx
        # - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*dx
        # + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
        #            both(Upwind*u))*dS
        # - fd.div(v)*(g*(h + b) + K)*dx)

        #Upwind is a switch so we don't differentiate it
        Upwind = 0.5 * (fd.sign(fd.dot(u1, n)) + 1)

        Jf = fd.inner(vf, f*perp(uf))*dx
        Jf += - fd.inner(perp(fd.grad(fd.inner(vf, perp(uf)))), u1)*dx
        Jf += - fd.inner(perp(fd.grad(fd.inner(vf, perp(u1)))), uf)*dx
        Jf += fd.inner(both(perp(n)*fd.inner(vf, perp(uf))),
                              both(Upwind*u1))*dS
        Jf += fd.inner(both(perp(n)*fd.inner(vf, perp(u1))),
                              both(Upwind*uf))*dS
        Jf += - fd.div(vf)*fd.inner(u1, uf)*dx

        # we have not added the pressure gradient yet, this comes next

        # differentiated abs operator
        # (this seemed to cause problems so not using currently.)
        uupf = 0.5 * (fd.dot(uf, n) + fd.dot(uf, n))*fd.sign(fd.dot(u1, n))
        # the original form for h equation
        #(- fd.inner(fd.grad(phi), u)*h*dx
        #        + fd.jump(phi)*(uup('+')*h('+')
        #                        - uup('-')*h('-'))*dS
        #    )
        # the elimination neglects terms with delta h
        # and any surface terms in double integration by parts
        hbit = -fd.div(vf)*fd.div(uf*h1)*dx
        Jf -= 0.5*dT*g*hbit
        Jm = fd.inner(vf, uf)*dx
        J = Jm + 0.5*dT*Jf
        #Returning None as bcs
        return (J, None)

if args.solver_mode == 'schurU':

    sparameters = {
        'snes_monitor': None,
        #"snes_lag_jacobian": 2,
        "ksp_type": "gmres",
        "ksp_atol": 1.0e-50,
        "ksp_rtol": 1.0e-6,
        "ksp_converged_reason": None,
        'ksp_monitor': None,
        #'ksp_view': None,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_0_fields": "1",
        "pc_fieldsplit_1_fields": "0",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
    }

    LU = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }

    ILU = {
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
    }

    approxSchur = {
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": f"{__name__}.ApproxUSchurPC",
        "aux_pc_type": "lu",
        #"aux_pc_factor_mat_solver_type": "mumps"
    }

    sparameters["fieldsplit_0"] = LU
    if args.schur_complement == 'approx_sub_D':
        sparameters["fieldsplit_1"] = approxSchur
    else:
        sparameters["fieldsplit_1"] = LU

elif args.solver_mode == 'monolithic':
    # monolithic solver options

    if nonlinear:
        snes = "newtonls"
    else:
        snes = "ksponly"
    
    sparameters = {
        "snes_type": snes,
        "snes_monitor": None,
        #"mat_type": "matfree",
        "ksp_type": "gmres",
        #"ksp_monitor_true_residual": None,
        "ksp_converged_reason": None,
        "snes_stol": 1e-50,
        "snes_atol": 1e-50,
        "snes_rtol": 1e-8,
        "ksp_atol": 1e-50,
        "ksp_rtol": 1e-10,
        "ksp_max_it": 40,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.95,
        "mg_levels_ksp_max_it": 1,
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

elif args.solver_mode == 'block':
    # block diagonal solver options

    fieldsplit0 = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        #"pc_factor_mat_solver_type": "mumps",
    }
    fieldsplit1 = {
        "ksp_type": "preonly",
        #"pc_type": "lu",
        #"pc_type": "jacobi",
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_construct_dim": 2
    }
    
    sparameters = {
        "snes_monitor": None,
        "snes_lag_jacobian": -2,
        "snes_lag_jacobian_persists": "true",
        "ksp_type": "gmres",
        #"ksp_monitor": None,
        "ksp_converged_reason": None,
        #"ksp_view": None,
        "ksp_atol": 1e-50,
        #"ksp_ew": None,
        #"ksp_ew_version": 1,
        #"ksp_ew_threshold": 1e-10,
        #"ksp_ew_rtol0": 1e-3,
        "ksp_rtol": 1e-12,
        "ksp_max_it": 400,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_off_diag_use_amat": None,
        "fieldsplit_0": fieldsplit0,
        "fieldsplit_1": fieldsplit1
    }
elif args.solver_mode == 'splitdirect':

    patch = {
        "pc_python_type": "firedrake.PatchPC",
        "patch_pc_patch_save_operators": True,
        "patch_pc_patch_partition_of_unity": True,
        "patch_pc_patch_sub_mat_type": "seqdense",
        "patch_pc_patch_construct_dim": 0,
        "patch_pc_patch_construct_type": "star",
        "patch_pc_patch_local_type": "additive",
        "patch_pc_patch_precompute_element_tensors": True,
        "patch_pc_patch_symmetrise_sweep": False,
        "patch_sub_ksp_type": "preonly",
        "patch_sub_pc_type": "lu",
        "patch_sub_pc_factor_shift_type": "nonzero"
    }

    helmholtz = {
        "pc_python_type": f"{__name__}.HelmholtzPC",
        "aux_pc_type": "lu",
        "aux_pc_factor_mat_solver_type": "mumps"
    }

    sparameters = {
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e5,
        # "snes_max_it": 1,
        # "snes_convergence_test": "skip",
        #"snes_lag_jacobian": -2,
        #"snes_lag_jacobian_persists": None,
        "ksp_monitor": None,
        "ksp_converged_rate": None,
        #"ksp_view": None,
        "ksp_type": "gmres",
        "ksp_atol": 1e-50,
        "ksp_rtol": 1e-6,
        "ksp_max_it": 30,
        "pc_type": "python",
        "pc_type": "composite",
        "pc_composite_type": "multiplicative",
        "pc_composite_pcs": "ksp,ksp",
        "sub_0": {
            "ksp_ksp_type": "richardson",
            "ksp_ksp_richardson_scale": 1,
            "ksp_ksp_rtol": 1e-1,
            "ksp_ksp_max_it": 1,
            "ksp_ksp_convergence_test": 'skip',
            "ksp_ksp_converged_maxits": None,
            # "ksp_ksp_converged_rate": None,
            "ksp_pc_type": "python",
            "ksp": helmholtz,
        },
        "sub_1": {
            "ksp_ksp_type": "richardson",
            "ksp_ksp_richardson_scale": 0.95,
            "ksp_ksp_rtol": 1e-1,
            "ksp_ksp_max_it": 2,
            "ksp_ksp_convergence_test": 'skip',
            "ksp_ksp_converged_maxits": None,
            # "ksp_ksp_converged_rate": None,
            "ksp_pc_type": "python",
            "ksp": patch,
        }
    }

elif args.solver_mode == 'lswe':

    sparameters = {
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e5,
        # "snes_max_it": 1,
        # "snes_convergence_test": "skip",
        #"snes_lag_jacobian": -2,
        #"snes_lag_jacobian_persists": None,
        "ksp_monitor": None,
        "ksp_converged_rate": None,
        # "ksp_view": None,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 30,
        "pc_type": "python",
        "pc_python_type": f"{__name__}.HelmholtzPC",
        "aux_pc_type": "lu",
        "aux_pc_factor_mat_solver_type": "mumps"
    }

elif args.solver_mode == 'patch':

    sparameters = {
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_atol": 1e-50,
        "snes_stol": 1e-50,
        # "snes_max_it": 1,
        # "snes_convergence_test": "skip",
        #"snes_lag_jacobian": -2,
        #"snes_lag_jacobian_persists": None,
        "ksp_monitor": None,
        "ksp_converged_rate": None,
        # "ksp_view": None,
        "ksp_type": "gmres",
        "ksp_rtol": 1e-3,
        "ksp_max_it": 30,
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        "patch_pc_patch_save_operators": True,
        "patch_pc_patch_partition_of_unity": True,
        "patch_pc_patch_sub_mat_type": "seqdense",
        "patch_pc_patch_construct_dim": 0,
        "patch_pc_patch_construct_type": "star",
        "patch_pc_patch_local_type": "additive",
        "patch_pc_patch_precompute_element_tensors": True,
        "patch_pc_patch_symmetrise_sweep": False,
        "patch_sub_ksp_type": "preonly",
        "patch_sub_pc_type": "lu",
        "patch_sub_pc_factor_shift_type": "nonzero"
    }
    
dt = 60*60*args.dt
dT.assign(dt)
t = 0.


if args.solver_mode == "block":
    u, eta = fd.TrialFunctions(W)
    v, phi = fd.TestFunctions(W)
    div = fd.div; dx = fd.dx; inner = fd.inner

    use_riesz = False
    if use_riesz:
        half = fd.Constant(0.5)
        aP = (
            inner(u, v) + dT**2*half**2*g*H*div(v)*div(u)
            + eta*phi
        )*dx
    else:
        aP = fd.derivative(eqn, Unp1)
        aP += (
            dT**2*g*H*div(v)*div(u)*fd.Constant(1/4.)
        )*dx

    nprob = fd.NonlinearVariationalProblem(testeqn, Unp1, Jp=aP)
    nsolver = fd.NonlinearVariationalSolver(nprob,
                                            solver_parameters=sparameters)
else:
    nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
    ctx = {"mu": gamma*2/g/dt}
    nsolver = fd.NonlinearVariationalSolver(nprob, options_prefix="swe",
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
lambda_x = fd.atan2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.min_value(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

u0, h0 = Un.subfunctions
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

file_sw = VTKFile(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)
Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTimestep {stepcount} at time {t}\n")
    t += dt
    tdump += dt

    with PETSc.Log.Event("time solver"):
        nsolver.solve()
    Un.assign(Unp1)

    if args.one_step:
        t = tmax + dt

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount,
                "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)
