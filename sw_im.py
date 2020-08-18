import firedrake as fd

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = 1
ref_level = 5 - base_level
deg = 1
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 1)}
basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=base_level, degree=deg,
                                    distribution_parameters = distribution_parameters)
mh = fd.MeshHierarchy(basemesh, ref_level)
for mesh in mh:
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
mesh = mh[-1]
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


V1 = fd.FunctionSpace(mesh, "BDM", 2)
V2 = fd.FunctionSpace(mesh, "DG", 1)
V0 = fd.FunctionSpace(mesh, "CG", 3)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)
gamma0 = 10.
gamma = fd.Constant(gamma0)

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
        fd.inner(v, u1 - u0)*dx + dT*fd.inner(v, f*perp(uh))*dx
        - dT*fd.inner(perp(fd.grad(fd.inner(v, perp(uh)))), uh)*dx
        + dT*fd.inner(both(perp(n)*fd.inner(v, perp(uh))),
                      both(Upwind*uh))*dS
        - dT*fd.div(v)*(g*(hh + b) + K)*dx
        + phi*(h1 - h0)*dx
        - dT*fd.inner(fd.grad(phi), uh)*hh*dx
        + dT*fd.jump(phi)*(uup('+')*hh('+')
                           - uup('-')*hh('-'))*dS
        # the extra bit
        + gamma*(fd.div(v)*(h1 - h0)*dx
                 - dT*fd.inner(fd.grad(fd.div(v)), uh)*hh*dx
                 + dT*fd.jump(fd.div(v))*(uup('+')*hh('+')
                                          - uup('-')*hh('-'))*dS)
        )

lu_parameters = {'mat_type': 'aij',
                 'snes_monitor': None,
                 'ksp_type': 'preonly',
                 'pc_type': 'lu',
                 'pc_factor_mat_solver_type': 'mumps'}


sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "gmres",
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",
    "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
    "fieldsplit_1_Mp_pc_type": "bjacobi",
    "fieldsplit_1_Mp_sub_pc_type": "ilu"
}

mgparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    'ksp_monitor': None,
    "ksp_rtol": 1e-5,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "mg",
    "fieldsplit_0_mg_coarse_ksp_type": "preonly",
    "fieldsplit_0_mg_coarse_pc_type": "python",
    "fieldsplit_0_mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_mg_coarse_assembled_pc_type": "lu",
    "fieldsplit_0_mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_0_mg_levels_ksp_type": "gmres",
    "fieldsplit_0_mg_levels_ksp_max_it": 3,
    "fieldsplit_0_mg_levels_pc_type": "python",
    "fieldsplit_0_mg_levels_pc_python_type": "firedrake.PatchPC",
    "fieldsplit_0_mg_levels_patch_pc_patch_save_operators": True,
    "fieldsplit_0_mg_levels_patch_pc_patch_partition_of_unity": False,
    "fieldsplit_0_mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "fieldsplit_0_mg_levels_patch_pc_patch_construct_type": "vanka",
    "fieldsplit_0_mg_levels_patch_pc_patch_multiplicative": False,
    "fieldsplit_0_mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "fieldsplit_0_mg_levels_patch_pc_patch_construct_dim": 0,
    "fieldsplit_0_mg_levels_patch_sub_ksp_type": "preonly",
    "fieldsplit_0_mg_levels_patch_sub_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
    "fieldsplit_1_Mp_pc_type": "bjacobi",
    "fieldsplit_1_Mp_sub_pc_type": "ilu"
}

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
ctx = {"mu": -1/gamma}
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=mgparameters,
                                        appctx=ctx)

hours = 0.05
dt = 60*60*hours
dT.assign(dt)
t = 0.
dmax = 15
hmax = 24*dmax
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

q = TrialFunction(V0)
p = TestFunction(V0)

qn = Function(V0, name="Relative Vorticity")
veqn = q*p*dx + inner(perp(grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(lhs(veqn), rhs(vqn), qn)
qsolver = fd.linearVariationalSolver(vprob)

name = "sw_imp"
file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)
Unp1.assign(Un)

print('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solver()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
