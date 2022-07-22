import firedrake as fd
from petsc4py import PETSc
from slice_utils import hydrostatic_rho, pi_formula,\
    slice_imr_form, both, maximum, minimum
import numpy as np

dT = fd.Constant(1)

nlayers = 70  # horizontal layers
base_columns = 180  # number of columns
L = 144e3
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
m = fd.PeriodicIntervalMesh(base_columns, L, distribution_parameters =
                            distribution_parameters)
m.coordinates.dat.data[:] -= L/2

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

# build volume mesh
H = 33e3  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = fd.FacetNormal(mesh)

# making a mountain out of a molehill
a = 1000.
xc = 0.
x, z = fd.SpatialCoordinate(mesh)
hm = 1.
zs = hm*a**2/((x-xc)**2 + a**2)

smooth_z = False
name = "mount_agnesi"
if smooth_z:
    name += '_smootherz'
    zh = 5000.
    xexpr = fd.as_vector([x, fd.conditional(z < zh, z + fd.cos(0.5*np.pi*z/zh)**6*zs, z)])
else:
    xexpr = fd.as_vector([x, z + ((H-z)/H)*zs])
mesh.coordinates.interpolate(xexpr)
    
horizontal_degree = 1
vertical_degree = 1

S1 = fd.FiniteElement("CG", fd.interval, horizontal_degree+1)
S2 = fd.FiniteElement("DG", fd.interval, horizontal_degree)

# vertical base spaces
T0 = fd.FiniteElement("CG", fd.interval, vertical_degree+1)
T1 = fd.FiniteElement("DG", fd.interval, vertical_degree)

# build spaces V2, V3, Vt
V2h_elt = fd.HDiv(fd.TensorProductElement(S1, T1))
V2t_elt = fd.TensorProductElement(S2, T0)
V3_elt = fd.TensorProductElement(S2, T1)
V2v_elt = fd.HDiv(V2t_elt)
V2_elt = V2h_elt + V2v_elt

V1 = fd.FunctionSpace(mesh, V2_elt, name="Velocity")
V2 = fd.FunctionSpace(mesh, V3_elt, name="Pressure")
Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

W = V1 * V2 * Vt #velocity, density, temperature

Un = fd.Function(W)
Unp1 = fd.Function(W)

x, z = fd.SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(288.)
thetab = Tsurf*fd.exp(N**2*z/g)

cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)]) # up direction

un, rhon, thetan = Un.split()
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)

Pi = fd.Function(V2)

PETSc.Sys.Print("Solve with Pi_b = 1")
bdyC = fd.Constant(1)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=None, pi_boundary=bdyC,
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi)
bdyval = minimum(Pi)

PETSc.Sys.Print("Solve with Pi_t =", bdyval)
bdyC.assign(bdyval)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=None, pi_boundary=bdyC,
                cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                top=True, Pi=Pi)
maxP = maximum(Pi)

bdyval0 = bdyval*0.9
bdyval1 = bdyval*2
Ptarg = 1.00001
while np.abs(maxP-Ptarg) > 1.0e-7:
    bdyval = 0.5*(bdyval0 + bdyval1)
    bdyC.assign(bdyval)
    PETSc.Sys.Print("Solve with Pi_t = ", bdyval)
    hydrostatic_rho(Vv, V2, mesh, thetan, rhon=None,
                    pi_boundary=bdyC,
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=True, Pi=Pi)
    maxP = maximum(Pi)
    PETSc.Sys.Print(bdyval, bdyval0, bdyval1)
    if maxP < Ptarg:
        bdyval0 = bdyval*1.0
    else:
        bdyval1 = bdyval*1.0
    PETSc.Sys.Print(bdyval, bdyval0, bdyval1)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=bdyC,
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=True)

rho_back = fd.Function(V2).assign(rhon)

#requires a mesh hierarchy
mg_sparameters = {
    "snes_converged_reason": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.AssembledPC",
    "mg_levels_assembled_pc_type": "python",
    "mg_levels_assembled_pc_python_type": "firedrake.ASMStarPC",
    "mg_levels_assmbled_pc_star_construct_dim": 0,
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu"
}

sparameters = {
    "snes_converged_reason": None,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMVankaPC",
    "assembled_pc_vanka_construct_dim": 0,
    "assembled_pc_vanka_sub_sub_pc_factor_mat_ordering_type": "rcm"
}

un.project(fd.as_vector([10.0, 0.0]))

#The timestepping solver
un, rhon, thetan = fd.split(Un)
unp1, rhonp1, thetanp1 = fd.split(Unp1)

dt = 5
dT.assign(dt)

du, drho, dtheta = fd.TestFunctions(W)

zc = H-10000.
mubar = 0.15
mu_top = fd.conditional(z <= zc, 0.0, mubar*fd.sin((np.pi/2.)*(z-zc)/(H-zc))**2)
mu = fd.Function(V2).interpolate(mu_top/dT)

eqn = slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                     du, drho, dtheta,
                     dT=dT, n=n, Up=Up, c_pen=fd.Constant(2.0**(-7./2)),
                     cp=cp, g=g, R_d=R_d, p_0=p_0, kappa=kappa, mu=mu)

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=sparameters,
                                        options_prefix="nsolver")
    
file_gw = fd.File(name+'.pvd')
un, rhon, thetan = Un.split()
delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)

DG0 = fd.FunctionSpace(mesh, "DG", 0)
One = fd.Function(DG0).assign(1.0)
unn = 0.5*(fd.inner(-un, n) + abs(fd.inner(-un, n))) # gives fluxes *into* cell only
v = fd.TestFunction(DG0)
Courant_num = fd.Function(DG0, name="Courant numerator")
Courant_num_form = dT*(
    both(unn*v)*(fd.dS_v + fd.dS_h)
    + unn*v*fd.ds_tb
)
Courant_denom = fd.Function(DG0, name="Courant denominator")
fd.assemble(One*v*fd.dx, tensor=Courant_denom)
Courant = fd.Function(DG0, name="Courant")

fd.assemble(Courant_num_form, tensor=Courant_num)
Courant.assign(Courant_num/Courant_denom)

file_gw.write(un, rhon, thetan, delta_rho, delta_theta, Courant)
Unp1.assign(Un)

t = 0.
dumpt = 500.
tdump = 0.
tmax = 9000.

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    PETSc.Sys.Print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        delta_theta.assign(thetan-theta_back)
        delta_rho.assign(rhon-rho_back)

        fd.assemble(Courant_num_form, tensor=Courant_num)
        Courant.assign(Courant_num/Courant_denom)
        file_gw.write(un, rhon, thetan, delta_rho, delta_theta,
                      Courant)
        tdump -= dumpt
