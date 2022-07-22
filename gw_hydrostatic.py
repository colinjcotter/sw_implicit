import firedrake as fd
from petsc4py import PETSc
from slice_utils import hydrostatic_rho, pi_formula,\
    slice_imr_form, both, maximum
import numpy as np

dT = fd.Constant(1)

nlayers = 10  # horizontal layers
base_columns = 300  # number of columns
L = 6000e3
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

m = fd.PeriodicRectangleMesh(base_columns, ny=1, Lx=L, Ly=1.0e-3*L,
                             direction="both",
                             quadrilateral=True,
                             distribution_parameters=distribution_parameters)
m.coordinates.dat.data[:,0] -= L/2
m.coordinates.dat.data[:,1] -= 1.0e-3*L/2

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

# build volume mesh
H = 10e3  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = fd.FacetNormal(mesh)

name = "gw_hydrostatic"
    
horizontal_degree = 1
vertical_degree = 1

S1 = fd.FiniteElement("RTCF", fd.quadrilateral, horizontal_degree+1)
S2 = fd.FiniteElement("DQ", fd.quadrilateral, horizontal_degree)

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

x, y, z = fd.SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(N**2*z/g)

cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
Up = fd.as_vector([fd.Constant(0.0),
                   fd.Constant(0.0),
                   fd.Constant(1.0)]) # up direction

un, rhon, thetan = Un.split()
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)

Pi = fd.Function(V2)

hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary=fd.Constant(1.0),
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi)

rho_back = fd.Function(V2).assign(rhon)

lines_parameters = {
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
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm",
    "assembled_pc_star_sub_sub_pc_factor_reuse_ordering": None,
    "assembled_pc_star_sub_sub_pc_factor_reuse_fill": None,
    "assembled_pc_star_sub_sub_pc_factor_fill": 1.2,
}

a = fd.Constant(100.0e3)
deltaTheta = fd.Constant(1.0e-2)
theta_pert = deltaTheta*fd.sin(np.pi*z/H)/(1 + x**2/a**2)
thetan.interpolate(thetan + theta_pert)
un.project(fd.as_vector([fd.Constant(20.0),
                         fd.Constant(0.0),
                         fd.Constant(0.0)]))

#The timestepping solver
un, rhon, thetan = fd.split(Un)
unp1, rhonp1, thetanp1 = fd.split(Unp1)

du, drho, dtheta = fd.TestFunctions(W)
f = fd.Constant(1.0e-4)
eqn = slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                     du, drho, dtheta,
                     dT=dT, n=n, Up=Up, c_pen=fd.Constant(2.0**(-7./2)),
                     cp=cp, g=g, R_d=R_d, p_0=p_0, kappa=kappa, mu=None,
                     f = f,
                     F = fd.as_vector([fd.Constant(0.),
                                       -f*fd.Constant(20.0),
                                       fd.Constant(0.)]))

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=
                                        lines_parameters,
                                        options_prefix="nsolver")
    
file_gw = fd.File(name+'.pvd')
un, rhon, thetan = Un.split()
delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)

dt = 100
dT.assign(dt)

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
tdump = 0.
tmax = 60000.
dumpt = tmax/60

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
