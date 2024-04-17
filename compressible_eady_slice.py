import firedrake as fd
from petsc4py import PETSc
from slice_utils import hydrostatic_rho, pi_formula,\
    slice_imr_form, both, maximum, minimum
import numpy as np

dT = fd.Constant(1)

nlayers = 20
base_columns = 60
L = 240e3
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

L = 1000000.
H = 10000.  # Height position of the model top
f = 1.e-04

m = fd.PeriodicRectangleMesh(base_columns, ny=1, Lx=2*L, Ly=1.0e-3*L,
                             direction="both",
                             quadrilateral=True,
                             distribution_parameters=distribution_parameters)

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)

# build volume mesh
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = fd.FacetNormal(mesh)

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
# Isothermal T = theta*pi is constant
# pi = T/theta => pi_z = -T/theta^2 theta_z
# so -g = cp theta pi_z = -T/theta theta_z
# so theta_z = theta*g/T/cp
# i.e. theta = theta_0 exp(g*z/T/cp)
Tsurf = fd.Constant(300.)
Nsq = 2.5e-5
thetab = Tsurf*fd.exp(N**2*z/g)

Up = fd.as_vector([fd.Constant(0.0),
                   fd.Constant(0.0),
                   fd.Constant(1.0)]) # up direction

un, rhon, thetan = Un.split()
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)

# set theta_pert
def coth(x):
    return fd.cosh(x)/fd.sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*fd.sqrt((Bu*0.5-fd.tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))

a = -4.5
Bu = 0.5
theta_exp = a*Tsurf/g*fd.sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))
                                 *fd.sinh(Z(z))*fd.cos(fd.pi*(x-L)/L)
                                 - n()*Bu*fd.cosh(Z(z))*fd.sin(fd.pi*(x-L)/L))

# set theta0
thetan.interpolate(thetab)

Pi_ref = fd.Function(V2)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=rhon, pi_boundary=fd.Constant(1),
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi_ref)

thetan.interpolate(thetab + theta_exp)
Pi = fd.Function(V2)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=rhon, pi_boundary=fd.Constant(1),
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi)
rho_back = fd.Function(V2).assign(rhon)

sparameters = {
    "snes_converged_reason": None,    
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "snes_monitor": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-50,
    "ksp_rtol": 1e-6,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm"
}

Pi0_value = fd.assemble(Pi_ref*fd.dx) / fd.assemble(
    fd.Constant(1.0)*fd.dx(domain=mesh))
Pi0 = fd.Constant(Pi0_value)

# set x-cpt of velocity
dbdy = -1.0e-07
dthetady = Tsurf/g*dbdy
u = cp*dthetady/f*(Pi_ref-Pi0)

# get Pi gradient
tg = fd.TrialFunction(V1)
wg = fd.TestFunction(V1)

n = fd.FacetNormal(mesh)

inner =fd.inner; dx = fd.dx; div = fd.div
a = inner(wg, tg)*dx
L = -div(wg)*Pi*dx + inner(wg, n)*Pi*(fd.ds_t + fd.ds_b)
pgrad = fd.Function(V1)
fd.solve(a == L, pgrad)

# get initial v
m = fd.TrialFunction(V2)
phi = fd.TestFunction(V2)

a = phi*f*m*dx
L = phi*cp*thetan*pgrad[0]*dx
v = fd.Function(V2)
fd.solve(a == L, v)

# set initial u
u_exp = fd.as_vector([u, v, 0.])
un.project(u_exp)

#The timestepping solver
un, rhon, thetan = fd.split(Un)
unp1, rhonp1, thetanp1 = fd.split(Unp1)

du, drho, dtheta = fd.TestFunctions(W)
f = fd.Constant(f)

eqn = slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                     du, drho, dtheta,
                     dT=dT, n=n, Up=Up, c_pen=fd.Constant(2.0**(-7./2)),
                     cp=cp, g=g, R_d=R_d, p_0=p_0, kappa=kappa,
                     f = f, Eady={"Pi0":Pi0, "dthetady":dthetady})

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=sparameters,
                                        options_prefix="nsolver")

name = 'eady_comp'
file_eady = fd.File(name+'.pvd')
un, rhon, thetan = Un.split()
delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)

dt = 60
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
Courant.interpolate(Courant_num/Courant_denom)

file_eady.write(un, rhon, thetan, delta_rho, delta_theta, Courant)
Unp1.assign(Un)

t = 0.
day = 60*60*24
tdump = 0.
tmax = 30*day #  30*day
dumpt = 0.25*day

itcount = 0
stepcount = 0

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
        Courant.interpolate(Courant_num/Courant_denom)
        file_eady.write(un, rhon, thetan, delta_rho, delta_theta,
                      Courant)
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount)
