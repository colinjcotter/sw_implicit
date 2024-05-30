import firedrake as fd
from petsc4py import PETSc
from slice_utils import hydrostatic_rho, pi_formula,\
    slice_imr_form, both, maximum, minimum
from firedrake.output import VTKFile
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Compressible Eady test case.')
parser.add_argument('--nlayers', type=int, default=30, help='Number of layers. Default 30.')
parser.add_argument('--ncolumns', type=int, default=30, help='Number of columns. Default 30.')
parser.add_argument('--c_pen', type=float, default=0., help='Diffusion coeff. Default 0.')
parser.add_argument('--pi0', type=float, default=0.864, help='Pi0 value. Default 0.864')
parser.add_argument('--filename', type=str, default='comp_eady', help='filename for pvd')
parser.add_argument('--tmax', type=float, default=35, help='Final time in days. Default 35.')
parser.add_argument('--dumpt', type=float, default=6, help='Dump time for fields in hours. Default 6.')
parser.add_argument('--diagt', type=float, default=2, help='Dump time for diagnostics in hours. Default 2.')
parser.add_argument('--dt', type=float, default=300, help='Timestep in seconds. Default 300.')
parser.add_argument('--a', type=float, default=-7.5, help='Strength of the initial amplitude. Default -7.5.')

args = parser.parse_known_args()
args = args[0]
PETSc.Sys.Print(args)

c_pen = fd.Constant(args.c_pen)
tmax = args.tmax*60*60*24
dumpt = args.dumpt*60*60
diagt = args.diagt*60*60
PETSc.Sys.Print("c_pen = ", c_pen)

dT = fd.Constant(1)

nlayers = args.nlayers
base_columns = args.ncolumns
L = 240e3
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

L = 1000000.
H = 10000.  # Height position of the model top
f = 1.e-04

m = fd.PeriodicRectangleMesh(base_columns, ny=1, Lx=2*L, Ly=1.0e-5*L,
                             direction="both",
                             quadrilateral=True,
                             distribution_parameters=distribution_parameters)

g = fd.Constant(9.810616)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
cv = 717.  # SHC of dry air at const. volume (J/kg/K)
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

ordering = "original"

# order is based on the original order V1 * V2 * Vt
# it gives the order to recover original order
if ordering == "tup":
    W = Vt * V1 * V2  #velocity, density, temperature
    order = (1, 2, 0)
if ordering == "put":
    W = V2 * V1 * Vt
    order = (1, 0, 2)
if ordering == "ptu":
    W = V2 * Vt * V1
    order = (2, 0, 1)
else:
    W = V1 * V2 * Vt
    order = (0,1,2)

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
Nsq = fd.Constant(2.5e-05)
thetab = Tsurf*fd.exp(Nsq*(z-H/2)/g)

Up = fd.as_vector([fd.Constant(0.0),
                   fd.Constant(0.0),
                   fd.Constant(1.0)]) # up direction

Uns = Un.subfunctions
un, rhon, thetan = tuple(Uns[i] for i in order)
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)

# set theta_pert
def coth(x):
    return fd.cosh(x)/fd.sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*fd.sqrt((Bu*0.5-fd.tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))

a = args.a
Bu = 0.5
theta_exp = a*Tsurf/g*fd.sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))
                                 *fd.sinh(Z(z))*fd.cos(fd.pi*(x-L)/L)
                                 - n()*Bu*fd.cosh(Z(z))*fd.sin(fd.pi*(x-L)/L))
PETSc.Sys.Print("a = ", a)

# set theta0
thetan.interpolate(thetab)

Pi_ref = fd.Function(V2)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=rhon, pi_boundary=fd.Constant(1),
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi_ref)
rho_back = fd.Function(V2).assign(rhon)
thetan.interpolate(thetab + theta_exp)
Pi = fd.Function(V2)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=rhon, pi_boundary=fd.Constant(1),
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi)

sparameters = {
    "snes_converged_reason": None,
    "snes_lag_jacobian" : 10,
    #"snes_lag_jacobian_persists" : None,
    "snes_lag_preconditioner" : 10,
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "snes_monitor": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-50,
    "ksp_rtol": 1e-6,
    #"ksp_view": None,
    "ksp_max_it": 400,
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "python",
    "assembled_pc_python_type": "firedrake.ASMStarPC",
    "assembled_pc_star_sub_sub_pc_type": "lu",
    "assembled_pc_star_sub_sub_ksp_type": "preonly",
    "assembled_pc_star_construct_dim": 0,
    "assembled_pc_star_sub_sub_pc_factor_mat_ordering_type": "rcm"
}

Pi0_value = fd.assemble(Pi_ref*fd.dx) / fd.assemble(
    fd.Constant(1.0)*fd.dx(domain=mesh))
Pi0 = fd.Constant(args.pi0)
PETSc.Sys.Print("Pi0 = ", Pi0_value, "is calculated but, Pi0 = ",
                args.pi0, "is used instead to keep the front less moving.")

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
Uns = fd.split(Un)
un, rhon, thetan = tuple(Uns[i] for i in order)

Uns = fd.split(Unp1)
unp1, rhonp1, thetanp1 = tuple(Uns[i] for i in order)

Uns = fd.TestFunctions(W)
du, drho, dtheta = tuple(Uns[i] for i in order)

f = fd.Constant(f)

eqn = slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                     du, drho, dtheta,
                     dT=dT, n=n, Up=Up, c_pen=c_pen,
                     cp=cp, g=g, R_d=R_d, p_0=p_0, kappa=kappa,
                     f = f, Eady={"Pi0":Pi0, "dthetady":dthetady},
                     vector_invariant=False)

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=sparameters,
                                        options_prefix="nsolver")

name = args.filename
file_eady = VTKFile(name+'.pvd')

Uns = Un.subfunctions
un, rhon, thetan = tuple(Uns[i] for i in order)

delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)

dt = args.dt
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
tdump = 0.
tdiag = 0.
itcount = 0
stepcount = 0

# calculate rms of v
def get_rmsv(un, t, mesh):
    y_vec = fd.as_vector([0., 1., 0.])
    y_vel = fd.inner(un, y_vec)
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    c = fd.Function(DG0)
    c.assign(1)
    rmsv = fd.sqrt(fd.assemble((fd.dot(y_vel,y_vel))*dx)/fd.assemble(c*dx))
    PETSc.Sys.Print("rmsv = ", rmsv, "t = ", t)
    return rmsv

# calculate v kinetic energy
def get_kinetic_energy_v(un, rhon, t, mesh):
    y_vec = fd.as_vector([0., 1., 0.])
    y_vel = fd.inner(un, y_vec)
    kineticv_form = 0.5*rhon*fd.dot(y_vel,y_vel)
    kineticv = fd.assemble(kineticv_form*dx)
    return kineticv

def get_kinetic_energy_uw(un, rhon, t, mesh):
    u_vec = fd.as_vector([1., 0., 0.])
    u_vel = fd.inner(un, u_vec)
    w_vec = fd.as_vector([0., 0., 1.])
    w_vel = fd.inner(un, w_vec)
    kineticuw_form = 0.5*rhon*(fd.dot(u_vel,u_vel)+fd.dot(w_vel,w_vel))
    kineticuw = fd.assemble(kineticuw_form*dx)
    return kineticuw

def get_potential_energy(rhon, thetan, Pi0, R_d, p_0, kappa, g, mesh):
    x, y, z = fd.SpatialCoordinate(mesh)
    Pi = pi_formula(rhon, thetan, R_d, p_0, kappa)
    potential_form = rhon*(g*z + cv*Pi*thetan - cp*Pi0*thetan)
    potential = fd.assemble(potential_form*dx)
    return potential

# store diagnostics at the initial stage
rmsv_list = []
rmsv = get_rmsv(un, t=t, mesh=mesh)
rmsv_list.append(rmsv)
PETSc.Sys.Print("rmsv =", rmsv_list)

kineticv_list = []
kineticv_ini = get_kinetic_energy_v(un, rhon, t=t, mesh=mesh)
kineticv_list.append(kineticv_ini-kineticv_ini)
PETSc.Sys.Print("kineticv_diff =", kineticv_list)

kineticuw_list = []
kineticuw_ini = get_kinetic_energy_uw(un, rhon, t=t, mesh=mesh)
kineticuw_list.append(kineticuw_ini-kineticuw_ini)
PETSc.Sys.Print("kineticuw_diff =", kineticuw_list)

potential_list = []
potential_ini = get_potential_energy(rhon, thetan, Pi0=Pi0, R_d=R_d, p_0=p_0, kappa=kappa, g=g, mesh=mesh)
potential_list.append(potential_ini-potential_ini)
PETSc.Sys.Print("potential_diff =", potential_list)

total_energy_list = []
total_energy_ini = kineticv_ini + kineticuw_ini + potential_ini
total_energy_list.append(total_energy_ini-total_energy_ini)
PETSc.Sys.Print("total_diff =", total_energy_list)

# time loop
PETSc.Sys.Print('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    PETSc.Sys.Print(t)
    t += dt
    tdump += dt
    tdiag += dt

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

    if tdiag > diagt - dt*0.5:
        # calculate and store rmsv
        rmsv = get_rmsv(un, t=t, mesh=mesh)
        rmsv_list.append(rmsv)
        PETSc.Sys.Print("rmsv =", rmsv_list)
        # calculate and store kineticv
        kineticv = get_kinetic_energy_v(un, rhon, t=t, mesh=mesh)
        kineticv_list.append(kineticv-kineticv_ini)
        PETSc.Sys.Print("kineticv_diff =", kineticv_list)
        # calculate and store kineticuw
        kineticuw = get_kinetic_energy_uw(un, rhon, t=t, mesh=mesh)
        kineticuw_list.append(kineticuw-kineticuw_ini)
        PETSc.Sys.Print("kineticuw_diff =", kineticuw_list)
        # calculate and store potential energy
        potential = get_potential_energy(rhon, thetan, Pi0=Pi0, R_d=R_d, p_0=p_0, kappa=kappa, g=g, mesh=mesh)
        potential_list.append(potential-potential_ini)
        PETSc.Sys.Print("potential_diff =", potential_list)
        # calculate and store total energy
        total_energy = kineticv + kineticuw + potential
        total_energy_list.append(total_energy-total_energy_ini)
        PETSc.Sys.Print("total_diff =", total_energy_list)

        tdiag -= diagt

    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()

PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount)
