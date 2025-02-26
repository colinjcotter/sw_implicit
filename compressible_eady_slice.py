from firedrake import *
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
parser.add_argument('--dumpt', type=float, default=2, help='Dump time for fields in hours. Default 6.')
parser.add_argument('--checkt', type=float, default=2, help='Create checkpointing file every checkt hours. Default 2.')
parser.add_argument('--diagt', type=float, default=2, help='Dump time for diagnostics in hours. Default 2.')
parser.add_argument('--dt', type=float, default=300, help='Timestep in seconds. Default 300.')
parser.add_argument('--a', type=float, default=-7.5, help='Strength of the initial amplitude. Default -7.5.')
parser.add_argument('--vector_invariant', action="store_true", help='use vector invariant form.')

args = parser.parse_known_args()
args = args[0]
PETSc.Sys.Print(args)

c_pen = Constant(args.c_pen)
tmax = args.tmax*60*60*24
dumpt = args.dumpt*60*60
diagt = args.diagt*60*60
checkt = args.checkt*60.*60.
PETSc.Sys.Print("c_pen = ", c_pen)

dT = Constant(1)

nlayers = args.nlayers
base_columns = args.ncolumns
L = 240e3
distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

L = 1000000.
H = 10000.  # Height position of the model top
f = 1.e-04

m = PeriodicRectangleMesh(base_columns, ny=1, Lx=2*L, Ly=1.0e-5*L,
                             direction="both",
                             quadrilateral=True,
                             distribution_parameters=distribution_parameters)

g = Constant(9.810616)
cp = Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
cv = 717.  # SHC of dry air at const. volume (J/kg/K)
R_d = Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = Constant(2.0/7.0)  # R_d/c_p
p_0 = Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)

# build volume mesh
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
n = FacetNormal(mesh)

horizontal_degree = 1
vertical_degree = 1

S1 = FiniteElement("RTCF", quadrilateral, horizontal_degree+1)
S2 = FiniteElement("DQ", quadrilateral, horizontal_degree)

# vertical base spaces
T0 = FiniteElement("CG", interval, vertical_degree+1)
T1 = FiniteElement("DG", interval, vertical_degree)

# build spaces V2, V3, Vt

V2h_elt = HDiv(TensorProductElement(S1, T1))
V2t_elt = TensorProductElement(S2, T0)
V3_elt = TensorProductElement(S2, T1)
V2v_elt = HDiv(V2t_elt)
V2_elt = V2h_elt + V2v_elt

V1 = FunctionSpace(mesh, V2_elt, name="Velocity")
V2 = FunctionSpace(mesh, V3_elt, name="Pressure")
Vt = FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = FunctionSpace(mesh, V2v_elt, name="Vv")

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

Un = Function(W, name="mixedfunction")
Unp1 = Function(W)

x, y, z = SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
# Isothermal T = theta*pi is constant
# pi = T/theta => pi_z = -T/theta^2 theta_z
# so -g = cp theta pi_z = -T/theta theta_z
# so theta_z = theta*g/T/cp
# i.e. theta = theta_0 exp(g*z/T/cp)
Tsurf = Constant(300.)
Nsq = Constant(2.5e-05)
thetab = Tsurf*exp(Nsq*(z-H/2)/g)

Up = as_vector([Constant(0.0),
                   Constant(0.0),
                   Constant(1.0)]) # up direction

Uns = Un.subfunctions
un, rhon, thetan = tuple(Uns[i] for i in order)
thetan.interpolate(thetab)
theta_back = Function(Vt, name="theta_back").assign(thetan)

# set theta_pert
def coth(x):
    return cosh(x)/sinh(x)


def Z(z):
    return Bu*((z/H)-0.5)


def n():
    return Bu**(-1)*sqrt((Bu*0.5-tanh(Bu*0.5))*(coth(Bu*0.5)-Bu*0.5))

a = args.a
Bu = 0.5
theta_exp = a*Tsurf/g*sqrt(Nsq)*(-(1.-Bu*0.5*coth(Bu*0.5))
                                 *sinh(Z(z))*cos(pi*(x-L)/L)
                                 - n()*Bu*cosh(Z(z))*sin(pi*(x-L)/L))
PETSc.Sys.Print("a = ", a)

# set theta0
thetan.interpolate(thetab)

Pi_ref = Function(V2)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=rhon, pi_boundary=Constant(1),
                    cp=cp, R_d=R_d, p_0=p_0, kappa=kappa, g=g, Up=Up,
                    top=False, Pi=Pi_ref)
rho_back = Function(V2, name="rho_back").assign(rhon)
thetan.interpolate(thetab + theta_exp)
Pi = Function(V2)
hydrostatic_rho(Vv, V2, mesh, thetan, rhon=rhon, pi_boundary=Constant(1),
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

Pi0_value = assemble(Pi_ref*dx) / assemble(
    Constant(1.0)*dx(domain=mesh))
Pi0 = Constant(args.pi0)
PETSc.Sys.Print("Pi0 = ", Pi0_value, "is calculated but, Pi0 = ",
                args.pi0, "is used instead to keep the front less moving.")

# set x-cpt of velocity
dbdy = -1.0e-07
dthetady = Tsurf/g*dbdy
u = cp*dthetady/f*(Pi_ref-Pi0)

# get Pi gradient
tg = TrialFunction(V1)
wg = TestFunction(V1)

n = FacetNormal(mesh)

inner =inner; dx = dx; div = div
a = inner(wg, tg)*dx
L = -div(wg)*Pi*dx + inner(wg, n)*Pi*(ds_t + ds_b)
pgrad = Function(V1)
solve(a == L, pgrad)

# get initial v
m = TrialFunction(V2)
phi = TestFunction(V2)

a = phi*f*m*dx
L = phi*cp*thetan*pgrad[0]*dx
v = Function(V2)
solve(a == L, v)

# set initial u
u_exp = as_vector([u, v, 0.])
un.project(u_exp)

#The timestepping solver
Uns = split(Un)
un, rhon, thetan = tuple(Uns[i] for i in order)

Uns = split(Unp1)
unp1, rhonp1, thetanp1 = tuple(Uns[i] for i in order)

Uns = TestFunctions(W)
du, drho, dtheta = tuple(Uns[i] for i in order)

f = Constant(f)

eqn = slice_imr_form(un, unp1, rhon, rhonp1, thetan, thetanp1,
                     du, drho, dtheta,
                     dT=dT, n=n, Up=Up, c_pen=c_pen,
                     cp=cp, g=g, R_d=R_d, p_0=p_0, kappa=kappa,
                     f = f, Eady={"Pi0":Pi0, "dthetady":dthetady},
                     vector_invariant=args.vector_invariant)

bcs = [DirichletBC(W.sub(0), 0., "bottom"),
       DirichletBC(W.sub(0), 0., "top")]

nprob = NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = NonlinearVariationalSolver(nprob, solver_parameters=sparameters,
                                        options_prefix="nsolver")

name = args.filename
file_eady = VTKFile(name+'.pvd')

Uns = Un.subfunctions
un, rhon, thetan = tuple(Uns[i] for i in order)

delta_theta = Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = Function(V2, name="delta rho").assign(rhon-rho_back)

dt = args.dt
dT.assign(dt)

DG0 = FunctionSpace(mesh, "DG", 0)
One = Function(DG0).assign(1.0)
unn = 0.5*(inner(-un, n) + abs(inner(-un, n))) # gives fluxes *into* cell only
v = TestFunction(DG0)
Courant_num = Function(DG0, name="Courant numerator")
Courant_num_form = dT*(
    both(unn*v)*(dS_v + dS_h)
    + unn*v*ds_tb
)
Courant_denom = Function(DG0, name="Courant denominator")
assemble(One*v*dx, tensor=Courant_denom)
Courant = Function(DG0, name="Courant")

assemble(Courant_num_form, tensor=Courant_num)
Courant.interpolate(Courant_num/Courant_denom)

divu = Function(DG0, name="div(u)")
divu.interpolate(div(un))

# dump initial condition
file_eady.write(un, rhon, thetan, delta_rho, delta_theta, Courant, divu)
Unp1.assign(Un)

t = 0.
tdump = 0.
tdiag = 0.
tcheck = 0.
itcount = 0
stepcount = 0
idx = 0

# calculate maximum v
def get_max_v(un, t, mesh):
    y_vec = as_vector([0., 1., 0.])
    y_vel = inner(un, y_vec)
    DG0 = FunctionSpace(mesh, "DG", 0)
    y_vel_dg0 = Function(DG0).project(y_vel)
    max_v = maximum(y_vel_dg0)
    return max_v

# calculate rms of v
def get_rms_v(un, t, mesh):
    y_vec = as_vector([0., 1., 0.])
    y_vel = inner(un, y_vec)
    DG0 = FunctionSpace(mesh, "DG", 0)
    c = Function(DG0)
    c.assign(1)
    rms_v = sqrt(assemble((dot(y_vel,y_vel))*dx)/assemble(c*dx))
    return rms_v

# calculate v kinetic energy
def get_kinetic_energy_v(un, rhon, t, mesh):
    y_vec = as_vector([0., 1., 0.])
    y_vel = inner(un, y_vec)
    kineticv_form = 0.5*rhon*dot(y_vel,y_vel)
    kineticv = assemble(kineticv_form*dx)
    return kineticv

# calculate uw kinetic energy
def get_kinetic_energy_uw(un, rhon, t, mesh):
    u_vec = as_vector([1., 0., 0.])
    u_vel = inner(un, u_vec)
    w_vec = as_vector([0., 0., 1.])
    w_vel = inner(un, w_vec)
    kineticuw_form = 0.5*rhon*(dot(u_vel,u_vel)+dot(w_vel,w_vel))
    kineticuw = assemble(kineticuw_form*dx)
    return kineticuw

# calculate potential energy
def get_potential_energy(rhon, thetan, Pi0, R_d, p_0, kappa, g, mesh):
    x, y, z = SpatialCoordinate(mesh)
    Pi = pi_formula(rhon, thetan, R_d, p_0, kappa)
    potential_form = rhon*(g*z + cv*Pi*thetan - cp*Pi0*thetan)
    potential = assemble(potential_form*dx)
    return potential

# calculate rms of div(u)
def get_rms_divu(un, t, mesh):
    DG0 = FunctionSpace(mesh, "DG", 0)
    c = Function(DG0)
    c.assign(1)
    rms_divu = sqrt(assemble((dot(div(un),div(un)))*dx)/assemble(c*dx))
    return rms_divu

# store diagnostics at the initial stage
max_v_list = []
max_v = get_max_v(un, t=t, mesh=mesh)
max_v_list.append(max_v)
PETSc.Sys.Print("max_v =", max_v_list)

rms_v_list = []
rms_v = get_rms_v(un, t=t, mesh=mesh)
rms_v_list.append(rms_v)
PETSc.Sys.Print("rms_v =", rms_v_list)

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

rms_divu_list = []
rms_divu = get_rms_divu(un, t=t, mesh=mesh)
rms_divu_list.append(rms_divu)
PETSc.Sys.Print("rms_divu =", rms_divu_list)


# create checkpoint
thours = int(t/3600)
chkfile = DumbCheckpoint(name+"_"+str(thours)+"h", mode=FILE_CREATE)
chkfile.store(Un)
chkfile.store(theta_back)
chkfile.store(rho_back)
chkfile.write_attribute("/", "time", t)
chkfile.write_attribute("/", "tdump", tdump)
chkfile.write_attribute("/", "tcheck", tcheck)
chkfile.close()
PETSc.Sys.Print("Checkpointed at t = ", t)

# for recording rmsv
max_rmsv = 0
time_at_max_rmsv = 0

# time loop
PETSc.Sys.Print('tmax', tmax, 'dt', dt)
while t < tmax - 0.5*dt:
    PETSc.Sys.Print(t)
    t += dt
    tdump += dt
    tdiag += dt
    tcheck += dt

    nsolver.solve()
    Un.assign(Unp1)

    # calculate and store max rmsv
    rmsv = get_rms_v(un, t=t, mesh=mesh)
    if rmsv > max_rmsv:
        max_rmsv = rmsv
        time_at_max_rmsv = t

    if tdump > dumpt - dt*0.5:
        delta_theta.assign(thetan-theta_back)
        delta_rho.assign(rhon-rho_back)

        assemble(Courant_num_form, tensor=Courant_num)
        Courant.interpolate(Courant_num/Courant_denom)

        divu.interpolate(div(un))

        file_eady.write(un, rhon, thetan, delta_rho, delta_theta,
                        Courant, divu)
        tdump -= dumpt

    #checkpointing every tcheck hours
    if tcheck > checkt - dt*0.5:
        tcheck -= checkt
        idx += 1
        thours = int(t/3600)
        chkfile = DumbCheckpoint(name+"_"+str(thours)+"h", mode=FILE_CREATE)
        chkfile.store(Un)
        chkfile.store(theta_back)
        chkfile.store(rho_back)
        chkfile.write_attribute("/", "time", t)
        chkfile.write_attribute("/", "tdump", tdump)
        chkfile.write_attribute("/", "tcheck", tcheck)
        chkfile.write_attribute("/", "max_rmsv", max_rmsv)
        chkfile.write_attribute("/", "time_at_max_rmsv", time_at_max_rmsv)
        chkfile.write_attribute("/", "max_v_list", max_v_list)
        chkfile.write_attribute("/", "rms_v_list", rms_v_list)
        chkfile.write_attribute("/", "kineticv_list", kineticv_list)
        chkfile.write_attribute("/", "kineticuw_list", kineticuw_list)
        chkfile.write_attribute("/", "potential_list", potential_list)
        chkfile.write_attribute("/", "total_energy_list", total_energy_list)
        chkfile.write_attribute("/", "rms_divu_list", rms_divu_list)
        chkfile.close()
        PETSc.Sys.Print("Checkpointed at t = ", t)

    if tdiag > diagt - dt*0.5:
        # calculate and store max_v
        max_v = get_max_v(un, t=t, mesh=mesh)
        max_v_list.append(max_v)
        PETSc.Sys.Print("max_v =", max_v_list)
        # calculate and store rms_v
        rms_v = get_rms_v(un, t=t, mesh=mesh)
        rms_v_list.append(rms_v)
        PETSc.Sys.Print("rms_v =", rms_v_list)
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
        # calculate and store rms_divu
        rms_divu = get_rms_divu(un, t=t, mesh=mesh)
        rms_divu_list.append(rms_divu)
        PETSc.Sys.Print("rms_divu =", rms_divu_list)

        tdiag -= diagt

    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()

PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount)
PETSc.Sys.Print("Completed calculation at t = ", t/3600, "hours")

# #pickup the final checkpoint for debug
# history = [0,2,4]

# for i in history:
#     chkfile = DumbCheckpoint(name+"_"+str(i)+"h", mode=FILE_READ)
#     chkfile.load(Un, name="mixedfunction")
#     chkfile.load(theta_back, name="theta_back")
#     chkfile.load(rho_back, name="rho_back")
#     t = chkfile.read_attribute("/", "time")
#     tdump = chkfile.read_attribute("/", "tdump")
#     tcheck = chkfile.read_attribute("/", "tcheck")
#     chkfile.close()
#     PETSc.Sys.Print("Picked up at t = ", t, ", tdump = ", tdump, ", tcheck = ", tcheck)

#     Uns = Un.subfunctions
#     un, rhon, thetan = tuple(Uns[i] for i in order)
#     delta_theta.assign(thetan-theta_back)
#     delta_rho.assign(rhon-rho_back)
#     unc = assemble(dot(un,un)*dx)
#     thetac = assemble(dot(delta_theta,delta_theta)*dx)
#     rhoc = assemble(dot(delta_rho,delta_rho)*dx)
#     PETSc.Sys.Print("unc = ", unc, " at t = ", t)
#     PETSc.Sys.Print("thetac = ", thetac, " at t = ", t)
#     PETSc.Sys.Print("rhoc = ", rhoc, " at t = ", t)
