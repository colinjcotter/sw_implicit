import firedrake as fd
import numpy as np
from petsc4py import PETSc

dT = fd.Constant(1)
tmax = 3600.

nlayers = 50  # horizontal layers
columns = 150  # number of columns
L = 3.0e5
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
m = fd.PeriodicIntervalMesh(columns, L, distribution_parameters =
                            distribution_parameters)

g = fd.Constant(9.810616)
N = fd.Constant(0.01)  # Brunt-Vaisala frequency (1/s)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)
R_d = fd.Constant(287.)  # Gas constant for dry air (J/kg/K)
kappa = fd.Constant(2.0/7.0)  # R_d/c_p
p_0 = fd.Constant(1000.0*100.0)  # reference pressure (Pa, not hPa)
cv = fd.Constant(717.)  # SHC of dry air at const. volume (J/kg/K)
T_0 = fd.Constant(273.15)  # ref. temperature

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

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
Tsurf = fd.Constant(300.)
thetab = Tsurf*fd.exp(N**2*z/g)

pi_boundary = 1.

Up = fd.as_vector([fd.Constant(0.0), fd.Constant(1.0)])

# Calculate hydrostatic Pi, rho
W_h = Vv * V2

n = fd.FacetNormal(mesh)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)

un, rhon, thetan = Un.split()
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)

wh = fd.Function(W_h)
v, rho = wh.split()
rho.assign(1.0)
v, rho = fd.split(wh)
dv, drho = fd.TestFunctions(W_h)

def pi_formula(rho, theta):
    return (rho * R_d * theta / p_0) ** (kappa / (1 - kappa))

Pi = pi_formula(rho, thetan)

rhoeqn = (
    (cp*fd.inner(v, dv) - cp*fd.div(dv*thetan)*Pi)*fd.dx
    + drho*fd.div(thetan*v)*fd.dx
)

top = False
if top:
    bmeasure = fd.ds_t
    bstring = "bottom"
else:
    bmeasure = fd.ds_b
    bstring = "top"

rhoeqn += cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
rhoeqn += g*fd.inner(dv, Up)*fd.dx
bcs = [fd.DirichletBC(W_h.sub(0), fd.as_vector([fd.Constant(0.0),
                                             fd.Constant(0.0)]), bstring)]

RhoProblem = fd.NonlinearVariationalProblem(rhoeqn, wh, bcs=bcs)

sparameters = {
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

schur_params = {'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',
                'pc_fieldsplit_schur_fact_type': 'full',
                'pc_fieldsplit_schur_precondition': 'selfp',
                'fieldsplit_1_ksp_type': 'preonly',
                'fieldsplit_1_pc_type': 'gamg',
                'fieldsplit_1_mg_levels_pc_type': 'bjacobi',
                'fieldsplit_1_mg_levels_sub_pc_type': 'ilu',
                'fieldsplit_0_ksp_type': 'richardson',
                'fieldsplit_0_ksp_max_it': 4,
                'ksp_atol': 1.e-08,
                'ksp_rtol': 1.e-08}

RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                          solver_parameters=schur_params,
                                          options_prefix="rhosolver")

RhoSolver.solve()
v, Rho0 = wh.split()

rhon.assign(Rho0)
rho_back = fd.Function(V2).assign(Rho0)

a = fd.Constant(5.0e3)
deltaTheta = fd.Constant(1.0e-2)
theta_pert = deltaTheta*fd.sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
thetan.interpolate(thetan + theta_pert)
un.project(fd.as_vector([20.0, 0.0]))

#The timestepping solver
un, rhon, thetan = fd.split(Un)
unp1, rhonp1, thetanp1 = fd.split(Unp1)
unph = 0.5*(un + unp1)
thetanph = 0.5*(thetan + thetanp1)
rhonph = 0.5*(rhon + rhonp1)
Pinph = pi_formula(rhonph, thetanph)

n = fd.FacetNormal(mesh)
Upwind = 0.5*(fd.sign(fd.dot(unph, n))+1)

def theta_tendency(q, u, theta):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        q*fd.inner(u,fd.grad(theta))*fd.dx
        + fd.jump(q)*(unn('+')*theta('+')
                      - unn('-')*theta('-'))*fd.dS_v
        - fd.jump(q*u*theta, n)*fd.dS_v
    )

def theta_eqn(q):
    qsupg = q + fd.Constant(0.5)*dT*fd.inner(unph, Up)*fd.inner(fd.grad(q), Up)
    return (
        qsupg*(thetanp1 - thetan)*fd.dx
        + dT*theta_tendency(qsupg, unph, thetanph)
    )

def rho_tendency(q, rho, u):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        - fd.inner(fd.grad(q), u*rho)*fd.dx +
        fd.jump(q)*(unn('+')*rho('+')
                    - unn('-')*rho('-'))*(fd.dS_v + fd.dS_h)
    )

def rho_eqn(q):
    return (
        q*(rhonp1 - rhon)*fd.dx
        + dT*rho_tendency(q, rhonph, unph)
    )


d = np.sum(mesh.cell_dimension())

def curl0(u):
    """
    Curl function from y-cpt field to x-z field
    """
    if d == 2:
        # equivalent vector is (0, u, 0)

        # |i   j   k  |
        # |d_x 0   d_z| = (- du/dz, 0, du/dx)
        # |0   u   0  |
        return fd.as_vector([-u.dx(1), u.dx(0)])
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError


def curl1(u):
    """
    dual curl function from dim-1 forms to dim-2 forms
    """
    if d == 2:
        # we have vector in x-z plane and return scalar
        # representing y component of the curl

        # |i   j   k   |
        # |d_x 0   d_z | = (0, -du_3/dx + du_1/dz, 0)
        # |u_1 0   u_3 |
        
        return -u[1].dx(0) + u[0].dx(1)
    elif d == 3:
        return fd.curl(u)
    else:
        raise NotImplementedError


def cross1(u, w):
    """
    cross product (slice vector field with slice vector field)
    """
    if d == 2:
        # cross product of two slice vectors goes into y cpt

        # |i   j   k   |
        # |u_1 0   u_3 | = (0, -u_1*w_3 + u_3*w_1, 0)
        # |w_1 0   w_3 |

        return w[0]*u[1] - w[1]*u[0]
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError


def cross0(u, w):
    """
    cross product (slice vector field with out-of-slice vector field)
    """

    # |i   j   k   |
    # |u_1 0   u_3 | = (-w*u_3, 0, w*u_1)
    # |0   w   0   |

    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return fd.as_vector([-w*u[1], w*u[0]])
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError
    

def both(u):
    return 2*fd.avg(u)

    
def u_eqn(w):
    """
    Written in a dimension agnostic way
    """
    K = fd.Constant(0.5)*fd.inner(unph, unph)
    
    return (
        fd.inner(w, unp1 - un)*fd.dx
        + dT*fd.inner(unph, curl0(cross1(unph, w)))*fd.dx
        - dT*fd.inner(both(Upwind*unph),
                      both(cross0(n, cross1(unph, w))))*(fd.dS_h + fd.dS_v)
        - dT*fd.div(w)*K*fd.dx
        - dT*cp*fd.div(thetanph*w)*Pinph*fd.dx
        + dT*cp*fd.jump(w*thetanph, n)*fd.avg(Pinph)*fd.dS_v
        + dT*fd.inner(w, Up)*g*fd.dx
        )

du, drho, dtheta = fd.TestFunctions(W)
eqn = u_eqn(du) + rho_eqn(drho) + theta_eqn(dtheta)

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=sparameters,
                                        options_prefix="nsolver")
    
name = "gw_imp"
file_gw = fd.File(name+'.pvd')
un, rhon, thetan = Un.split()
delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)

dt = 12.
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
tmax = 12.
dumpt = 60.
tdump = 0.

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
