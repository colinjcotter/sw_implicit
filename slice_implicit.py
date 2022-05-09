import firedrake as fd
import numpy as np

dT = fd.Constant(1)
tmax = 3600.

nlayers = 10  # horizontal layers
columns = 150  # number of columns
L = 3.0e5
m = fd.PeriodicIntervalMesh(columns, L)

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

W = V1 * V2 * V2 * Vt #velocity, density, pressure, temperature

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

v, Pi = fd.TrialFunctions(W_h)
dv, dpi = fd.TestFunctions(W_h)

n = fd.FacetNormal(mesh)
cp = fd.Constant(1004.5)  # SHC of dry air at const. pressure (J/kg/K)

un, rhon, Pin, thetan = Un.split()
thetan.interpolate(thetab)
theta_back = fd.Function(Vt).assign(thetan)

alhs = (
    (cp*fd.inner(v, dv) - cp*fd.div(dv*thetan)*Pi)*fd.dx
    + dpi*fd.div(thetan*v)*fd.dx
)

top = False
if top:
    bmeasure = fd.ds_t
    bstring = "bottom"
else:
    bmeasure = fd.ds_b
    bstring = "top"

arhs = -cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
arhs -= g*fd.inner(dv, Up)*fd.dx
bcs = [fd.DirichletBC(W_h.sub(0), fd.as_vector([fd.Constant(0.0),
                                             fd.Constant(0.0)]), bstring)]

wh = fd.Function(W_h)
PiProblem = fd.LinearVariationalProblem(alhs, arhs, wh, bcs=bcs)

lu_params = {
    'snes_monitor':None,
    'mat_type':'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type':'mumps'}

PiSolver = fd.LinearVariationalSolver(PiProblem,
                                   solver_parameters=lu_params,
                                   options_prefix="pisolver")

PiSolver.solve()
v, Pi0 = wh.split()

Pin.assign(Pi0)

#get the nonlinear rho
rho_bal = fd.Function(V2)
q = fd.TestFunction(V2)

rho_bal.project(p_0*Pin**(-kappa/(1-kappa))/thetan/R_d)
piform = (rho_bal * R_d * thetan / p_0) ** (kappa / (1 - kappa))
rho_eqn = q*(Pin - piform)*fd.dx

RhoProblem = fd.NonlinearVariationalProblem(rho_eqn, rho_bal)
RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                       solver_parameters=lu_params,
                                       options_prefix="rhosolver")
RhoSolver.solve()

rhon.assign(rho_bal)
rho_back = fd.Function(V2).assign(rho_bal)

a = fd.Constant(5.0e3)
deltaTheta = fd.Constant(1.0e-2)
theta_pert = deltaTheta*fd.sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
thetan.interpolate(thetan + theta_pert)
#un.project(fd.as_vector([20.0, 0.0]))

#The timestepping solver
unp1, rhonp1, Pinp1, thetanp1 = fd.split(Unp1)
unph = 0.5*(un + unp1)
thetanph = 0.5*(thetan + thetanp1)
rhonph = 0.5*(rhon + rhonp1)
Pinph = 0.5*(Pin + Pinp1)

unn = 0.5*(np.dot(unph, n) + abs(np.dot(unph, n)))
n = fd.FacetNormal(mesh)
Upwind = 0.5*(fd.sign(fd.dot(unph, n))+1)

def theta_eqn(q):
    qsupg = q + fd.Constant(0.5)*dT*fd.inner(unph, Up)*fd.inner(fd.grad(q), Up)
    return (
        qsupg*(thetanp1 - thetan)*fd.dx
        + dT*qsupg*fd.inner(unph,fd.grad(thetanph))*fd.dx
        + dT*fd.jump(qsupg)*(unn('+')*thetanph('+')
                    - unn('-')*thetanph('-'))*fd.dS_v
        - dT*fd.jump(qsupg*unph*thetanph, n)*fd.dS_v
    )


def rho_eqn(q):
    return (
        q*(rhonp1 - rhon)*fd.dx -
        dT*fd.inner(fd.grad(q), unph*rhonph)*fd.dx +
        dT*fd.jump(q)*(unn('+')*rhonph('+')
                       - unn('-')*rhonph('-'))*(fd.dS_v + fd.dS_h)
    )


def pi_eqn(q):
    piform = (rho_bal * R_d * thetanph / p_0) ** (kappa / (1 - kappa))
    return q*(Pinph - piform)*fd.dx



d = np.sum(mesh.cell_dimension())

def perp(u):
    #vertical slice sign convention
    return fd.as_vector([u[1], -u[0]])


def curl0(u):
    """
    Curl function from y-cpt field to x-z field
    """
    if d == 2:
        # equivalent vector is (0, u, 0)

        # |i   j   k  |
        # |d_x 0   d_z| = (- du/dz, 0, du_2/dx)
        # |0   u   0  |
        return -perp(fd.grad(u))
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
        return fd.div(perp(u))
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
        return w[0]*u[1] - w[1]*u[0]
    elif d == 3:
        return fd.cross(u, w)
    else:
        raise NotImplementedError


def cross0(u, w):
    """
    cross product (slice vector field with out-of-slice vector field)
    """
    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return -perp(w*u)
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
        #+ dT*fd.inner(unph, curl0(cross1(unph, w)))*fd.dx
        #- dT*fd.inner(both(Upwind*unph),
        #              both(cross0(n, cross1(unph, w))))*(fd.dS_h + fd.dS_v)
        #- dT*fd.div(w)*K*fd.dx
        - dT*cp*fd.div(thetanph*w)*Pinph*fd.dx
        + dT*cp*fd.jump(w*thetanph, n)*fd.avg(Pinph)*fd.dS_v
        + dT*fd.inner(w, Up)*g*fd.dx
        )

du, drho, dpi, dtheta = fd.TestFunctions(W)
eqn = u_eqn(du) + rho_eqn(drho) + pi_eqn(dpi) + theta_eqn(dtheta)

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=lu_params,
                                        options_prefix="nsolver")
    
name = "gw_imp"
file_gw = fd.File(name+'.pvd')
delta_theta = fd.Function(Vt, name="delta theta").assign(thetan-theta_back)
delta_rho = fd.Function(V2, name="delta rho").assign(rhon-rho_back)
file_gw.write(un, rhon, Pin, thetan, delta_rho, delta_theta)
Unp1.assign(Un)

t = 0.
dt = 6.
dT.assign(dt)
tmax = 3600.
dumpt = 6.
tdump = 0.

print('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        delta_theta.assign(thetan-theta_back)
        delta_rho.assign(rhon-rho_back)
        file_gw.write(un, rhon, Pin, thetan, delta_rho, delta_theta)
        tdump -= dumpt
