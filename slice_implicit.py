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
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

horizontal_degree = 1
vertical_degree = 1

S1 = FiniteElement(family, interval, horizontal_degree+1)
S2 = FiniteElement("DG", interval, horizontal_degree)

# vertical base spaces
T0 = FiniteElement("CG", interval, vertical_degree+1)
T1 = FiniteElement("DG", interval, vertical_degree)

# build spaces V2, V3, Vt
V2h_elt = HDiv(TensorProductElement(S1, T1))
V2t_elt = TensorProductElement(S2, T0)
V3_elt = TensorProductElement(S2, T1)
V2v_elt = HDiv(V2t_elt)
V2_elt = V2h_elt + V2v_elt

V1 = fd.FunctionSpace("HDiv", mesh, V2_elt)
V2 = fd.FunctionSpace("DG", mesh, V3_elt)
Vt = fd.FunctionSpace("Temperature", mesh, V2t_elt)
Vv = fd.FunctionSpace("Vv", mesh, V2v_elt)

W = V1 * V2 * V2 * Vt #velocity, density, pressure, temperature

Un = fd.Function(W)
Unp1 = fd.Function(W)

x, z = fd.SpatialCoordinate(mesh)

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = fd.Constant(300.)
thetab = Tsurf*exp(N**2*z/g)

pi_boundary = 1.

theta_b = fd.Function(Vt).interpolate(thetab)
rho_b = fd.Function(Vr)

# Calculate hydrostatic Pi, rho
W_h = Vv * V2

v, pi = fd.TrialFunctions(W)
dv, dpi = fd.TestFunctions(W)

n = fd.FacetNormal(state.mesh)
cp = state.parameters.cp

alhs = (
    (cp*fd.inner(v, dv) - cp*fd.div(dv*theta)*pi)*fd.dx
    + dpi*fd.div(theta*v)*fd.dx
)

top = False
if top:
    bmeasure = fd.ds_t
    bstring = "bottom"
else:
    bmeasure = fd.ds_b
    bstring = "top"

arhs = -cp*fd.inner(dv, n)*theta*pi_boundary*bmeasure
bcs = [fd.DirichletBC(W_h.sub(0), Constant(0.0), bstring)]

wh = fd.Function(W_h)
PiProblem = fd.LinearVariationalProblem(alhs, arhs, wh, bcs=bcs)

params = {'mat_type':'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu',
          'pc_factor_mat_solver_type','mumps'}

PiSolver = fd.LinearVariationalSolver(PiProblem,
                                   solver_parameters=params,
                                   options_prefix="pisolver")

PiSolver.solve()
v, Pi0 = wh.split()
un, rhon, Pin, thetan = Un.split()
Pin.assign(Pi0)

#get the nonlinear rho
rho_bal = fd.Function(V2)
q = fd.TestFunction(V2)
piform = (rho_bal * R_d * thetan / p_0) ** (kappa / (1 - kappa))
rho_eqn = q*(pi - piform)*fd.dx

RhoProblem = fd.NonlinearVariationalProblem(rho_eqn, rho_bal)
RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                       solver_parameters=params,
                                       options_prefix="rhosolver")
RhoSolver.solve()
rhon.assign(rho_bal)

a = fd.Constant(5.0e3)
deltaTheta = fd.Constant(1.0e-2)
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
thetan.interpolate(thetan + theta_pert)
un.project(fd.as_vector([20.0, 0.0]))

#The timestepping solver
unp1, rhonp1, pinp1, thetanp1 = fd.split(Unp1)
unph = 0.5*(un + unp1)
thetanph = 0.5*(thetan + thetanp1)
rhonph = 0.5*(rhon + rhonp1)
Pinh = 0.5*(pin + pinp1)

un = 0.5*(np.dot(unph, n) + abs(np.dot(unph, n)))
n = FacetNormal(self.state.mesh)
Upwind = 0.5*(sign(dot(unph, n))+1)
        
Up = as_vector([Constant(1.0), Constant(0.0)])

def theta_eqn(q):
    qsupg = q + Constant(0.5)*dT*inner(unph, up)*inner(grad(theta), up)
    return (
        qsupg*(thetanp1 - thetan)*dx -
        dT*qsupg*inner(unph,grad(thetanph))*dx +
        dT*jump(qsupg)*(un('+')*thetanph('+')
                    - un('-')*thetanph('-'))*dS_v
        - dT*jump(qsupg*un*thetanph)*dS_v
    )


def rho_eqn(q):
    return (
        q*(rhonp1 - rhon)*dx -
        dT*inner(grad(q), unph*thetanph)*dx +
        dT*jump(q)*(un('+')*rhoph('+') - un('-')*rhoph('-'))*(dS_v + dS_h)
    )


def pi_eqn(q):
    piform = (rho_bal * R_d * thetanph / p_0) ** (kappa / (1 - kappa))
    return q*(pinph - piform)*fd.dx



d = mesh.cell_dimension()


def perp(u):
    #vertical slice sign convention
    return as_vector([u[1], -u[0]])


def curl0(u):
    """
    Curl function from dim-2 forms to dim-1 forms
    """
    if d == 2:
        # equivalent vector is (0, u, 0)
        return -perp(grad(u))
    elif d == 3:
        return curl(u)
    else:
        Raise NotImplementedError


def curl1(u):
    """
    dual curl function from dim-1 forms to dim-2 forms
    """
    if d == 2:
        # we have vector in x-z plane and return scalar
        # representing y component of the curl
        return div(perp(u))
    elif d == 3:
        return curl(u)
    else:
        Raise NotImplementedError


def cross1(u, w):
    """
    cross product (slice vector field with slice vector field)
    """
    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return w[0]*u[2] - w[2]*u[0]
    elif d == 3:
        return cross(u, w)
    else:
        raise NotImplementedError


def cross0(u, w):
    """
    cross product (slice vector field with out-of-slice vector field)
    """
    if d == 2:
        # cross product of two slice vectors goes into y cpt
        return w*(-u[2] + u[1])
    elif d == 3:
        return cross(u, w)
    else:
        raise NotImplementedError
    

def both(u):
    return 2*fd.avg(u)

    
def u_eqn(w):
    """
    Written in a dimension agnostic way
    """
    return (
        inner(w, unp1 - un)*dx
        inner(unph, curl0(cross1(unph, w)))*dx
                - inner(both(Upwind*unph),
                        both(cross0(n, cross1(unph, w))))*(dS_h + dS_v)
            )
        dT*cp*div(thetanph*w)*Pinph*dx
        - dT*jump(w*thetanph, n)*avg(Pinph)*dS_v

du, drho, dpi, dtheta = fd.TestFunctions(W)
eqn = u_eqn(du) + rho_eqn(drho) + pi_eqn(dpi) + theta_eqn(dtheta)

bcs = [fd.DirichletBC(W.sub(0), 0., "bottom"),
       fd.DirichletBC(W.sub(0), 0., "top")]

nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)

lu_params = {"mat_type": "aij",
             "ksp_type": "preonly",
             "pc_type": "lu",
             "pc_factor_mat_solver_type": "mumps"}

nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=lu_params)
    
name = "gw_imp"
file_gw = fd.File(name+'.pvd')
file_gw.write(un, rhon, pin, thetan)
Unp1.assign(Un)

t = 0.
dt = 6.
dT.assign(dt)
tmax = 3600.
dumpt = 
tdump = 0.

print('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        file_gw.write(un, rhon, pin, thetan)
        tdump -= dumpt
