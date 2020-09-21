import firedrake as fd

dt = 6.
dT = fd.Constant(dt)
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
Unp = fd.Function(W)

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
bcs = [DirichletBC(W_h.sub(0), Constant(0.0), bstring)]

wh = Function(W_h)
PiProblem = LinearVariationalProblem(alhs, arhs, wh, bcs=bcs)

params = {'mat_type':'aij',
          'ksp_type': 'preonly',
          'pc_type': 'lu',
          'pc_factor_mat_solver_type','mumps'}

PiSolver = LinearVariationalSolver(PiProblem,
                                   solver_parameters=params,
                                   options_prefix="pisolver")

PiSolver.solve()
v, Pi0 = wh.split()
un, rhon, Pin, thetan = Un.split()
Pin.assign(Pi0)

#get the nonlinear rho
rho = fd.TrialFunction(V2)
rho_bal = Function(V2)
q = fd.TestFunction(V2)
piform = (rho * R_d * thetan / p_0) ** (kappa / (1 - kappa))
rho_eqn = w*(pi - piform)*dx

RhoProblem = NonlinearVariationalProblem(rho_eqn, rho_bal)
RhoSolver = NonlinearVariationalSolver(RhoProblem,
                                       solver_parameters=params,
                                       options_prefix="rhosolver")
RhoSolver.solve()
rhon.assign(rho_bal)

a = fd.Constant(5.0e3)
deltaTheta = fd.Constant(1.0e-2)
theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
thetan.interpolate(thetan + theta_pert)
un.project(as_vector([20.0, 0.0]))

#The timestepping solver
unp1, rhonp1, Pinp1, thetanp1
unph = 0.5*(un + unp1)
thetanph = 0.5*(thetan + thetanp1)
rhonph = 0.5*(rhon + rhonp1)
Pinh = 0.5*(Pin + Pinp1)
un = 0.5*(np.dot(unph, n) + abs(np.dot(unph, n)))


def theta_eqn(q):
    NEEDS SUPG
    return (
        q*(thetanp1 - thetan)*dx -
        dT*div(q*unph)*thetanph*dx +
        dT*jump(q)*(un('+')*thetanph('+')
                    - un('-')*thetanph('-'))*(dS_v + dS_h)
    )


def rho_eqn(q):
    return (
        q*(rhonp1 - rhon)*dx -
        dT*inner(grad(q), unph*thetanph)*dx +
        dT*jump(q)*(un('+')*rhoph('+') - un('-')*rhoph('-'))*(dS_v + dS_h)
    )


def pi_eqn(q):
    return q*(Pinp1 -
              (rhonp1 * R_d * thetanp1 / p_0) ** (kappa / (1 - kappa)))*dx


def u_eqn(w):
    return (
        inner(w, unp1 - un)*dx - dT*cp*div(thetanph*w)*Pinph*dx
        - dT*jump(w*thetanph, n)*avg(Pinph)*dS_v
        


        
DONT FORGET BCS!
    
name = "gw_imp"
file_gw = fd.File(name+'.pvd')
file_gw.write(un, rhon, pin, thetan)
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
        qsolver.solve()
        file_gw.write(un, etan, qn)
        tdump -= dumpt
