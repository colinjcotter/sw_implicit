import firedrake as fd

dT = fd.Constant(0)

nlayers = 10  # horizontal layers
columns = 150  # number of columns
L = 3.0e5
m = fd.PeriodicIntervalMesh(columns, L)

cs = fd.Constant(100.)
f = fd.Constant(1.0)
N = fd.Constant(1.0e-2)
U = fd.Constant(20.)

# build volume mesh
H = 1.0e4  # Height position of the model top
mesh = fd.ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
H = fd.Constant(H)

family = "CG"
horizontal_degree = 1
vertical_degree = 1
S1 = fd.FiniteElement(family, fd.interval, horizontal_degree+1)
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

V1 = fd.FunctionSpace(mesh, V2_elt, name="HDiv")
V2 = fd.FunctionSpace(mesh, V3_elt, name="DG")
Vt = fd.FunctionSpace(mesh, V2t_elt, name="Temperature")
Vv = fd.FunctionSpace(mesh, V2v_elt, name="Vv")

W = V1 * V2 * Vt #velocity, pressure, temperature

Un = fd.Function(W)
Unp1 = fd.Function(W)

x, z = fd.SpatialCoordinate(mesh)

xc = L/2
a = fd.Constant(5000)

un, Pin, bn = Un.split()
bn.interpolate(fd.sin(fd.pi*z/H)/(1+(x-xc)**2/a**2))

#The timestepping solver
un, Pin, bn = fd.split(Un)
unp1, Pinp1, bnp1 = fd.split(Unp1)
unph = 0.5*(un + unp1)
bnph = 0.5*(bn + bnp1)
Pinph = 0.5*(Pin + Pinp1)
Ubar = fd.as_vector([U, 0])
n = fd.FacetNormal(mesh)
unn = 0.5*(fd.dot(Ubar, n) + abs(fd.dot(Ubar, n)))

k = fd.as_vector([0, 1])
def theta_eqn(q):
    return (
        q*(bnp1 - bn)*fd.dx -
        dT*fd.div(q*Ubar)*bnph*fd.dx +
        dT*N**2*q*fd.inner(k, unph)*fd.dx +
        dT*fd.jump(q)*(unn('+')*bnph('+')
        - unn('-')*bnph('-'))*(fd.dS_v + fd.dS_h)
    )

def pi_eqn(q):
    return (
        q*(Pinp1 - Pin)*fd.dx -
        dT*fd.inner(fd.grad(q), Ubar*Pinph)*fd.dx
        + dT*fd.jump(q)*(unn('+')*Pinph('+') - unn('-')*Pinph('-'))*(fd.dS_v + fd.dS_h)
        + cs**2*dT*q*fd.div(unph)*fd.dx
    )

def u_eqn(w):
    return (
        fd.inner(w, unp1 - un)*fd.dx
        - dT*fd.inner(fd.grad(w), fd.outer(unph, Ubar))*fd.dx
        +dT*fd.dot(fd.jump(w), unn('+')*unph('+')
                         - unn('-')*unph('-'))*(fd.dS_v + fd.dS_h)
        -dT*fd.div(w)*Pinph*fd.dx - dT*fd.inner(w, k)*bnph*fd.dx
        )

w, phi, q = fd.TestFunctions(W)
gamma = fd.Constant(1000.0)
eqn = u_eqn(w) + theta_eqn(q) + pi_eqn(phi) + gamma*pi_eqn(fd.div(w))

bcs = [fd.DirichletBC(W.sub(0), 0., "top"),
       fd.DirichletBC(W.sub(0), 0., "bottom")]
nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)
sparams = {'snes_monitor':None,
    'mat_type':'aij',
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps'}
nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=sparams)

name = "gw_imp"
file_gw = fd.File(name+'.pvd')
un, Pin, bn = Un.split()
file_gw.write(un, Pin, bn)
Unp1.assign(Un)

dt = 6.
dumpt = 60.
tdump = 0.
dT.assign(dt)
tmax = 3600.
print('tmax', tmax, 'dt', dt)
t = 0.
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        file_gw.write(un, Pin, bn)
        tdump -= dumpt
