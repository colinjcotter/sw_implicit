import firedrake as fd

dt = 6.
dT = fd.Constant(dt)
tmax = 3600.

nlayers = 10  # horizontal layers
columns = 150  # number of columns
L = 3.0e5
m = fd.PeriodicIntervalMesh(columns, L)

cs = fd.Constant(100.)
f = fd.Constant(1.0)
N = fd.Constant(1.0e-2)
U = fd.Constant(20)

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

V1 = fd.FunctionSpace("HDiv", mesh, V2_elt)
V2 = fd.FunctionSpace("DG", mesh, V3_elt)
Vt = fd.FunctionSpace("Temperature", mesh, V2t_elt)
Vv = fd.FunctionSpace("Vv", mesh, V2v_elt)

W = V1 * V2 * Vt #velocity, pressure, temperature

Un = fd.Function(W)
Unp = fd.Function(W)

x, z = fd.SpatialCoordinate(mesh)

xc = L/2
a = Constant(5000)

un, Pin, bn = Un.split()
bn.interpolate(sin(pi*z/H)/(1+(x-xc)**2/a**2))

#The timestepping solver
un, Pin, bn = split(Un)
unp1, Pinp1, bnp1 = split(Unp)
unph = 0.5*(un + unp1)
bnph = 0.5*(bn + bnp1)
Pinh = 0.5*(Pin + Pinp1)
Ubar = as_vector([U, 0])
un = 0.5*(np.dot(Ubar, n) + abs(np.dot(Ubar, n)))

k = as_vector([0, 1])
def theta_eqn(q):
    return (
        q*(bnp1 - bn)*dx -
        dT*div(q*ubar)*bph*dx +
        dT*N**2*q*inner(k, unph)*dx +
        dT*jump(q)*(un('+')*bnph('+')
        - un('-')*bnph('-'))*(dS_v + dS_h)
    )

def pi_eqn(q):
    return (
        q*(Pinp1 - Pin)*dx -
        dT*inner(grad(q), Ubar*Piph)*dx
        + dT*jump(q)*(un('+')*Piph('+') - un('-')*Piph('-'))*(dS_v + dS_h)
        + cs**2*dT*q*div(unph)*dx
    )

def u_eqn(w):
    return (
        inner(w, unp1 - un)*dx -
        dT*inner(grad(w), outer(unph, Ubar))*dx +
        dT*dot(jump(w), (un('+')*bnph('+')
                         - un('-')*bnph('-'))*(dS_v + dS_h)) +
        dT*div(w)*Piph*dx - dT*inner(w, k)*bnh*dx
        )

w, phi, q = TestFunctions(W)
gamma = fd.Constant(1000.0)
eqn = u_eqn(w) + theta_eqn(q) + pi_eqn(phi) + gamma*pi_eqn(div(w))

bcs = [DirichletBC(W.sub(0), 0., "top"),
       DirichletBC(W.sub(0), 0., "bottom")]
nprob = fd.NonlinearVariationalProblem(eqn, Unp1, bcs=bcs)
sparams = {'ksp_type':'preonly',
           'pc_type':'lu',
           'pc_factor_mat_solver_type':'mumps'}
nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=sparams)

name = "gw_imp"
file_gw = fd.File(name+'.pvd')
un, Pin, bn = Un.split()
file_gw.write(un, Pin, bn)
Unp1.assign(Un)

print('tmax', tmax, 'dt', dt)
while t < tmax + 0.5*dt:
    print(t)
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        qsolver.solve()
        file_gw.write(un, Pin, bn)
        tdump -= dumpt
