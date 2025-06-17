from firedrake import *

mesh = UnitIntervalMesh(1)
V = FunctionSpace(mesh, "DG", 0)
W = V

q_T = Constant(1.0)

q0_v = Function(W)

q1_v = Function(W)

tc = Constant(0.)

dt = 0.05
dtc = Constant(dt)

qsat = Constant(1.2) - sin(tc)

dq_v = TestFunction(W)

eqn = dq_v*(q1_v - qsat)*dx

# variational solver
params = {
    "snes_type": "vinewtonrsls",
    #"snes_vi_monitor": None,
    #"snes_monitor": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

nproblem = NonlinearVariationalProblem(
    eqn, q1_v)
nsolver = NonlinearVariationalSolver(
    nproblem, solver_parameters=params)

# set the bounds

lbound = Function(W).assign(PETSc.NINFINITY)
ubound = Function(W).assign(PETSc.INFINITY)

ubound.assign(q_T)

q0_v.assign(q_T)

T = 10.
t = 0.
q1_v.assign(q0_v)
while t < T - dt:
    nsolver.solve(bounds=(lbound, ubound))
    q0_v.assign(q1_v)
    print("t", t, "q", q0_v.dat.data[:], "qsat", float(qsat))
    t += dt
    tc.assign(t)
